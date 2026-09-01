"""
1-D CNN for MIT-BIH beat classification (5-class AAMI)
=====================================================
Input  : one beat, 256 samples (~0.7 s @ 360 Hz), Z-score normalized.
Output : 5 logits (N / S / V / F / Q).

Design rationale
----------------
The task is *morphology* classification of a single beat, not rhythm over a
long window. So the receptive field only needs to span one QRS complex plus
its immediate P/T context. We use a small residual 1D-CNN:

  - First conv kernel = 7 (~19 ms): wide enough to see the QRS upstroke/
    downstroke as one feature, narrow enough not to blur P vs QRS vs T.
  - Later kernels = 5 (~14 ms): refine local morphology after pooling.
  - One residual block keeps gradients clean at depth without extra width.
  - GAP → FC: shift-invariant to small R-peak misalignment, and cheap.

Parameter budget ≤ 3000 (hardware constraint). BatchNorm params are counted.

  Conv1 (1→8,  k7)            : 8*7 + 8                       =   64
  bn1                          : 8*2                          =   16
  Conv2 (8→16, k5)            : 8*16*5 + 16                   =  656
  bn2                          : 16*2                         =   32
  Conv3 (16→16,k5) [res]      : 16*16*5 + 16                  = 1296
  bn3                          : 16*2                         =   32
  FC ((16+4)→5)                : 20*5 + 5                     =  105
  ----------------------------------------------------------------------
  total                                                       = 2201

RR features (4): pre-RR, post-RR, pre/post ratio, RR/local-median — all
heart-rate-normalized. These carry the *rhythm* cue (a beat arriving early)
that single-beat morphology cannot express, which is exactly what separates
supraventricular ectopics (S) from normal (N). They are concatenated to the
GAP vector before the FC layer (de Chazal-style morphology+rhythm fusion).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """Conv-BN-ReLU with an identity skip (same channels, same length)."""

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=pad, bias=True)
        self.bn   = nn.BatchNorm1d(channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)) + x)


class ECG_BeatCNN(nn.Module):
    """
    Compact residual 1D-CNN for 5-class MIT-BIH beat classification.

    Input : (B, 256) or (B, 1, 256)
    Output: (B, 5) logits
    """

    def __init__(self, num_classes=5, input_length=256, n_rr=4):
        super().__init__()
        self.num_classes  = num_classes
        self.input_length = input_length
        self.n_rr         = n_rr

        # Stem: 1→8, wide kernel to capture QRS shape, /2 pool.
        self.conv1 = nn.Conv1d(1, 8, kernel_size=7, padding=3, bias=True)
        self.bn1   = nn.BatchNorm1d(8)
        self.pool1 = nn.MaxPool1d(2)                 # 256 → 128

        # 8→16, /4 pool.
        self.conv2 = nn.Conv1d(8, 16, kernel_size=5, padding=2, bias=True)
        self.bn2   = nn.BatchNorm1d(16)
        self.pool2 = nn.MaxPool1d(4)                 # 128 → 32

        # Residual refinement at 16 ch.
        self.res   = ResidualBlock1D(16, kernel_size=5)

        # GAP → concat RR features → FC.
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(16 + n_rr, num_classes, bias=True)

    def forward(self, x, rr=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)                       # (B, L) → (B, 1, L)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))   # (B, 8, 128)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))   # (B, 16, 32)
        x = self.res(x)                                   # (B, 16, 32)
        x = self.gap(x).squeeze(-1)                       # (B, 16)

        if rr is None:                                    # allow morphology-only
            rr = x.new_zeros(x.size(0), self.n_rr)
        rr = rr[:, :self.n_rr]                             # use first n_rr feats
        x = torch.cat([x, rr], dim=1)                     # (B, 16+n_rr)
        return self.fc(x)                                 # (B, 5)

    def predict(self, x, rr=None):
        return self.forward(x, rr).argmax(dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_summary(self):
        lines = [f"{'Layer':<24}{'Params':>10}", "=" * 34]
        total = 0
        for name, mod in [('conv1', self.conv1), ('bn1', self.bn1),
                          ('conv2', self.conv2), ('bn2', self.bn2),
                          ('res.conv', self.res.conv), ('res.bn', self.res.bn),
                          ('fc', self.fc)]:
            p = sum(x.numel() for x in mod.parameters())
            total += p
            lines.append(f"{name:<24}{p:>10}")
        lines += ["=" * 34, f"{'TOTAL':<24}{total:>10}"]
        return "\n".join(lines)


class InceptionBlock1D(nn.Module):
    """
    Parallel k3/k5/k7 conv branches over the same input, concatenated.

    Multi-scale receptive fields capture both narrow QRS complexes (e.g. the
    sharp upstroke of ventricular beats) and wider morphology in one layer —
    a single fixed kernel must trade these off.
    """

    def __init__(self, in_ch, br_ch):
        super().__init__()
        self.b3 = nn.Conv1d(in_ch, br_ch, 3, padding=1, bias=True)
        self.b5 = nn.Conv1d(in_ch, br_ch, 5, padding=2, bias=True)
        self.b7 = nn.Conv1d(in_ch, br_ch, 7, padding=3, bias=True)
        self.bn = nn.BatchNorm1d(br_ch * 3)

    def forward(self, x):
        x = torch.cat([self.b3(x), self.b5(x), self.b7(x)], dim=1)
        return F.relu(self.bn(x))


class ECG_BeatInception(nn.Module):
    """
    Multi-scale (Inception) 1D-CNN for 5-class MIT-BIH beats. RR fused at FC.

    Stem conv → Inception(k3/k5/k7) → GAP → concat RR → FC.  ≤ 3000 params.
      conv1 (1->6, k7)      : 6*7+6              =   48
      bn1                    : 6*2                =   12
      inc.b3 (6->5,k3)       : 6*5*3+5           =   95
      inc.b5 (6->5,k5)       : 6*5*5+5           =  155
      inc.b7 (6->5,k7)       : 6*5*7+5           =  215
      inc.bn (15)            : 15*2              =   30
      fc ((15+n_rr)->5)      : (15+n_rr)*5+5            (n_rr=4 -> 100)
      ------------------------------------------------------------
      total (n_rr=4)                             =  655   (well under budget)
    """

    def __init__(self, num_classes=5, input_length=256, n_rr=4):
        super().__init__()
        self.num_classes  = num_classes
        self.input_length = input_length
        self.n_rr         = n_rr

        self.conv1 = nn.Conv1d(1, 6, kernel_size=7, padding=3, bias=True)
        self.bn1   = nn.BatchNorm1d(6)
        self.pool1 = nn.MaxPool1d(4)                 # 256 → 64
        self.inc   = InceptionBlock1D(6, br_ch=5)    # → 15 ch
        self.pool2 = nn.MaxPool1d(4)                 # 64 → 16
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(15 + n_rr, num_classes, bias=True)

    def forward(self, x, rr=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))   # (B, 6, 64)
        x = self.pool2(self.inc(x))                       # (B, 15, 16)
        x = self.gap(x).squeeze(-1)                       # (B, 15)
        if rr is None:
            rr = x.new_zeros(x.size(0), self.n_rr)
        rr = rr[:, :self.n_rr]
        return self.fc(torch.cat([x, rr], dim=1))

    def predict(self, x, rr=None):
        return self.forward(x, rr).argmax(dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_summary(self):
        return f"ECG_BeatInception  total params: {self.count_parameters()}"


class ECG_BeatThesis(nn.Module):
    """
    The Thesis Chapman CNN (4 conv, 1->4->8->8->16, K=5, pad=2; ReLU only
    after Conv4; GAP -> FC) ported to 256-sample MIT-BIH beats, with RR fused.

    The original applies MaxPool/5 after every conv for a 2500-sample rhythm
    window. A beat is ~10x shorter, so /5 four times would collapse the length
    to 0. Per the chosen port: keep all four convs, drop Pool1 and Pool3, keep
    Pool2 and Pool4 (/5). Length: 256 -c1-> 256 -c2-> 256 -p2/5-> 51 -c3-> 51
    -c4-> 51 -p4/5-> 10 -> GAP -> 16.

      conv1 (1->4,  k5)     : 1*4*5 + 4          =   24
      conv2 (4->8,  k5)     : 4*8*5 + 8          =  168
      conv3 (8->8,  k5)     : 8*8*5 + 8          =  328
      conv4 (8->16, k5)     : 8*16*5 + 16        =  656
      fc ((16+n_rr)->5)     : (16+n_rr)*5 + 5         (n_rr=8 -> 125)
      ------------------------------------------------------------
      total (n_rr=8)                             = 1301
    """

    def __init__(self, num_classes=5, input_length=256, n_rr=8):
        super().__init__()
        self.num_classes  = num_classes
        self.input_length = input_length
        self.n_rr         = n_rr

        self.conv1 = nn.Conv1d(1,  4,  kernel_size=5, padding=2, bias=True)
        self.conv2 = nn.Conv1d(4,  8,  kernel_size=5, padding=2, bias=True)
        self.pool2 = nn.MaxPool1d(5)                  # 256 → 51
        self.conv3 = nn.Conv1d(8,  8,  kernel_size=5, padding=2, bias=True)
        self.conv4 = nn.Conv1d(8,  16, kernel_size=5, padding=2, bias=True)
        self.pool4 = nn.MaxPool1d(5)                  # 51 → 10
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(16 + n_rr, num_classes, bias=True)

    def forward(self, x, rr=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv1(x)                             # (B, 4, 256) [no pool, no ReLU]
        x = self.pool2(self.conv2(x))                 # (B, 8, 51)  [no ReLU]
        x = self.conv3(x)                             # (B, 8, 51)  [no pool, no ReLU]
        x = self.pool4(F.relu(self.conv4(x)))         # (B, 16, 10) [ReLU after Conv4]
        x = self.gap(x).squeeze(-1)                   # (B, 16)
        if rr is None:
            rr = x.new_zeros(x.size(0), self.n_rr)
        rr = rr[:, :self.n_rr]
        return self.fc(torch.cat([x, rr], dim=1))

    def predict(self, x, rr=None):
        return self.forward(x, rr).argmax(dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_summary(self):
        return f"ECG_BeatThesis  total params: {self.count_parameters()}"


class ECG_BeatDeep(nn.Module):
    """
    Deeper residual 1D-CNN for 5-class beats, ~15k-param budget. RR fused.

    Wider channels (16->32->48) and two residual blocks give the capacity the
    3k-budget models lacked for the morphology-bound classes (V, and as far as
    data allows, F). BatchNorm + skips keep it trainable. Still lightweight:
    ~14k INT8 weights ~= 14 KB, trivially fits FPGA/MCU on-chip memory.

    256 -c1-> 256 -p/2-> 128 -c2-> 128 -res24-> -p/4-> 32 -c3-> 32 -res40->
        -p/4-> 8 -GAP-> 40 -concat RR(8)-> FC.
    """

    def __init__(self, num_classes=5, input_length=256, n_rr=8):
        super().__init__()
        self.num_classes  = num_classes
        self.input_length = input_length
        self.n_rr         = n_rr

        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3, bias=True)
        self.bn1   = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)                  # 256 → 128
        self.conv2 = nn.Conv1d(16, 24, kernel_size=5, padding=2, bias=True)
        self.bn2   = nn.BatchNorm1d(24)
        self.res2  = ResidualBlock1D(24, kernel_size=3)
        self.pool2 = nn.MaxPool1d(4)                  # 128 → 32
        self.conv3 = nn.Conv1d(24, 40, kernel_size=5, padding=2, bias=True)
        self.bn3   = nn.BatchNorm1d(40)
        self.res3  = ResidualBlock1D(40, kernel_size=3)
        self.pool3 = nn.MaxPool1d(4)                  # 32 → 8
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(40 + n_rr, num_classes, bias=True)

    def forward(self, x, rr=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))   # (B,16,128)
        x = self.res2(F.relu(self.bn2(self.conv2(x))))    # (B,24,128)
        x = self.pool2(x)                                 # (B,24,32)
        x = self.res3(F.relu(self.bn3(self.conv3(x))))    # (B,40,32)
        x = self.pool3(x)                                 # (B,40,8)
        x = self.gap(x).squeeze(-1)                       # (B,40)
        if rr is None:
            rr = x.new_zeros(x.size(0), self.n_rr)
        rr = rr[:, :self.n_rr]
        return self.fc(torch.cat([x, rr], dim=1))

    def predict(self, x, rr=None):
        return self.forward(x, rr).argmax(dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_summary(self):
        return f"ECG_BeatDeep  total params: {self.count_parameters()}"


class ECG_BeatInceptionBig(nn.Module):
    """
    Multi-scale (Inception) 1D-CNN done right within the 15k budget. RR fused.

    Unlike the tiny 655-param inception (which over-pooled and starved F), this
    keeps more channels and pools gently to preserve the temporal detail the
    fusion class F needs. Stem 1->16, Inception(3x16) at two stages.

    256 -c1-> 256 -p/2-> 128 -inc1(48)-> -p/4-> 32 -inc2(48)-> -p/4-> 8
        -GAP-> 48 -concat RR(8)-> FC.
    """

    def __init__(self, num_classes=5, input_length=256, n_rr=8):
        super().__init__()
        self.num_classes  = num_classes
        self.input_length = input_length
        self.n_rr         = n_rr

        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3, bias=True)
        self.bn1   = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)                  # 256 → 128
        self.inc1  = InceptionBlock1D(16, br_ch=16)   # → 48 ch
        self.pool2 = nn.MaxPool1d(4)                  # 128 → 32
        self.inc2  = InceptionBlock1D(48, br_ch=12)   # → 36 ch
        self.pool3 = nn.MaxPool1d(4)                  # 32 → 8
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(36 + n_rr, num_classes, bias=True)

    def forward(self, x, rr=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))   # (B,16,128)
        x = self.pool2(self.inc1(x))                      # (B,48,32)
        x = self.pool3(self.inc2(x))                      # (B,36,8)
        x = self.gap(x).squeeze(-1)                       # (B,36)
        if rr is None:
            rr = x.new_zeros(x.size(0), self.n_rr)
        rr = rr[:, :self.n_rr]
        return self.fc(torch.cat([x, rr], dim=1))

    def predict(self, x, rr=None):
        return self.forward(x, rr).argmax(dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_summary(self):
        return f"ECG_BeatInceptionBig  total params: {self.count_parameters()}"


class DilatedBlock1D(nn.Module):
    """Dilated Conv-BN-ReLU with identity skip (same ch, same length)."""

    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=pad, dilation=dilation, bias=True)
        self.bn   = nn.BatchNorm1d(channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)) + x)


class ECG_BeatTCN(nn.Module):
    """
    Temporal Conv Net with dilated convs. RR fused at FC.

    Dilations 1/2/4 grow the receptive field exponentially WITHOUT heavy
    pooling, so the network sees wide rhythm context while keeping fine
    temporal resolution — the detail the fusion class F needs. Only one /4
    pool, late, to keep the GAP input rich.

    256 -c1-> 256 -p/2-> 128 -[dil1,dil2,dil4 @32ch]-> -p/4-> 32 -GAP-> 32
        -concat RR(8)-> FC.
    """

    def __init__(self, num_classes=5, input_length=256, n_rr=8):
        super().__init__()
        self.num_classes  = num_classes
        self.input_length = input_length
        self.n_rr         = n_rr

        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=True)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)                       # 256 → 128
        self.d1    = DilatedBlock1D(32, 3, dilation=1)
        self.d2    = DilatedBlock1D(32, 3, dilation=2)
        self.d4    = DilatedBlock1D(32, 3, dilation=4)
        self.pool2 = nn.MaxPool1d(4)                       # 128 → 32
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(32 + n_rr, num_classes, bias=True)

    def forward(self, x, rr=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))    # (B,32,128)
        x = self.d4(self.d2(self.d1(x)))                   # (B,32,128)
        x = self.pool2(x)                                  # (B,32,32)
        x = self.gap(x).squeeze(-1)                        # (B,32)
        if rr is None:
            rr = x.new_zeros(x.size(0), self.n_rr)
        rr = rr[:, :self.n_rr]
        return self.fc(torch.cat([x, rr], dim=1))

    def predict(self, x, rr=None):
        return self.forward(x, rr).argmax(dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_summary(self):
        return f"ECG_BeatTCN  total params: {self.count_parameters()}"


class ECG_BeatDualBranch(nn.Module):
    """
    Two-branch: a morphology CNN and a dedicated RR-MLP, fused before FC.

    The previous models concatenate raw RR features into a single linear FC,
    so RR can only contribute linearly. A small MLP lets the rhythm features
    interact non-linearly (e.g. "premature AND compensatory pause") before
    fusion — aimed squarely at the S class, the standing ceiling. The CNN
    branch is the proven deep-residual morphology stack.

    CNN: 256 -c1-> -p/2-> 128 -c2->res24-> -p/4-> 32 -c3->res40-> -p/4-> 8
         -GAP-> 40.   RR-MLP: 8 -> 16 -> 16.   concat(56) -> FC.
    """

    def __init__(self, num_classes=5, input_length=256, n_rr=8):
        super().__init__()
        self.num_classes  = num_classes
        self.input_length = input_length
        self.n_rr         = n_rr

        # morphology branch (same as ECG_BeatDeep up to GAP)
        self.conv1 = nn.Conv1d(1, 16, 7, padding=3, bias=True)
        self.bn1   = nn.BatchNorm1d(16); self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 24, 5, padding=2, bias=True)
        self.bn2   = nn.BatchNorm1d(24); self.res2 = ResidualBlock1D(24, 3)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(24, 40, 5, padding=2, bias=True)
        self.bn3   = nn.BatchNorm1d(40); self.res3 = ResidualBlock1D(40, 3)
        self.pool3 = nn.MaxPool1d(4)
        self.gap   = nn.AdaptiveAvgPool1d(1)

        # rhythm branch (RR-MLP)
        self.rr_mlp = nn.Sequential(
            nn.Linear(n_rr, 16), nn.ReLU(),
            nn.Linear(16, 16),   nn.ReLU(),
        )
        self.fc = nn.Linear(40 + 16, num_classes, bias=True)

    def forward(self, x, rr=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.pool1(F.relu(self.bn1(self.conv1(x))))
        h = self.pool2(self.res2(F.relu(self.bn2(self.conv2(h)))))
        h = self.pool3(self.res3(F.relu(self.bn3(self.conv3(h)))))
        h = self.gap(h).squeeze(-1)                        # (B,40)
        if rr is None:
            rr = h.new_zeros(h.size(0), self.n_rr)
        r = self.rr_mlp(rr[:, :self.n_rr])                 # (B,16)
        return self.fc(torch.cat([h, r], dim=1))

    def predict(self, x, rr=None):
        return self.forward(x, rr).argmax(dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_summary(self):
        return f"ECG_BeatDualBranch  total params: {self.count_parameters()}"


class ECG_BeatTiny(nn.Module):
    """
    The 727-param hybrid backbone of the 5-symbol model, single-head.

    Same conv stack (1->4->8->8, k5/k5/k3) + GAP + RR-MLP(12->8->8) concat as
    ECG_TinyMultiTask, but one FC head of `num_classes`. Used to run the tiny
    model on the AAMI N/S/V inter-patient task for a like-for-like comparison
    with the matched-filter CNN (1,267 params) at a smaller footprint.
    """

    def __init__(self, num_classes=5, input_length=256, n_rr=12, rr_hidden=8):
        super().__init__()
        self.num_classes = num_classes
        self.n_rr = n_rr
        self.conv1 = nn.Conv1d(1, 4, 5, padding=2, bias=True)
        self.bn1   = nn.BatchNorm1d(4); self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(4, 8, 5, padding=2, bias=True)
        self.bn2   = nn.BatchNorm1d(8); self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(8, 8, 3, padding=1, bias=True)
        self.bn3   = nn.BatchNorm1d(8); self.pool3 = nn.MaxPool1d(4)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.rr_mlp = nn.Sequential(
            nn.Linear(n_rr, rr_hidden), nn.ReLU(),
            nn.Linear(rr_hidden, rr_hidden), nn.ReLU(),
        )
        self.fc = nn.Linear(8 + rr_hidden, num_classes, bias=True)

    def forward(self, x, rr=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.pool1(F.relu(self.bn1(self.conv1(x))))
        h = self.pool2(F.relu(self.bn2(self.conv2(h))))
        h = self.pool3(F.relu(self.bn3(self.conv3(h))))
        h = self.gap(h).squeeze(-1)                        # (B,8)
        if rr is None:
            rr = h.new_zeros(h.size(0), self.n_rr)
        r = self.rr_mlp(rr[:, :self.n_rr])
        return self.fc(torch.cat([h, r], dim=1))

    def predict(self, x, rr=None):
        return self.forward(x, rr).argmax(dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_summary(self):
        return f"ECG_BeatTiny  total params: {self.count_parameters()}"


# ------------------------------------------------------------
#  Registry — 5-class variants. <=3k group + 15k-budget group.
# ------------------------------------------------------------

MODEL_REGISTRY = {
    'baseline':  (ECG_BeatCNN,          4),   # residual CNN + 4 RR (3k)
    'rr8':       (ECG_BeatCNN,          8),   # residual CNN + 8 RR (3k, prev best)
    'inception': (ECG_BeatInception,    4),   # tiny multi-scale (3k)
    'thesis':    (ECG_BeatThesis,       8),   # Thesis Chapman CNN + 8 RR (3k)
    'deep':      (ECG_BeatDeep,         8),   # deeper residual + 8 RR (15k, best)
    'incep15':   (ECG_BeatInceptionBig, 8),   # proper multi-scale + 8 RR (15k)
    'tcn':       (ECG_BeatTCN,          8),   # dilated TCN + 8 RR (15k)
    'dualbranch':(ECG_BeatDualBranch,   8),   # morph-CNN + RR-MLP (15k)
    'tiny':      (ECG_BeatTiny,        12),   # 727-param hybrid (vs MF-CNN)
}


def build_model(name='baseline', num_classes=5, input_length=256):
    cls, n_rr = MODEL_REGISTRY[name]
    return cls(num_classes=num_classes, input_length=input_length, n_rr=n_rr)


if __name__ == "__main__":
    BUDGET = 15000
    for name in MODEL_REGISTRY:
        m = build_model(name)
        p = m.count_parameters()
        flag = "OK" if p <= BUDGET else "OVER BUDGET"
        print(f"{name:<12} params={p:<6} [{flag}]")
        dummy = torch.randn(4, 256); rr = torch.randn(4, 8)
        assert m(dummy, rr).shape == (4, 5)
        assert p <= BUDGET, f"{name} exceeds {BUDGET}"
    print(f"\nall variants forward OK, all <= {BUDGET} params")
