"""
Tiny multi-task 1D-CNN for MIT-BIH 5-symbol beats (~500 weights)
================================================================
One small shared backbone, two independent dense heads:

  Dense 1 (binary)  : normal {N,L,R} vs abnormal {A,V}
  Dense 2 (5-class) : N / L / R / A / V

Target: dense parameter count ~500 (deploy-friendly, no pruning needed).

  conv1 (1->4,  k5)  : 1*4*5 + 4          =  24
  bn1                 : 4*2                =   8
  conv2 (4->8,  k5)  : 4*8*5 + 8          = 168
  bn2                 : 8*2                =  16
  conv3 (8->8,  k3)  : 8*8*3 + 8          = 200
  bn3                 : 8*2                =  16
  head_bin ((8+n_rr)->2) : (8+8)*2 + 2          =  34   (n_rr=8)
  head_5   ((8+n_rr)->5) : (8+8)*5 + 5          =  85
  -------------------------------------------------------------
  total (n_rr=8)                          = 551
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECG_TinyMultiTask(nn.Module):
    """Shared CNN backbone + binary head + 5-class head. RR fused before heads."""

    def __init__(self, n_rr=8, input_length=256, rr_hidden=8):
        super().__init__()
        self.n_rr = n_rr
        self.rr_hidden = rr_hidden

        self.conv1 = nn.Conv1d(1, 4, kernel_size=5, padding=2, bias=True)
        self.bn1   = nn.BatchNorm1d(4)
        self.pool1 = nn.MaxPool1d(4)                  # 256 → 64
        self.conv2 = nn.Conv1d(4, 8, kernel_size=5, padding=2, bias=True)
        self.bn2   = nn.BatchNorm1d(8)
        self.pool2 = nn.MaxPool1d(4)                  # 64 → 16
        self.conv3 = nn.Conv1d(8, 8, kernel_size=3, padding=1, bias=True)
        self.bn3   = nn.BatchNorm1d(8)
        self.pool3 = nn.MaxPool1d(4)                  # 16 → 4
        self.gap   = nn.AdaptiveAvgPool1d(1)

        # RR branch: a small MLP so rhythm features interact non-linearly
        # ("premature AND compensatory pause") — the cue that separates the
        # atrial-premature class A from morphologically-identical N.
        if rr_hidden > 0:
            self.rr_mlp = nn.Sequential(
                nn.Linear(n_rr, rr_hidden), nn.ReLU(),
                nn.Linear(rr_hidden, rr_hidden), nn.ReLU(),
            )
            feat = 8 + rr_hidden
        else:
            self.rr_mlp = None
            feat = 8 + n_rr

        self.head_bin = nn.Linear(feat, 2)            # Dense 1
        self.head_5   = nn.Linear(feat, 5)            # Dense 2

    def _backbone(self, x, rr):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).squeeze(-1)                   # (B, 8)
        if rr is None:
            rr = x.new_zeros(x.size(0), self.n_rr)
        rr = rr[:, :self.n_rr]
        if self.rr_mlp is not None:
            rr = self.rr_mlp(rr)
        return torch.cat([x, rr], dim=1)

    def forward(self, x, rr=None):
        f = self._backbone(x, rr)
        return self.head_bin(f), self.head_5(f)       # (logit_bin, logit_5)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_summary(self):
        return f"ECG_TinyMultiTask  total params: {self.count_parameters()}"


def build_model(n_rr=8, input_length=256, rr_hidden=8):
    return ECG_TinyMultiTask(n_rr=n_rr, input_length=input_length,
                             rr_hidden=rr_hidden)


if __name__ == "__main__":
    for rh in (0, 8):
        m = build_model(n_rr=8, rr_hidden=rh)
        p = m.count_parameters()
        print(f"rr_hidden={rh}: {p} params  [{'OK' if p <= 800 else 'OVER 800'}]")
    dummy = torch.randn(4, 256); rr = torch.randn(4, 8)
    lb, l5 = m(dummy, rr)
    print(f"logit_bin {tuple(lb.shape)}  logit_5 {tuple(l5.shape)}")
