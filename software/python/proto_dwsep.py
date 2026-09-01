"""
Depthwise-Separable 1D-CNN — software prototype (V-B, ECG philosophy)
=====================================================================
GO/NO-GO accuracy probe BEFORE any RTL work. Self-contained: does NOT
touch model.py / train.py / prune_finetune.py (production pipeline that
feeds the working RTL export).

Variant V-B (chosen): BN after every conv (train-time only, folded at
inference), ReLU ONLY after Conv4 — preserves negative ECG features
(Q/S waves) the baseline keeps by having no ReLU on Conv1-3.

Topology (MobileNet-style separable, stem not split):
  Input 2500x1 INT8
  Conv1(1->4, K5,p2)            +BN  -> Pool/5 -> 500x4   (stem)
  DW2(4,K5,p2)+PW2(4->4,K1)     +BN  -> Pool/5 -> 100x4
  DW3(4,K5,p2)+PW3(4->8,K1)     +BN  -> Pool/5 ->  20x8
  DW4(8,K5,p2)+PW4(8->8,K1) +BN +ReLU-> Pool/5 ->   4x8
  GAP -> FC(8->4) -> argmax

Conv weights: 580 (baseline) -> 212 INT8 (-63%).

Flow:
  1. train float (Chapman, reuse get_dataloaders/evaluate)
  2. fold BN into preceding conv -> plain Conv stack (eval must match float)
  3. INT8 power-of-2 round-half-up sim on folded model (RTL-fidelity)
  4. report params / MAC / acc / F1 vs baseline 94.65% / 0.9396

Usage:
  python proto_dwsep.py                 # train + eval
  python proto_dwsep.py --epochs 100
"""

import os
import sys
import copy
import json
import math
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset  import get_dataloaders, CLASS_NAMES
from utils.evaluate import evaluate_model, compute_metrics, print_classification_report


# ============================================================
#  Model — V-B (BN everywhere, ReLU only after Conv4)
# ============================================================

class SepBlock(nn.Module):
    """Depthwise(K5) + BN + Pointwise(K1) + BN. ReLU optional (Conv4 only)."""

    def __init__(self, in_ch, out_ch, relu=False):
        super().__init__()
        self.dw    = nn.Conv1d(in_ch, in_ch, kernel_size=5, padding=2,
                               groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm1d(in_ch)
        self.pw    = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm1d(out_ch)
        self.relu  = relu

    def forward(self, x):
        x = self.bn_dw(self.dw(x))
        x = self.bn_pw(self.pw(x))
        if self.relu:
            x = F.relu(x)
        return x


class ECG_DWSep(nn.Module):
    """
    DW-separable 1D-CNN, V-B: ReLU only after the last block (Conv4),
    matching the baseline's negative-feature-preserving philosophy.

    Channels 1->4->4->8->8, K=5, pool/5 each stage. GAP -> FC(8->4).
    """

    def __init__(self, num_classes=4, input_length=2500, channels=(4, 4, 8, 8)):
        super().__init__()
        self.num_classes  = num_classes
        self.input_length = input_length
        self.channels     = tuple(channels)          # (c1,c2,c3,c4) output ch
        c1, c2, c3, c4 = self.channels

        # Stem: standard conv (in_ch=1, splitting is pointless)
        self.conv1  = nn.Conv1d(1, c1, kernel_size=5, padding=2, bias=False)
        self.bn1    = nn.BatchNorm1d(c1)
        self.pool1  = nn.MaxPool1d(5)

        self.block2 = SepBlock(c1, c2, relu=False)
        self.pool2  = nn.MaxPool1d(5)

        self.block3 = SepBlock(c2, c3, relu=False)
        self.pool3  = nn.MaxPool1d(5)

        self.block4 = SepBlock(c3, c4, relu=True)   # ReLU here only
        self.pool4  = nn.MaxPool1d(5)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(c4, num_classes, bias=True)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(self.bn1(self.conv1(x)))   # 500x4  [no ReLU]
        x = self.pool2(self.block2(x))            # 100x4  [no ReLU]
        x = self.pool3(self.block3(x))            #  20x8  [no ReLU]
        x = self.pool4(self.block4(x))            #   4x8  [+ReLU]
        x = self.gap(x).squeeze(-1)               #   8
        return self.fc(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
#  BN folding -> plain Conv stack (inference-equivalent)
# ============================================================

class ECG_DWSep_Folded(nn.Module):
    """
    BN folded into each preceding conv -> bias-enabled conv with NO BN.
    Layer attribute names match the INT8 sim contract:
      conv1, dw2, pw2, dw3, pw3, dw4, pw4, fc
    """

    _CONV_NAMES = ['conv1', 'dw2', 'pw2', 'dw3', 'pw3', 'dw4', 'pw4']

    def __init__(self, trained: ECG_DWSep):
        super().__init__()
        self.num_classes = trained.num_classes
        m = trained.eval()

        def fold(conv, bn):
            """Return (W', b') = conv folded with following BN."""
            w = conv.weight.data.clone()
            gamma = bn.weight.data
            beta  = bn.bias.data
            mean  = bn.running_mean
            var   = bn.running_var
            eps   = bn.eps
            std   = torch.sqrt(var + eps)
            scale = gamma / std                       # (out_ch,)
            w_fold = w * scale.reshape(-1, *([1] * (w.dim() - 1)))
            b_prev = conv.bias.data if conv.bias is not None else torch.zeros_like(beta)
            b_fold = beta + (b_prev - mean) * scale
            return w_fold, b_fold

        c1, c2, c3, c4 = m.channels

        # Stem conv1 + bn1
        self.conv1 = nn.Conv1d(1, c1, 5, padding=2, bias=True)
        w, b = fold(m.conv1, m.bn1)
        self.conv1.weight.data.copy_(w); self.conv1.bias.data.copy_(b)

        # Separable blocks: each has dw+bn_dw and pw+bn_pw
        def make_sep(block, dw_io, pw_io):
            dw = nn.Conv1d(dw_io[0], dw_io[1], 5, padding=2,
                           groups=dw_io[0], bias=True)
            pw = nn.Conv1d(pw_io[0], pw_io[1], 1, bias=True)
            wd, bd = fold(block.dw, block.bn_dw)
            wp, bp = fold(block.pw, block.bn_pw)
            dw.weight.data.copy_(wd); dw.bias.data.copy_(bd)
            pw.weight.data.copy_(wp); pw.bias.data.copy_(bp)
            return dw, pw

        self.dw2, self.pw2 = make_sep(m.block2, (c1, c1), (c1, c2))
        self.dw3, self.pw3 = make_sep(m.block3, (c2, c2), (c2, c3))
        self.dw4, self.pw4 = make_sep(m.block4, (c3, c3), (c3, c4))

        self.pool1 = m.pool1; self.pool2 = m.pool2
        self.pool3 = m.pool3; self.pool4 = m.pool4
        self.gap   = m.gap

        self.fc = nn.Linear(c4, self.num_classes, bias=True)
        self.fc.weight.data.copy_(m.fc.weight.data)
        self.fc.bias.data.copy_(m.fc.bias.data)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.pw2(self.dw2(x)))
        x = self.pool3(self.pw3(self.dw3(x)))
        x = self.pool4(F.relu(self.pw4(self.dw4(x))))   # ReLU after Conv4
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
#  INT8 power-of-2 round-half-up sim (RTL fidelity)
# ============================================================

def _shift_bits(abs_max, qmax=127):
    if abs_max == 0:
        return 0
    return max(0, min(int(math.floor(math.log2(qmax / abs_max))), 15))


def _round_shift(x, n):
    """Round-half-up arithmetic right shift, matches RTL pe_requantize_fused."""
    if n > 0:
        return torch.floor((x + 2.0 ** (n - 1)) / (2.0 ** n))
    return x


class ECG_DWSep_INT8(nn.Module):
    """
    Hardware-fidelity INT8 sim on the FOLDED DW-sep model.

    Per-conv:
      w_int8 = clamp(round(w * 2^w_shift), -127, 127)
      acc    = conv(x_int8, w_int8) + round(b * 2^nb)
      out    = clamp(round_shift(acc, nb), -127, 127)   [+ReLU only after pw4]
    nb(conv1) = input_shift + w_shift ; nb(other) = w_shift.

    NOTE: DW/PW are separate rescale points (each its own nb), so a separable
    block has 2 rescales vs 1 in a standard conv — mirrors the RTL cost.
    """

    _CONV_NAMES = ['conv1', 'dw2', 'pw2', 'dw3', 'pw3', 'dw4', 'pw4']

    def __init__(self, folded: ECG_DWSep_Folded, calib_loader,
                 device=None, n_cal_batches=20):
        super().__init__()
        if device is None:
            device = next(folded.parameters()).device
        self.device = device

        self._w_int8  = {}
        self._b_float = {}
        self._w_shift = {}
        self._groups  = {}

        for name in self._CONV_NAMES + ['fc']:
            layer = getattr(folded, name)
            w_np = layer.weight.data.cpu().numpy()
            abs_max = max(abs(w_np.min()), abs(w_np.max()))
            n = _shift_bits(abs_max)
            self._w_shift[name] = n
            self._w_int8[name]  = np.clip(np.round(w_np * (2.0 ** n)),
                                          -127, 127).astype(np.float32)
            self._b_float[name] = (layer.bias.data.cpu().numpy().astype(np.float32)
                                   if layer.bias is not None else None)
            self._groups[name]  = getattr(layer, 'groups', 1)

        # Input shift calibrated from data
        max_input = 0.0
        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                if i >= n_cal_batches:
                    break
                v = batch[0].abs().max().item()
                max_input = max(max_input, v)
        self._input_shift = _shift_bits(max_input)

        self._pool1 = folded.pool1; self._pool2 = folded.pool2
        self._pool3 = folded.pool3; self._pool4 = folded.pool4
        self._gap   = folded.gap

        # nb calibrated from real activations (DW-sep accumulators are small;
        # the conv-standard rule nb=w_shift over-shifts and crushes the range).
        # nb[L] = floor(log2(acc_absmax / 127)) so out uses the full [-127,127].
        self._nb = self._calibrate_nb(calib_loader, n_cal_batches)

    def _calibrate_nb(self, calib_loader, n_cal_batches):
        """One INT8 pass with nb=0 to measure pre-rescale acc range per layer."""
        acc_max = {n: 0.0 for n in self._CONV_NAMES}

        def raw_conv(x, name, padding):
            w = torch.tensor(self._w_int8[name]).to(self.device)
            b = (torch.tensor(np.round(self._b_float[name]).astype(np.float32)).to(self.device)
                 if self._b_float[name] is not None else None)
            return F.conv1d(x, w, b, padding=padding, groups=self._groups[name])

        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                if i >= n_cal_batches:
                    break
                x = batch[0].unsqueeze(1).float().to(self.device)
                x = torch.clamp(torch.round(x * (2.0 ** self._input_shift)), -127, 127)
                # walk layers, clamp output with provisional nb from running estimate
                seq = [('conv1', 2, self._pool1),
                       ('dw2', 2, None), ('pw2', 0, self._pool2),
                       ('dw3', 2, None), ('pw3', 0, self._pool3),
                       ('dw4', 2, None), ('pw4', 0, self._pool4)]
                for name, pad, pool in seq:
                    a = raw_conv(x, name, pad)
                    acc_max[name] = max(acc_max[name], a.abs().max().item())
                    n = max(0, min(int(math.floor(math.log2(acc_max[name] / 127.0)))
                                   if acc_max[name] > 127 else 0, 15))
                    out = torch.clamp(_round_shift(a, n), -127, 127)
                    if name == 'pw4':
                        out = F.relu(out)
                    x = pool(out) if pool is not None else out

        nb = {}
        for name in self._CONV_NAMES:
            nb[name] = (max(0, min(int(math.floor(math.log2(acc_max[name] / 127.0))), 15))
                        if acc_max[name] > 127 else 0)
        return nb

    def _conv(self, x, name, padding, relu=False):
        w  = torch.tensor(self._w_int8[name]).to(self.device)
        n  = self._nb[name]
        b  = (torch.tensor(np.round(self._b_float[name] * (2.0 ** n)).astype(np.float32)).to(self.device)
              if self._b_float[name] is not None else None)
        out = F.conv1d(x, w, b, padding=padding, groups=self._groups[name])
        out = torch.clamp(_round_shift(out, n), -127, 127)
        if relu:
            out = F.relu(out)
        return out

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = torch.clamp(torch.round(x * (2.0 ** self._input_shift)), -127, 127)

        x = self._pool1(self._conv(x, 'conv1', padding=2))
        x = self._pool2(self._conv(self._conv(x, 'dw2', padding=2), 'pw2', padding=0))
        x = self._pool3(self._conv(self._conv(x, 'dw3', padding=2), 'pw3', padding=0))
        # block4: ReLU applied at the pw4 output (last conv before pool4)
        x = self._conv(x, 'dw4', padding=2)
        x = self._conv(x, 'pw4', padding=0, relu=True)
        x = self._pool4(x)

        x = self._gap(x).squeeze(-1)

        # FC: no rescale (nb_fc=0), bias scaled to logit domain by 2^w_shift[fc]
        w_fc = torch.tensor(self._w_int8['fc']).to(self.device)
        fc_shift = self._w_shift['fc']
        b_fc = torch.tensor(np.round(self._b_float['fc'] * (2.0 ** fc_shift)).astype(np.float32)).to(self.device)
        return F.linear(x, w_fc, b_fc)


# ============================================================
#  MAC counter (for the cost story)
# ============================================================

def count_macs(channels=(4, 4, 8, 8)):
    """Multiply-accumulates per inference, conv layers only (dominant)."""
    c1, c2, c3, c4 = channels
    # (name, out_len, in_ch, out_ch, K, groups)
    layers = [
        ('conv1', 500, 1,  c1, 5, 1),   # out_len after pool not used; use conv out_len
        ('dw2',   500, c1, c1, 5, c1),
        ('pw2',   500, c1, c2, 1, 1),
        ('dw3',   100, c2, c2, 5, c2),
        ('pw3',   100, c2, c3, 1, 1),
        ('dw4',    20, c3, c3, 5, c3),
        ('pw4',    20, c3, c4, 1, 1),
    ]
    # conv1 runs over full 2500 before pool
    layers[0] = ('conv1', 2500, 1, c1, 5, 1)
    total = 0
    rows = []
    for name, L, cin, cout, K, g in layers:
        macs = L * (cin // g) * cout * K
        total += macs
        rows.append((name, macs))
    return total, rows


def count_conv_weights(channels=(4, 4, 8, 8)):
    """INT8 conv weight count (RTL-relevant), excludes BN (folded) and FC."""
    c1, c2, c3, c4 = channels
    return (1 * c1 * 5            # conv1
            + c1 * 5 + c1 * c2    # dw2 + pw2
            + c2 * 5 + c2 * c3    # dw3 + pw3
            + c3 * 5 + c3 * c4)   # dw4 + pw4


# ============================================================
#  Train
# ============================================================

def train_float(model, train_loader, val_loader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    half = max(1, epochs // 2)
    sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[half], gamma=0.1)

    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    print(f"\n  Training {epochs} epochs (lr {lr:.0e}, drop x0.1 @ ep{half}):")
    print(f"  {'Ep':>4} {'TrLoss':>8} {'TrAcc':>7} {'VaLoss':>8} {'VaAcc':>7} {'Time':>6}")
    print("  " + "-" * 52)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        rl, c, t = 0.0, 0, 0
        for ecg, labels, _ in train_loader:
            ecg = ecg.unsqueeze(1).float().to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(ecg)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            rl += loss.item() * ecg.size(0)
            c  += out.detach().argmax(1).eq(labels).sum().item()
            t  += labels.size(0)
        tr_loss, tr_acc = rl / t, c / t

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for ecg, labels, _ in val_loader:
                ecg = ecg.unsqueeze(1).float().to(device)
                labels = labels.to(device)
                out = model(ecg)
                vl += criterion(out, labels).item() * ecg.size(0)
                vc += out.argmax(1).eq(labels).sum().item()
                vt += labels.size(0)
        va_loss, va_acc = vl / vt, vc / vt

        star = ""
        if va_loss < best_val:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            star = " *"
        print(f"  {ep:>4} {tr_loss:>8.4f} {tr_acc:>7.4f} "
              f"{va_loss:>8.4f} {va_acc:>7.4f} {time.time()-t0:>5.1f}s{star}")
        sched.step()

    model.load_state_dict(best_state)
    print(f"\n  Best val_loss = {best_val:.4f}")
    return model


# ============================================================
#  Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir',    type=str, default='../../data/Chapman')
    ap.add_argument('--output_dir',  type=str, default='./results/dwsep_proto')
    ap.add_argument('--epochs',      type=int, default=100)
    ap.add_argument('--batch_size',  type=int, default=128)
    ap.add_argument('--lr',          type=float, default=1e-3)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--channels',    type=str, default='4,4,8,8',
                    help='Conv1-4 output channels, comma-separated')
    args = ap.parse_args()
    channels = tuple(int(c) for c in args.channels.split(','))
    assert len(channels) == 4, "need 4 channel values"

    seed = 42
    np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  ECG DW-Separable prototype (V-B: ReLU only after Conv4)")
    print("=" * 60)
    print(f"  Device: {device}  seed={seed}  channels={channels}")

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    model = ECG_DWSep(num_classes=4, channels=channels).to(device)
    n_params = model.count_parameters()
    macs, mac_rows = count_macs(channels)
    print(f"\n  Trainable params (incl BN): {n_params}")
    print(f"  Conv MACs/inference: {macs:,}")
    for name, m in mac_rows:
        print(f"    {name:<7} {m:>10,}")

    # 1. train float
    model = train_float(model, train_loader, val_loader, device,
                        args.epochs, args.lr)
    torch.save({'model_state_dict': model.state_dict()},
               os.path.join(args.output_dir, 'dwsep_vb.pth'))

    # float eval
    fp, fl = evaluate_model(model, test_loader, device)
    m_float = compute_metrics(fp, fl, CLASS_NAMES)
    print_classification_report(m_float, title="DW-sep V-B — Float (fp32)")

    # 2. fold BN, eval (should match float within rounding)
    folded = ECG_DWSep_Folded(model).to(device)
    gp, gl = evaluate_model(folded, test_loader, device)
    m_fold = compute_metrics(gp, gl, CLASS_NAMES)
    print_classification_report(m_fold, title="DW-sep V-B — BN folded (fp32)")
    fold_drift = abs(m_fold['accuracy'] - m_float['accuracy'])
    print(f"  [check] fold drift = {fold_drift:.5f} (expect ~0)")

    # conv-only param count after fold (RTL-relevant weights)
    conv_w = sum(getattr(folded, n).weight.numel()
                 for n in ECG_DWSep_Folded._CONV_NAMES)
    print(f"  Conv INT8 weights (RTL): {conv_w}  vs baseline 580")

    # 3. INT8 power-of-2 sim
    int8 = ECG_DWSep_INT8(folded, calib_loader=train_loader, device=device)
    ip, il = evaluate_model(int8, test_loader, device)
    m_int8 = compute_metrics(ip, il, CLASS_NAMES)
    print_classification_report(m_int8, title="DW-sep V-B — INT8 power-of-2 (hardware sim)")
    print(f"  input_shift={int8._input_shift}  "
          f"w_shift={int8._w_shift}  nb={int8._nb}")

    # 4. summary vs baseline
    print("\n" + "=" * 60)
    print("  GO/NO-GO SUMMARY  (baseline pruned INT8 = 94.65% / F1 0.9396)")
    print("=" * 60)
    print(f"  {'Stage':<22} {'Acc':>8} {'F1':>8}")
    print("  " + "-" * 40)
    for nm, m in [("Float", m_float), ("BN folded", m_fold), ("INT8 p2", m_int8)]:
        print(f"  {nm:<22} {m['accuracy']:>8.4f} {m['f1_macro']:>8.4f}")
    print("  " + "-" * 40)
    print(f"  Conv weights: {conv_w} vs 580 ({100*(1-conv_w/580):.0f}% fewer)")
    print(f"  INT8 acc delta vs baseline: {m_int8['accuracy']-0.9465:+.4f}")

    results = {
        'variant': 'V-B (ReLU only after Conv4)',
        'params_trainable': n_params,
        'conv_int8_weights': conv_w,
        'conv_weights_baseline': 580,
        'macs': macs,
        'float':     {'acc': m_float['accuracy'], 'f1': m_float['f1_macro']},
        'bn_folded': {'acc': m_fold['accuracy'],  'f1': m_fold['f1_macro'],
                      'fold_drift': fold_drift},
        'int8_p2':   {'acc': m_int8['accuracy'],  'f1': m_int8['f1_macro']},
        'shifts': {'input': int8._input_shift,
                   'w_shift': int8._w_shift, 'nb': int8._nb},
        'baseline_int8': {'acc': 0.9465, 'f1': 0.9396},
    }
    with open(os.path.join(args.output_dir, 'dwsep_proto_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results -> {args.output_dir}/dwsep_proto_results.json")


if __name__ == '__main__':
    main()
