"""QAT W4A4 ablation — full INT4 (weights + activations), two scale modes.

Software-only ablation rows for Table 4 (C1). Does NOT touch RTL, does NOT
modify qat_int8.py / qat_int8_general.py.

Two scale modes, selected with --scale-mode:

  p2       Power-of-2 shift scale (consistent with A2). The KEY fix vs the
           naive first attempt: the fake-quant scale during QAT is ROUNDED to
           the same power-of-2 the integer convert uses, so the model trains
           against the real shifted range instead of a finer EMA scale it can
           never get at convert time. (Naive p2 without this fix collapsed
           81% fake-quant -> 43% integer because activations with abs_max~127
           need a negative shift that clamps to 0, saturating everything >7.)

  general  General float scale s = abs_max / 7 (the INT4 analogue of A3). No
           power-of-2 constraint -> this is the true accuracy CEILING of INT4
           on this model, telling us whether INT4 is inherently too coarse or
           whether power-of-2 alignment is the culprit.

qmax = 7 for both weights and activations; bias stays INT32 (never narrowed).

Usage (from software/python, venv active):
    # power-of-2 (shift-only, hardware-cheap), the fixed version
    python quantization/qat_int4.py --scale-mode p2 \\
        --checkpoint ./results/best_model_pruned.pth \\
        --output_dir ./results/ablation_quant/a7_qat_w4a4_p2 --epochs 50

    # general-scale (multiplier rescale) — INT4 accuracy ceiling
    python quantization/qat_int4.py --scale-mode general \\
        --checkpoint ./results/best_model_pruned.pth \\
        --output_dir ./results/ablation_quant/a8_qat_w4a4_general --epochs 50
"""

import os
import sys
import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from prune_finetune import ECG_1DCNN_Pruned
from model.model import ECG_1DCNN
from utils.dataset import get_dataloaders, CLASS_NAMES
from utils.evaluate import compute_metrics, print_classification_report

from quantization.qat_int8 import round_shift, LAYER_ORDER, CONV_LAYERS

QMAX = 7   # INT4 signed: [-7, 7]


# ============================================================
#  FakeQuantize — power-of-2 aligned, and general float
# ============================================================

class FakeQuantizeP2(nn.Module):
    """Fake-quant whose scale is rounded to a power-of-2, matching the integer
    convert path exactly. scale = 2^(-shift), shift = floor(log2(qmax/abs_max)),
    clamped >= 0 (shift can't go negative -> matches RTL barrel-shift-only).

    This is the fix: the model now trains against the SAME coarse, possibly-
    saturating range the integer forward will use, so fake-quant ~ integer.
    """

    def __init__(self, qmax=QMAX, momentum=0.01):
        super().__init__()
        self.qmax = qmax
        self.momentum = momentum
        self.register_buffer('abs_max', torch.tensor(1.0))
        self.register_buffer('step', torch.tensor(0))

    def _shift(self):
        am = self.abs_max.clamp(min=1e-8)
        n = torch.floor(torch.log2(self.qmax / am))
        return n.clamp(min=0, max=15)

    def forward(self, x):
        if self.training:
            am = x.detach().abs().max().clamp(min=1e-8)
            if self.step == 0:
                self.abs_max.copy_(am)
            else:
                self.abs_max.mul_(1.0 - self.momentum).add_(am * self.momentum)
            self.step += 1
        scale = 2.0 ** (-self._shift())          # = 1 / 2^shift
        x_q = (x / scale).round().clamp(-self.qmax, self.qmax) * scale
        return x + (x_q - x).detach()


class FakeQuantizeGen(nn.Module):
    """General float scale s = abs_max / qmax (EMA). INT4 analogue of A3."""

    def __init__(self, qmax=QMAX, momentum=0.01):
        super().__init__()
        self.qmax = qmax
        self.momentum = momentum
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('step', torch.tensor(0))

    def forward(self, x):
        if self.training:
            am = x.detach().abs().max().clamp(min=1e-8)
            new = am / self.qmax
            if self.step == 0:
                self.scale.copy_(new)
            else:
                self.scale.mul_(1.0 - self.momentum).add_(new * self.momentum)
            self.step += 1
        s = self.scale.clamp(min=1e-8)
        x_q = (x / s).round().clamp(-self.qmax, self.qmax) * s
        return x + (x_q - x).detach()


class ECG_QAT4(nn.Module):
    """Topology identical to ECG_1DCNN_QAT; FQ class chosen by scale_mode."""

    def __init__(self, c1_out=4, c2_out=4, c3_out=8, c4_out=8,
                 num_classes=4, scale_mode='p2'):
        super().__init__()
        self.c1_out, self.c2_out, self.c3_out, self.c4_out = c1_out, c2_out, c3_out, c4_out
        FQ = FakeQuantizeP2 if scale_mode == 'p2' else FakeQuantizeGen

        self.conv1 = nn.Conv1d(1,      c1_out, 5, padding=2, bias=True)
        self.conv2 = nn.Conv1d(c1_out, c2_out, 5, padding=2, bias=True)
        self.conv3 = nn.Conv1d(c2_out, c3_out, 5, padding=2, bias=True)
        self.conv4 = nn.Conv1d(c3_out, c4_out, 5, padding=2, bias=True)
        self.pool1 = nn.MaxPool1d(5); self.pool2 = nn.MaxPool1d(5)
        self.pool3 = nn.MaxPool1d(5); self.pool4 = nn.MaxPool1d(5)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(c4_out, num_classes, bias=True)

        self.fq_w1, self.fq_w2, self.fq_w3, self.fq_w4, self.fq_wfc = (FQ() for _ in range(5))
        self.fq_in, self.fq_a1, self.fq_a2, self.fq_a3, self.fq_a4, self.fq_gap = (FQ() for _ in range(6))

    def forward(self, x, quantize=True):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if not quantize:
            x = self.pool1(F.conv1d(x, self.conv1.weight, self.conv1.bias, padding=2))
            x = self.pool2(F.conv1d(x, self.conv2.weight, self.conv2.bias, padding=2))
            x = self.pool3(F.conv1d(x, self.conv3.weight, self.conv3.bias, padding=2))
            x = self.pool4(F.relu(F.conv1d(x, self.conv4.weight, self.conv4.bias, padding=2)))
            return F.linear(self.gap(x).squeeze(-1), self.fc.weight, self.fc.bias)
        x = self.fq_a1(self.pool1(F.conv1d(self.fq_in(x), self.fq_w1(self.conv1.weight), self.conv1.bias, padding=2)))
        x = self.fq_a2(self.pool2(F.conv1d(x, self.fq_w2(self.conv2.weight), self.conv2.bias, padding=2)))
        x = self.fq_a3(self.pool3(F.conv1d(x, self.fq_w3(self.conv3.weight), self.conv3.bias, padding=2)))
        x = self.pool4(F.relu(self.fq_a4(F.conv1d(x, self.fq_w4(self.conv4.weight), self.conv4.bias, padding=2))))
        x = self.fq_gap(self.gap(x).squeeze(-1))
        return F.linear(x, self.fq_wfc(self.fc.weight), self.fc.bias)


def build(base, scale_mode):
    if isinstance(base, ECG_1DCNN_Pruned):
        m = ECG_QAT4(base.c1_out, base.c2_out, base.c3_out, base.c4_out, scale_mode=scale_mode)
    else:
        m = ECG_QAT4(scale_mode=scale_mode)
    with torch.no_grad():
        for name in LAYER_ORDER:
            src, dst = getattr(base, name), getattr(m, name)
            dst.weight.copy_(src.weight)
            if src.bias is not None:
                dst.bias.copy_(src.bias)
    return m


def shift_bits(abs_max, qmax=QMAX):
    if abs_max == 0:
        return 0
    return max(0, min(int(math.floor(math.log2(qmax / abs_max))), 15))


# ---------------- power-of-2 integer forward ----------------

def convert_p2(model, train_loader, device, n_cal=20):
    model.eval()
    w_shift, w_int, b_float = {}, {}, {}
    for name in LAYER_ORDER:
        layer = getattr(model, name)
        w = layer.weight.data.cpu().numpy()
        n = shift_bits(max(abs(w.min()), abs(w.max())))
        w_shift[name] = n
        w_int[name] = np.clip(np.round(w * (2.0 ** n)), -QMAX, QMAX).astype(np.int32)
        if layer.bias is not None:
            b_float[name] = layer.bias.data.cpu().numpy()
    mx = 0.0
    with torch.no_grad():
        for i, b in enumerate(train_loader):
            if i >= n_cal:
                break
            mx = max(mx, b[0].abs().max().item())
    input_shift = shift_bits(mx)
    nb = {n: (input_shift + w_shift[n]) if n == 'conv1' else w_shift[n] for n in CONV_LAYERS}
    return w_int, b_float, w_shift, nb, input_shift


def forward_p2(model, x, w_int, b_float, nb, w_shift, input_shift):
    if x.dim() == 2:
        x = x.unsqueeze(1)
    dev = next(model.parameters()).device
    x = torch.clamp(torch.round(x * (2.0 ** input_shift)), -QMAX, QMAX)

    def cl(x, name):
        w = torch.tensor(w_int[name].astype(np.float32)).to(dev)
        n = nb[name]
        b = torch.tensor(np.round(b_float[name] * (2.0 ** n)).astype(np.float32)).to(dev)
        out = F.conv1d(x, w, b, padding=getattr(model, name).padding)
        return torch.clamp(round_shift(out, n), -QMAX, QMAX)

    x = model.pool1(cl(x, 'conv1')); x = model.pool2(cl(x, 'conv2')); x = model.pool3(cl(x, 'conv3'))
    w4 = torch.tensor(w_int['conv4'].astype(np.float32)).to(dev); n4 = nb['conv4']
    b4 = torch.tensor(np.round(b_float['conv4'] * (2.0 ** n4)).astype(np.float32)).to(dev)
    x = torch.clamp(round_shift(F.conv1d(x, w4, b4, padding=model.conv4.padding), n4), -QMAX, QMAX)
    x = torch.clamp(x, min=0); x = model.pool4(x)
    x = model.gap(x).squeeze(-1)
    # FC bias scaled to logit domain by 2^w_shift[fc] (no output rescale).
    w_fc = torch.tensor(w_int['fc'].astype(np.float32)).to(dev)
    b_fc = torch.tensor(
        np.round(b_float['fc'] * (2.0 ** w_shift['fc'])).astype(np.float32)).to(dev)
    return F.linear(x, w_fc, b_fc)


# ---------------- general-scale integer forward ----------------

def convert_gen(model, train_loader, device, n_cal=20):
    model.eval()
    w_scale, w_int, b_float = {}, {}, {}
    for name in LAYER_ORDER:
        layer = getattr(model, name)
        w = layer.weight.data.cpu().numpy()
        am = max(max(abs(w.min()), abs(w.max())), 1e-8)
        s = am / QMAX
        w_scale[name] = s
        w_int[name] = np.clip(np.round(w / s), -QMAX, QMAX).astype(np.int32)
        if layer.bias is not None:
            b_float[name] = layer.bias.data.cpu().numpy()
    mx = 0.0
    with torch.no_grad():
        for i, b in enumerate(train_loader):
            if i >= n_cal:
                break
            mx = max(mx, b[0].abs().max().item())
    input_scale = max(mx, 1e-8) / QMAX
    fq = {'conv1': model.fq_a1.scale.item(), 'conv2': model.fq_a2.scale.item(),
          'conv3': model.fq_a3.scale.item(), 'conv4': model.fq_a4.scale.item()}
    x_in = {'conv1': input_scale, 'conv2': fq['conv1'], 'conv3': fq['conv2'], 'conv4': fq['conv3']}
    x_out = {n: fq[n] for n in CONV_LAYERS}
    return w_int, b_float, w_scale, x_in, x_out, input_scale


def forward_gen(model, x, w_int, b_float, w_scale, x_in, x_out, input_scale):
    if x.dim() == 2:
        x = x.unsqueeze(1)
    dev = next(model.parameters()).device
    x = torch.clamp(torch.round(x / input_scale), -QMAX, QMAX)

    def cl(x, name, s_in, s_out):
        s_acc = s_in * w_scale[name]
        factor = s_acc / s_out
        w = torch.tensor(w_int[name].astype(np.float32)).to(dev)
        b = torch.tensor(np.round(b_float[name] / s_acc).astype(np.float32)).to(dev)
        acc = F.conv1d(x, w, b, padding=getattr(model, name).padding)
        return torch.clamp(torch.round(acc * factor), -QMAX, QMAX)

    x = model.pool1(cl(x, 'conv1', x_in['conv1'], x_out['conv1']))
    x = model.pool2(cl(x, 'conv2', x_in['conv2'], x_out['conv2']))
    x = model.pool3(cl(x, 'conv3', x_in['conv3'], x_out['conv3']))
    s_acc4 = x_in['conv4'] * w_scale['conv4']; factor4 = s_acc4 / x_out['conv4']
    w4 = torch.tensor(w_int['conv4'].astype(np.float32)).to(dev)
    b4 = torch.tensor(np.round(b_float['conv4'] / s_acc4).astype(np.float32)).to(dev)
    x = torch.clamp(torch.round(F.conv1d(x, w4, b4, padding=model.conv4.padding) * factor4), -QMAX, QMAX)
    x = torch.clamp(x, min=0); x = model.pool4(x)
    x = model.gap(x).squeeze(-1)
    # FC bias in logit domain: s_acc_fc = x_out['conv4'] * w_scale['fc'] (GAP
    # preserves conv4 output scale). b_int = round(b_float / s_acc_fc).
    w_fc = torch.tensor(w_int['fc'].astype(np.float32)).to(dev)
    s_acc_fc = x_out['conv4'] * w_scale['fc']
    b_fc = torch.tensor(
        np.round(b_float['fc'] / s_acc_fc).astype(np.float32)).to(dev)
    return F.linear(x, w_fc, b_fc)


def evaluate(model, loader, fwd, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for b in loader:
            logits = fwd(b[0].to(device))
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(b[1].numpy())
    preds, labels = np.array(preds), np.array(labels)
    return (preds == labels).mean(), preds, labels


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] QAT W4A4 scale-mode={args.scale_mode} (qmax={QMAX}, bias INT32)")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    is_pruned = 'c1_out' in ckpt
    if is_pruned:
        print(f"[INFO] Pruned (c1={ckpt['c1_out']},c2={ckpt['c2_out']},c3={ckpt['c3_out']},c4={ckpt['c4_out']})")
        base = ECG_1DCNN_Pruned(c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
                                c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'])
    else:
        base = ECG_1DCNN(num_classes=4)
    base.load_state_dict(ckpt['model_state_dict'])
    base = base.to(device)

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=2)

    print(f"\n{'='*60}\n  Phase 1: QAT INT4 training "
          f"({args.epochs} ep, lr={args.lr}, {args.scale_mode})\n{'='*60}")
    model = build(base, args.scale_mode).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    qat_path = os.path.join(args.output_dir, f'model_qat_w4a4_{args.scale_mode}_float.pth')

    best_val, history = 0.0, []
    for ep in range(args.epochs):
        model.train()
        for b in train_loader:
            x, y = b[0].to(device), b[1].to(device)
            opt.zero_grad(); crit(model(x, quantize=True), y).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv, lv = [], []
            for b in val_loader:
                pv.extend(model(b[0].to(device), quantize=True).argmax(1).cpu().numpy())
                lv.extend(b[1].numpy())
        va = (np.array(pv) == np.array(lv)).mean()
        if va > best_val:
            best_val = va
            torch.save({'model_state_dict': model.state_dict(),
                        **({k: ckpt[k] for k in ('c1_out','c2_out','c3_out','c4_out')} if is_pruned else {})},
                       qat_path)
        history.append({'epoch': ep, 'val_acc': float(va)})
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  Epoch {ep+1:3d}/{args.epochs}  val_acc={va:.4f}"
                  + ("  <- best" if va >= best_val else ""))
    print(f"\n  Best fake-quant val_acc: {best_val:.4f}")
    with open(os.path.join(args.output_dir, 'qat_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    qc = torch.load(qat_path, map_location=device, weights_only=False)
    model.load_state_dict(qc['model_state_dict']); model.eval()

    print(f"\n{'='*60}\n  Phase 2: INT4 conversion + integer eval ({args.scale_mode})\n{'='*60}")
    if args.scale_mode == 'p2':
        w_int, b_float, w_shift, nb, input_shift = convert_p2(model, train_loader, device)
        print(f"  input_shift={input_shift}  w_shift={w_shift}  nb={nb}")
        fwd = lambda x: forward_p2(model, x, w_int, b_float, nb, w_shift, input_shift)
        meta = {'w_shift': w_shift, 'nb': nb, 'input_shift_bits': int(input_shift)}
    else:
        w_int, b_float, w_scale, x_in, x_out, input_scale = convert_gen(model, train_loader, device)
        print(f"  input_scale={input_scale:.4f}")
        print(f"  w_scale={ {k: round(v,4) for k,v in w_scale.items()} }")
        fwd = lambda x: forward_gen(model, x, w_int, b_float, w_scale, x_in, x_out, input_scale)
        meta = {'w_scale': {k: float(v) for k, v in w_scale.items()},
                'input_scale': float(input_scale), 'dsp_extra_rescale': 4}

    acc, preds, labels = evaluate(model, test_loader, fwd, device)
    metrics = compute_metrics(preds, labels, CLASS_NAMES)
    print(f"\n  QAT W4A4 ({args.scale_mode}) INT4 acc : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-macro : {metrics['f1_macro']:.4f}")
    print(f"  fake-quant vs integer gap : {(best_val-acc)*100:+.2f}pp (small = convert faithful)")
    print_classification_report(metrics)

    results = {
        'variant': f'qat_w4a4_{args.scale_mode}',
        'w_bits': 4, 'a_bits': 4, 'qmax': QMAX, 'scale_mode': args.scale_mode,
        'int_acc': float(acc), 'fq_val_acc': float(best_val),
        'fq_int_gap_pp': float((best_val - acc) * 100),
        'f1_macro': float(metrics['f1_macro']),
        'per_class_f1': {k: float(v['f1']) for k, v in metrics['per_class'].items()},
        'epochs': args.epochs, 'lr': args.lr, **meta,
        'note': (f'QAT full INT4 (W4A4), {args.scale_mode} scale. weights+acts clamped '
                 f'[-7,7], bias INT32. Software-only ablation for bit-vs-accuracy gradient '
                 f'in Table 4; no INT4 RTL built. p2 = shift-only (0 DSP rescale, hardware '
                 f'analogue of A2); general = float multiplier rescale (4 DSP, accuracy '
                 f'ceiling of INT4).'),
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {args.output_dir}/results.json")
    print(f"\n  Table 4 row — QAT W4A4 ({args.scale_mode}): acc={acc*100:.2f}%  "
          f"F1={metrics['f1_macro']:.4f}  (vs A2 INT8 94.37%)")


def main():
    p = argparse.ArgumentParser(description='QAT W4A4 ablation (full INT4) for Table 4')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--data_dir', type=str, default='../../data/Chapman')
    p.add_argument('--scale-mode', type=str, default='p2', choices=['p2', 'general'])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=128)
    run(p.parse_args())


if __name__ == "__main__":
    main()
