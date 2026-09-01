"""Bit-width sweep INT4 / INT8 / INT16 on the same ningba checkpoint.

Post-training quantization with power-of-2 scales at each width, so the only
variable is the bit-width itself (same weights, same calibration rule, same
round-half-up rescale). This is the fair "how much does the width buy us"
comparison; a per-width QAT would confound width with re-training.

Also reports the hardware cost that follows from the width: multiplier size,
weight ROM bits, accumulator width.

Usage:
  python bitwidth_sweep.py --checkpoint results/ningba/qat_int8/model_qat_int8.pth \
      --npz ../../data/ningba_processed/ningbo_dataset_clip16.npz \
      --out results/ningba/bitwidth
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantization.qat_int8 import ECG_1DCNN_QAT, round_shift

CONV = ['conv1', 'conv2', 'conv3', 'conv4']
N_WEIGHTS = 640          # pruned model parameter count


def shift_bits(abs_max, qmax):
    if abs_max <= 0:
        return 0
    return max(0, min(int(math.floor(math.log2(qmax / abs_max))), 30))


def quantize(model, X_cal, qmax):
    """PTQ power-of-2 at the given qmax. Returns int weights + shifts."""
    w_i, b_f, w_sh = {}, {}, {}
    for name in CONV + ['fc']:
        L = getattr(model, name)
        w = L.weight.data.cpu().numpy()
        n = shift_bits(max(abs(w.min()), abs(w.max())), qmax)
        w_sh[name] = n
        w_i[name] = np.clip(np.round(w * (2.0 ** n)), -qmax, qmax)
        b_f[name] = L.bias.data.cpu().numpy() if L.bias is not None else np.zeros(w.shape[0], np.float32)

    ishift = shift_bits(float(np.abs(X_cal).max()), qmax)
    nb = {c: (ishift + w_sh[c] if c == 'conv1' else w_sh[c]) for c in CONV}
    return w_i, b_f, w_sh, nb, ishift


def forward(model, x, w_i, b_f, w_sh, nb, ishift, qmax):
    """Bit-exact integer forward at the given qmax (RTL-shaped pipeline)."""
    h = torch.clamp(torch.round(x * (2.0 ** ishift)), -qmax, qmax)
    for i, name in enumerate(CONV, 1):
        w = torch.tensor(w_i[name].astype(np.float32))
        n = nb[name]
        b = torch.tensor(np.round(b_f[name] * (2.0 ** n)).astype(np.float32))
        acc = F.conv1d(h, w, b, padding=getattr(model, name).padding)
        h = torch.clamp(round_shift(acc, n), -qmax, qmax)
        if name == 'conv4':
            h = torch.clamp(h, min=0)
        h = getattr(model, f'pool{i}')(h)
    g = torch.floor(h.sum(dim=-1) / 4.0)
    wfc = torch.tensor(w_i['fc'].astype(np.float32))
    bfc = torch.tensor(np.round(b_f['fc'] * (2.0 ** w_sh['fc'])).astype(np.float32))
    return F.linear(g, wfc, bfc)


def hw_cost(bits):
    """Cost that follows directly from the width (multiplier + storage)."""
    # acc must hold sum of 5 taps x in_ch(<=8) products of (bits x bits)
    prod = 2 * bits
    acc = prod + math.ceil(math.log2(5 * 8))
    return dict(bits=bits,
                multiplier=f'{bits}x{bits}',
                acc_width=acc,
                weight_rom_bits=N_WEIGHTS * bits,
                weight_rom_ratio=round(bits / 8.0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--clip', type=float, default=16.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    m = ECG_1DCNN_QAT(c1_out=ck['c1_out'], c2_out=ck['c2_out'],
                      c3_out=ck['c3_out'], c4_out=ck['c4_out'])
    m.load_state_dict(ck['model_state_dict'])
    m.eval()

    d = np.load(args.npz)
    Xtr = np.clip(d['X_train'][:2000], -args.clip, args.clip).astype(np.float32)
    X = np.clip(d['X_test'], -args.clip, args.clip).astype(np.float32)
    y = d['y_test'].astype(np.int64)
    xt = torch.from_numpy(X).unsqueeze(1)

    rows = []
    with torch.no_grad():
        fl = []
        for i in range(0, len(X), 256):
            fl.append(m(xt[i:i + 256], quantize=False).numpy())
        fp = np.concatenate(fl).argmax(1)
        rows.append(dict(name='float32', bits=32,
                         acc=float(accuracy_score(y, fp)),
                         f1_macro=float(f1_score(y, fp, average='macro')),
                         hw=dict(bits=32, multiplier='float32',
                                 acc_width=32, weight_rom_bits=N_WEIGHTS * 32,
                                 weight_rom_ratio=4.0)))

        for bits in (4, 8, 16):
            qmax = 2 ** (bits - 1) - 1
            w_i, b_f, w_sh, nb, ish = quantize(m, Xtr, qmax)
            lg = []
            for i in range(0, len(X), 256):
                lg.append(forward(m, xt[i:i + 256], w_i, b_f, w_sh, nb, ish, qmax).numpy())
            p = np.concatenate(lg).argmax(1)
            rows.append(dict(name=f'INT{bits}', bits=bits,
                             acc=float(accuracy_score(y, p)),
                             f1_macro=float(f1_score(y, p, average='macro')),
                             qmax=qmax, input_shift=ish,
                             w_shift=w_sh, nb=nb, hw=hw_cost(bits)))

    print(f"[INFO] test n={len(y)}  (PTQ power-of-2, same checkpoint)\n")
    print(f"{'variant':<9}{'acc %':>8}{'F1-macro':>10}{'mult':>10}"
          f"{'acc_w':>7}{'ROM bits':>10}{'ROM x':>7}")
    for r in rows:
        print(f"{r['name']:<9}{r['acc']*100:>8.2f}{r['f1_macro']:>10.4f}"
              f"{r['hw']['multiplier']:>10}{r['hw']['acc_width']:>7}"
              f"{r['hw']['weight_rom_bits']:>10}{r['hw']['weight_rom_ratio']:>7}")

    with open(os.path.join(args.out, 'bitwidth_sweep.json'), 'w') as f:
        json.dump(dict(n_test=int(len(y)), rows=rows), f, indent=2)
    print(f"\n[OK] -> {args.out}/bitwidth_sweep.json")


if __name__ == '__main__':
    main()
