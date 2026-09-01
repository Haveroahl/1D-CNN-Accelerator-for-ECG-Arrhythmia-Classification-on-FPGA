"""Per-layer quantization error: where does INT8 lose information?

Runs the float32 and the bit-exact INT8 path side by side on the same inputs
and compares the two activation tensors after every stage. The INT8 tensor is
de-quantized (x_int / 2^s, s = accumulated output scale) so both live on the
same physical scale and the comparison is meaningful.

Metrics per stage:
  SQNR (dB)   10*log10(sum(ref^2) / sum(err^2))  — higher = cleaner
  NRMSE       rms(err) / rms(ref)
  cos_sim     directional agreement (survives a scale mismatch)
  sat_rate    fraction of INT8 values pinned at +/-127 (clamp saturation)

Usage:
  python layer_quant_error.py --checkpoint results/ningba/qat_int8/model_qat_int8.pth \
      --npz ../../data/ningba_processed/ningbo_dataset_clip16.npz \
      --out results/ningba/quant_error
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantization.qat_int8 import ECG_1DCNN_QAT, round_shift

LAYERS = ['conv1', 'conv2', 'conv3', 'conv4']


def float_stages(m, x):
    """Float32 activations after each conv+pool, plus GAP and logits."""
    out = {}
    h = x
    for i, name in enumerate(LAYERS, 1):
        h = getattr(m, name)(h)
        if name == 'conv4':
            h = torch.clamp(h, min=0)           # ReLU only after conv4
        h = getattr(m, f'pool{i}')(h)
        out[f'pool{i}'] = h
    g = h.mean(dim=-1)
    out['gap'] = g
    out['logits'] = m.fc(g)
    return out


def int8_stages(m, x, w8, b8, nb, wsh, ish):
    """Bit-exact INT8 activations + the scale (2^s) each one carries."""
    out, scales = {}, {}
    xq = torch.clamp(torch.round(x * (2.0 ** ish)), -127, 127)
    s = ish                                     # current activation scale exponent
    h = xq
    for i, name in enumerate(LAYERS, 1):
        w = torch.tensor(w8[name].astype(np.float32))
        n = nb[name]
        b = torch.tensor(np.round(b8[name] * (2.0 ** n)).astype(np.float32))
        acc = F.conv1d(h, w, b, padding=getattr(m, name).padding)
        # acc scale = s + wsh; rescale by >>nb leaves s + wsh - nb
        h = torch.clamp(round_shift(acc, n), -127, 127)
        s = s + wsh[name] - n
        if name == 'conv4':
            h = torch.clamp(h, min=0)
        h = getattr(m, f'pool{i}')(h)
        out[f'pool{i}'], scales[f'pool{i}'] = h, s
    g = torch.floor(h.sum(dim=-1) / 4.0)        # RTL integer GAP
    out['gap'], scales['gap'] = g, s
    wfc = torch.tensor(w8['fc'].astype(np.float32))
    bfc = torch.tensor(np.round(b8['fc'] * (2.0 ** wsh['fc'])).astype(np.float32))
    out['logits'] = F.linear(g, wfc, bfc)
    scales['logits'] = s + wsh['fc']
    return out, scales


def compare(ref, qt, scale):
    """ref = float tensor, qt = INT8 tensor, scale = its exponent."""
    deq = qt / (2.0 ** scale)
    r, e = ref.flatten(), (deq - ref).flatten()
    ss, se = float((r ** 2).sum()), float((e ** 2).sum())
    sqnr = 10 * np.log10(ss / se) if se > 0 else float('inf')
    nrmse = float(e.pow(2).mean().sqrt() / r.pow(2).mean().sqrt())
    cos = float(F.cosine_similarity(deq.flatten(), ref.flatten(), dim=0))
    sat = float((qt.abs() >= 127).float().mean())
    return dict(sqnr_db=float(sqnr), nrmse=nrmse, cos_sim=cos,
                sat_rate=sat, max_abs_int8=float(qt.abs().max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--clip', type=float, default=16.0)
    ap.add_argument('--n', type=int, default=1000, help='test samples to use')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    m = ECG_1DCNN_QAT(c1_out=ck['c1_out'], c2_out=ck['c2_out'],
                      c3_out=ck['c3_out'], c4_out=ck['c4_out'])
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    w8 = {k: np.array(v, dtype=np.int8) for k, v in ck['w_int8'].items()}
    b8 = {k: np.array(v, dtype=np.float32) for k, v in ck['b_int8'].items()}

    d = np.load(args.npz)
    X = np.clip(d['X_test'][:args.n], -args.clip, args.clip).astype(np.float32)
    x = torch.from_numpy(X).unsqueeze(1)

    with torch.no_grad():
        fs = float_stages(m, x)
        qs, sc = int8_stages(m, x, w8, b8, ck['nb'], ck['w_shift'],
                             ck['input_shift_bits'])

    # weight quantization error per layer (independent of activations)
    wq = {}
    for name in LAYERS + ['fc']:
        wf = getattr(m, name).weight.detach()
        wi = torch.tensor(w8[name].astype(np.float32)) / (2.0 ** ck['w_shift'][name])
        e = (wi - wf).flatten()
        wq[name] = dict(
            sqnr_db=float(10 * np.log10(float((wf ** 2).sum()) / float((e ** 2).sum()))),
            max_abs_int8=float(np.abs(w8[name]).max()),
            w_shift=int(ck['w_shift'][name]),
            n_weights=int(w8[name].size))

    act = {}
    for k in ['pool1', 'pool2', 'pool3', 'pool4', 'gap', 'logits']:
        act[k] = compare(fs[k], qs[k], sc[k])
        act[k]['scale_exp'] = int(sc[k])
        if k.startswith('pool'):
            act[k]['nb'] = int(ck['nb'][LAYERS[int(k[-1]) - 1]])

    print(f"[INFO] {args.n} samples\n")
    print("=== Weight quantization (INT8 vs float32) ===")
    print(f"{'layer':<8}{'w_shift':>9}{'max|w_int8|':>13}{'SQNR dB':>10}{'#w':>7}")
    for k, v in wq.items():
        print(f"{k:<8}{v['w_shift']:>9}{v['max_abs_int8']:>13.0f}"
              f"{v['sqnr_db']:>10.2f}{v['n_weights']:>7}")

    print("\n=== Activation error after each stage ===")
    print(f"{'stage':<9}{'nb':>4}{'SQNR dB':>10}{'NRMSE':>9}{'cos':>8}"
          f"{'sat%':>8}{'max|q|':>8}")
    for k, v in act.items():
        print(f"{k:<9}{v.get('nb', ''):>4}{v['sqnr_db']:>10.2f}{v['nrmse']:>9.4f}"
              f"{v['cos_sim']:>8.5f}{v['sat_rate']*100:>8.2f}{v['max_abs_int8']:>8.0f}")

    with open(os.path.join(args.out, 'layer_quant_error.json'), 'w') as f:
        json.dump(dict(n_samples=args.n, weights=wq, activations=act), f, indent=2)
    print(f"\n[OK] -> {args.out}/layer_quant_error.json")


if __name__ == '__main__':
    main()
