"""
Generate hardware-simulation golden for a test set.

For each dataset produces, under <out>/:
  label_golden/<CLASS>/   : one representative sample per class (4 total)
      input_int8.mem, after_pool1..4.mem, after_gap.mem, logits_fc.mem,
      ecg_sampleX.hex, meta.json   -> feeds RTL per-label simulation test
  bitexact/               : copy of ONE of the four (first correct) for the
      21-checkpoint bit-exact tb                        -> C2 bit-exact
  fullset/                : ecg hex per sample + expected_argmax.hex (int8x int8
      bit-exact prediction for EVERY test sample)       -> RTL vs software acc

The INT8 forward here is generate_golden.int8_forward_golden (bit-exact GAP
floor), so expected_argmax equals what the RTL emits per sample.

Data source (one of):
  --npz PATH     : X_test / y_test
  --byclass DIR  : folder-per-class .npy tree (georgia_by_class); whole = test

Usage:
  python gen_test_golden.py --checkpoint results/ningba/qat_int8/model_qat_int8.pth \
      --npz ../../data/ningba_processed/ningbo_dataset.npz --out results/ningba/test_golden
"""
import os, sys, json, glob, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_golden import (int8_forward_golden, write_int8_mem,
                             write_int32_mem, to_hex2)
from quantization.qat_int8 import ECG_1DCNN_QAT

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']

STAGES = [
    ('input_int8',  'input_int8.mem',  'int8'),
    ('after_pool1', 'after_pool1.mem', 'int8'),
    ('after_pool2', 'after_pool2.mem', 'int8'),
    ('after_pool3', 'after_pool3.mem', 'int8'),
    ('after_pool4', 'after_pool4.mem', 'int8'),
    ('after_gap',   'after_gap.mem',   'int8'),
    ('logits_fc',   'logits_fc.mem',   'int32'),
]


def write_ecg_hex(path, x_raw, input_shift):
    """ECG sample as INT8 hex (2500 lines) — same quant as RTL input_sram load."""
    q = torch.clamp(torch.round(x_raw * (2.0 ** input_shift)), -127, 127)
    q = q.cpu().numpy().astype(np.int32).flatten()
    with open(path, 'w') as f:
        for v in q:
            f.write(to_hex2(v) + "\n")


def load_data(args):
    if args.npz:
        d = np.load(args.npz)
        X, y = d['X_test'].astype(np.float32), d['y_test'].astype(np.int64)
    else:
        X, y = [], []
        for cidx, cname in enumerate(CLASS_NAMES):
            for f in sorted(glob.glob(os.path.join(args.byclass, cname, '*.npy'))):
                X.append(np.load(f)); y.append(cidx)
        X, y = np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)
    if args.clip > 0:                       # match training preprocess clip
        X = np.clip(X, -args.clip, args.clip).astype(np.float32)
    return X, y


def dump_stages(stages, out_dir, x_raw, input_shift, sample_tag):
    os.makedirs(out_dir, exist_ok=True)
    for key, fname, dtype in STAGES:
        t = stages[key]
        p = os.path.join(out_dir, fname)
        (write_int8_mem if dtype == 'int8' else write_int32_mem)(p, t)
    write_ecg_hex(os.path.join(out_dir, f'{sample_tag}.hex'), x_raw, input_shift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--npz', default=None)
    ap.add_argument('--byclass', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--clip', type=float, default=16.0,
                    help='Clip input to ±clip (match training preprocess); 0=off')
    args = ap.parse_args()
    device = torch.device('cpu')

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    qat = ECG_1DCNN_QAT(c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
                        c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'])
    qat.load_state_dict(ckpt['model_state_dict']); qat.eval()
    w_int8 = {k: np.array(v, dtype=np.int8)    for k, v in ckpt['w_int8'].items()}
    b_int8 = {k: np.array(v, dtype=np.float32) for k, v in ckpt['b_int8'].items()}
    nb, w_shift, ishift = ckpt['nb'], ckpt['w_shift'], ckpt['input_shift_bits']

    X, y = load_data(args)
    print(f"[INFO] test set: {len(y)} samples, dist={np.bincount(y, minlength=4).tolist()}")

    def run_one(idx):
        xs = torch.from_numpy(X[idx]).unsqueeze(0)
        with torch.no_grad():
            st = int8_forward_golden(qat, xs, w_int8, b_int8, nb, w_shift, ishift, device)
        return st, xs

    # ---- (1) label_golden: 1 representative correct-pred sample per class ----
    label_dir = os.path.join(args.out, 'label_golden')
    bitexact_dir = os.path.join(args.out, 'bitexact')
    picked = {}
    for cidx, cname in enumerate(CLASS_NAMES):
        cand = np.where(y == cidx)[0]
        chosen = None
        for idx in cand:                       # prefer a correctly-classified one
            st, xs = run_one(idx)
            if st['predicted_class'] == cidx:
                chosen = (idx, st, xs); break
        if chosen is None and len(cand):       # fallback: first of class
            idx = cand[0]; st, xs = run_one(idx); chosen = (idx, st, xs)
        if chosen is None:
            print(f"[WARN] no samples for class {cname}"); continue
        idx, st, xs = chosen
        picked[cname] = dict(idx=int(idx), true=cidx, pred=st['predicted_class'])
        dump_stages(st, os.path.join(label_dir, cname), xs.squeeze(0), ishift, f'ecg_{cname}')
        print(f"  [{cname}] idx={idx} pred={st['predicted_class']} "
              f"({'OK' if st['predicted_class']==cidx else 'WRONG'})")

    # ---- (2) bitexact: reuse the first class's picked sample ----
    first = next(iter(picked))
    st, xs = run_one(picked[first]['idx'])
    dump_stages(st, bitexact_dir, xs.squeeze(0), ishift, 'ecg_bitexact')
    with open(os.path.join(bitexact_dir, 'meta.json'), 'w') as f:
        json.dump(dict(sample_idx=picked[first]['idx'], true_class=picked[first]['true'],
                       predicted_class=st['predicted_class'], input_shift_bits=ishift,
                       nb=nb, stage_files={k: fn for k, fn, _ in STAGES}), f, indent=2)
    with open(os.path.join(label_dir, 'picked.json'), 'w') as f:
        json.dump(picked, f, indent=2)

    # ---- (3) fullset: ecg hex + expected argmax for EVERY test sample ----
    full_dir = os.path.join(args.out, 'fullset')
    os.makedirs(full_dir, exist_ok=True)
    # batched argmax via int8_eval_batch path (identical bit-exact formula;
    # int8_forward_golden squeezes to a single sample so is not batch-usable here)
    from int8_eval_batch import int8_forward_bitexact
    with torch.no_grad():
        chunks = []
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i+256])
            chunks.append(int8_forward_bitexact(qat, xb, w_int8, b_int8, nb, w_shift, ishift).argmax(1).numpy())
    preds = np.concatenate(chunks).astype(np.uint8)
    acc = float((preds == y).mean())
    with open(os.path.join(full_dir, 'expected_argmax.hex'), 'w') as f:
        for v in preds:
            f.write(f"{int(v):01x}\n")
    np.save(os.path.join(full_dir, 'labels.npy'), y.astype(np.uint8))
    np.save(os.path.join(full_dir, 'int8_argmax.npy'), preds)
    print(f"[INFO] fullset INT8 bit-exact acc vs labels = {acc:.4f} ({(preds==y).sum()}/{len(y)})")
    print(f"[INFO] wrote label_golden/(4 classes) + bitexact/ + fullset/ under {args.out}")


if __name__ == '__main__':
    main()
