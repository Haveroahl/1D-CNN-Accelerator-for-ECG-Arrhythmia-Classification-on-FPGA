"""Bootstrap confidence intervals + McNemar significance tests.

Answers: is 94.27% a trustworthy number, and is the float32-vs-INT8 gap real
or noise? Both run on stored per-sample predictions, so no re-training.

  CI      : percentile bootstrap (resample the test set with replacement B times,
            recompute the metric, take the 2.5/97.5 percentiles).
  McNemar : exact binomial test on discordant pairs (b, c) of two models
            evaluated on the SAME test set — the correct paired test here.

Usage:
  python stats_ci_mcnemar.py --npz ../../data/ningba_processed/ningbo_dataset_clip16.npz \
      --checkpoint results/ningba/qat_int8/model_qat_int8.pth --tag ningba \
      --out results/ningba/stats
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from int8_eval_batch import CLASS_NAMES, int8_forward_bitexact
from quantization.qat_int8 import ECG_1DCNN_QAT


def bootstrap_ci(y, pred, prob, B, seed, alpha=0.05):
    """Percentile bootstrap CI for acc / F1-macro / macro-AUC."""
    rng = np.random.default_rng(seed)
    n = len(y)
    accs, f1s, aucs = [], [], []
    yb_full = np.eye(4)[y]
    for _ in range(B):
        idx = rng.integers(0, n, n)
        ys, ps = y[idx], pred[idx]
        accs.append(accuracy_score(ys, ps))
        f1s.append(f1_score(ys, ps, average='macro', zero_division=0))
        # AUC needs every class present in the resample; skip degenerate draws
        if len(np.unique(ys)) == 4:
            aucs.append(roc_auc_score(yb_full[idx], prob[idx], average='macro'))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def pack(point, samples):
        s = np.array(samples)
        return dict(point=float(point), lo=float(np.percentile(s, lo)),
                    hi=float(np.percentile(s, hi)), se=float(s.std(ddof=1)),
                    n_boot=len(s))

    return dict(
        acc=pack(accuracy_score(y, pred), accs),
        f1_macro=pack(f1_score(y, pred, average='macro', zero_division=0), f1s),
        macro_auc=pack(roc_auc_score(yb_full, prob, average='macro'), aucs),
    )


def mcnemar(y, pred_a, pred_b):
    """Exact McNemar on the discordant pairs of two models, same test set."""
    ca, cb = pred_a == y, pred_b == y
    b = int(np.sum(ca & ~cb))   # A right, B wrong
    c = int(np.sum(~ca & cb))   # A wrong, B right
    n = b + c
    p = 1.0 if n == 0 else binomtest(b, n, 0.5).pvalue
    return dict(b=b, c=c, n_discordant=n, p_value=float(p),
                acc_a=float(ca.mean()), acc_b=float(cb.mean()),
                delta=float(ca.mean() - cb.mean()),
                significant=bool(p < 0.05))


def load_data(npz, byclass, clip):
    if npz:
        d = np.load(npz)
        X, y = d['X_test'].astype(np.float32), d['y_test'].astype(np.int64)
    else:
        X, y = [], []
        for ci, cn in enumerate(CLASS_NAMES):
            for f in sorted(glob.glob(os.path.join(byclass, cn, '*.npy'))):
                X.append(np.load(f)); y.append(ci)
        X, y = np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)
    if clip > 0:
        X = np.clip(X, -clip, clip).astype(np.float32)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--npz', default=None)
    ap.add_argument('--byclass', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='eval')
    ap.add_argument('--clip', type=float, default=16.0)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    qat = ECG_1DCNN_QAT(c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
                        c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'])
    qat.load_state_dict(ckpt['model_state_dict'])
    qat.eval()
    w8 = {k: np.array(v, dtype=np.int8) for k, v in ckpt['w_int8'].items()}
    b8 = {k: np.array(v, dtype=np.float32) for k, v in ckpt['b_int8'].items()}

    X, y = load_data(args.npz, args.byclass, args.clip)
    print(f"[INFO] {args.tag}: {len(y)} samples")

    il, fl = [], []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256])
            il.append(int8_forward_bitexact(qat, xb, w8, b8, ckpt['nb'],
                                            ckpt['w_shift'],
                                            ckpt['input_shift_bits']).numpy())
            fl.append(qat(xb.unsqueeze(1), quantize=False).numpy())
    il, fl = np.concatenate(il), np.concatenate(fl)
    ip, fp = il.argmax(1), fl.argmax(1)

    def probs(logits):
        return torch.softmax(torch.from_numpy(logits.astype(np.float32)), 1).numpy()

    res = {'n_test': int(len(y)), 'n_boot': args.boot, 'seed': args.seed,
           'ci': {'int8': bootstrap_ci(y, ip, probs(il), args.boot, args.seed),
                  'float32': bootstrap_ci(y, fp, probs(fl), args.boot, args.seed)},
           'mcnemar': {'float32_vs_int8': mcnemar(y, fp, ip)}}

    print(f"\n=== Bootstrap 95% CI ({args.boot} resamples, n={len(y)}) ===")
    for m in ('float32', 'int8'):
        c = res['ci'][m]
        print(f"{m:>8}  acc {c['acc']['point']*100:.2f}% "
              f"[{c['acc']['lo']*100:.2f}, {c['acc']['hi']*100:.2f}]   "
              f"F1 {c['f1_macro']['point']:.4f} "
              f"[{c['f1_macro']['lo']:.4f}, {c['f1_macro']['hi']:.4f}]   "
              f"AUC {c['macro_auc']['point']:.4f} "
              f"[{c['macro_auc']['lo']:.4f}, {c['macro_auc']['hi']:.4f}]")
    m = res['mcnemar']['float32_vs_int8']
    print(f"\n=== McNemar float32 vs INT8 ===")
    print(f"b(float right,int8 wrong)={m['b']}  c(float wrong,int8 right)={m['c']}  "
          f"p={m['p_value']:.4g}  {'SIGNIFICANT' if m['significant'] else 'not significant'}")

    np.save(os.path.join(args.out, f'{args.tag}_float32_argmax.npy'),
            fp.astype(np.uint8))
    with open(os.path.join(args.out, f'{args.tag}_stats.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(f"[OK] -> {args.out}/{args.tag}_stats.json")


if __name__ == '__main__':
    main()
