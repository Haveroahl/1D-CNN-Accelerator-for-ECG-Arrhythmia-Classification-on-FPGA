"""TH2 build — PTB-XL train domain with SR subsampled to N BEFORE splitting.

Source: data/ptbxl_by_class/<CLASS>/*.npy (already preprocessed, 19,952).
Step: random-drop SR (class 3) down to --keep_sr (default 2567) over the WHOLE
      PTB-XL set, THEN record-level split 70/15/15 (seed=42).
Cache -> data/case_study/case2_ptbxl_sr<N>.npz.

External test set (Chapman+Ningbo merged) is NOT built here — TH2 train reuses
the X_all/y_all/src_all already cached in case1_merged.npz.

Usage:
    python cross_eval/case2_build.py
"""

import os, glob, argparse
import numpy as np

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
PTBXL_ROOT  = r'data/ptbxl_by_class'
OUT_DIR     = r'data/case_study'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep_sr', type=int, default=2567)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    X, y = [], []
    for label, c in enumerate(CLASS_NAMES):
        files = sorted(glob.glob(os.path.join(PTBXL_ROOT, c, '*.npy')))
        if c == 'SR' and args.keep_sr < len(files):
            idx = np.sort(rng.choice(len(files), size=args.keep_sr, replace=False))
            files = [files[i] for i in idx]
        for f in files:
            X.append(np.load(f)); y.append(label)
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.int64)
    print(f"[INFO] PTB-XL after SR->{args.keep_sr}: {len(y)}  "
          f"{dict(zip(CLASS_NAMES, np.bincount(y, minlength=4)))}")

    # record-level 70/15/15
    perm = rng.permutation(len(y))
    n = len(perm)
    tr, va, te = perm[:int(.70*n)], perm[int(.70*n):int(.85*n)], perm[int(.85*n):]
    for nm, idx in [('train', tr), ('val', va), ('test', te)]:
        print(f"  {nm}: {len(idx)}  {dict(zip(CLASS_NAMES, np.bincount(y[idx], minlength=4)))}")

    out = os.path.join(OUT_DIR, f'case2_ptbxl_sr{args.keep_sr}.npz')
    np.savez(out, X_train=X[tr], y_train=y[tr],
             X_val=X[va], y_val=y[va], X_test=X[te], y_test=y[te])
    print(f"[INFO] saved: {out}")


if __name__ == '__main__':
    main()
