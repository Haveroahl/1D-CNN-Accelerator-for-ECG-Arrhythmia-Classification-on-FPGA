"""TH1 build — preprocess the FULL merged Chapman+Ningbo tree into one npz.

Source: data/ningba/WFDBRecords/**/*.hea  (45,152 records, NO JS cut).
Label: SNOMED #Dx -> 4-class via SNOMED_TO_4CLASS (verified mapping).
Signal: Lead II, resample 500->250 Hz (5000->2500) float64, z-score, float32 last
        — identical numeric path to utils/dataset.py / ningbo_preprocess.py.

Record-level patient-independent split 70/15/15 (seed=42).
Caches to data/case_study/case1_merged.npz so TH2 external eval reuses it.

Usage:
    python cross_eval/case1_build.py
"""

import os, glob, argparse
import numpy as np
from scipy.signal import resample
from collections import Counter

SNOMED_TO_4CLASS = {
    '164889003': 0, '164890007': 0,
    '427084000': 1, '426761007': 1, '713422000': 1,
    '233896004': 1, '233897008': 1, '195101003': 1,
    '426177001': 2,
    '426783006': 3, '427393009': 3,
}
CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
ORIG_FS, TARGET_FS, TARGET_LEN, LEAD_IDX = 500, 250, 2500, 1
CHAPMAN_MAX = 10646   # JS<=this = Chapman-half, >this = Ningbo-half (source tag only)


def jsid(p):
    return int(os.path.splitext(os.path.basename(p))[0][2:])


def parse_dx(p):
    with open(p) as f:
        for line in f:
            if line.startswith('#Dx:'):
                return [c.strip() for c in line.split(':', 1)[1].strip().split(',')]
    return []


def get_label(codes):
    cl = {SNOMED_TO_4CLASS[c] for c in codes if c in SNOMED_TO_4CLASS}
    return cl.pop() if len(cl) == 1 else None


def preprocess_signal(sig):
    down = resample(sig, TARGET_LEN)
    mu, std = np.mean(down), np.std(down) + 1e-8
    return ((down - mu) / std).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=r'd:\Thesis101\data')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    import wfdb
    merged = os.path.join(args.data_dir, 'ningba', 'WFDBRecords')
    out_dir = os.path.join(args.data_dir, 'case_study')
    os.makedirs(out_dir, exist_ok=True)

    heas = sorted(glob.glob(os.path.join(merged, '**', '*.hea'), recursive=True))
    print(f"[INFO] {len(heas)} .hea in merged tree")

    X, y, src = [], [], []     # src: 0=Chapman-half, 1=Ningbo-half
    skipped_nolabel = skipped_err = 0
    for i, h in enumerate(heas):
        label = get_label(parse_dx(h))
        if label is None:
            skipped_nolabel += 1
            continue
        try:
            rec = wfdb.rdrecord(h[:-4])
            sig = rec.p_signal[:, LEAD_IDX].astype(np.float64)
            proc = preprocess_signal(sig)
        except Exception:
            skipped_err += 1
            continue
        X.append(proc)
        y.append(label)
        src.append(0 if jsid(h) <= CHAPMAN_MAX else 1)
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1}/{len(heas)}  kept={len(y)}")

    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.int64)
    src = np.asarray(src, np.int8)
    print(f"\n[INFO] kept={len(y)}  skipped_nolabel={skipped_nolabel}  skipped_err={skipped_err}")
    print(f"[INFO] class dist (all): {dict(zip(CLASS_NAMES, np.bincount(y, minlength=4)))}")
    print(f"[INFO] source: Chapman-half={int((src==0).sum())}  Ningbo-half={int((src==1).sum())}")

    # ── record-level 70/15/15 split ──────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(y))
    n = len(perm)
    tr = perm[:int(0.70 * n)]
    va = perm[int(0.70 * n):int(0.85 * n)]
    te = perm[int(0.85 * n):]
    for nm, idx in [('train', tr), ('val', va), ('test', te)]:
        print(f"  {nm}: {len(idx)}  {dict(zip(CLASS_NAMES, np.bincount(y[idx], minlength=4)))}")

    out = os.path.join(out_dir, 'case1_merged.npz')
    np.savez(out,
             X_train=X[tr], y_train=y[tr], src_train=src[tr],
             X_val=X[va],   y_val=y[va],   src_val=src[va],
             X_test=X[te],  y_test=y[te],  src_test=src[te],
             # full pool kept too, for TH2 external eval (no split needed there)
             X_all=X, y_all=y, src_all=src)
    print(f"\n[INFO] saved: {out}")


if __name__ == '__main__':
    main()
