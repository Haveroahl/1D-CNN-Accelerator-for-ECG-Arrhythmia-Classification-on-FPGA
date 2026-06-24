"""Rebuild merged Chapman+Ningbo npz WITH an SVT subclass mask.

Identical pipeline to case1_build.py (same sorted glob, same get_label, same
wfdb signal load + preprocess, same 70/15/15 split seed=42) but additionally
records, per kept record, whether its #Dx contains the SVT SNOMED code
(426761007). Loading signals guarantees index alignment (header-only replay
was off-by-one due to a wfdb load failure on one GSVT record).

Saves data/case_study/case1_merged_svtmask.npz with the same arrays plus
svt_train/svt_val/svt_test/svt_all.

Usage:
    python cross_eval/svt_build_mask.py
"""

import os, glob
import numpy as np
from scipy.signal import resample

SNOMED_TO_4CLASS = {
    '164889003': 0, '164890007': 0,
    '427084000': 1, '426761007': 1, '713422000': 1,
    '233896004': 1, '233897008': 1, '195101003': 1,
    '426177001': 2,
    '426783006': 3, '427393009': 3,
}
CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
TARGET_LEN, LEAD_IDX = 2500, 1
CHAPMAN_MAX = 10646
SVT_CODE = '426761007'
DATA = r'd:\Thesis101\data'


def jsid(p): return int(os.path.splitext(os.path.basename(p))[0][2:])
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
    import wfdb
    merged = os.path.join(DATA, 'ningba', 'WFDBRecords')
    out_dir = os.path.join(DATA, 'case_study')
    os.makedirs(out_dir, exist_ok=True)

    heas = sorted(glob.glob(os.path.join(merged, '**', '*.hea'), recursive=True))
    print(f"[INFO] {len(heas)} .hea in merged tree")

    X, y, src, svt = [], [], [], []
    skipped_nolabel = skipped_err = 0
    for i, h in enumerate(heas):
        codes = parse_dx(h)
        label = get_label(codes)
        if label is None:
            skipped_nolabel += 1; continue
        try:
            rec = wfdb.rdrecord(h[:-4])
            sig = rec.p_signal[:, LEAD_IDX].astype(np.float64)
            proc = preprocess_signal(sig)
        except Exception:
            skipped_err += 1; continue
        X.append(proc); y.append(label)
        src.append(0 if jsid(h) <= CHAPMAN_MAX else 1)
        svt.append(1 if SVT_CODE in codes else 0)
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1}/{len(heas)}  kept={len(y)}")

    X = np.asarray(X, np.float32); y = np.asarray(y, np.int64)
    src = np.asarray(src, np.int8); svt = np.asarray(svt, np.int8)
    print(f"\n[INFO] kept={len(y)}  skipped_nolabel={skipped_nolabel}  skipped_err={skipped_err}")
    print(f"[INFO] class dist: {dict(zip(CLASS_NAMES, np.bincount(y, minlength=4)))}")
    print(f"[INFO] SVT records: {int(svt.sum())}")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y)); n = len(perm)
    tr = perm[:int(0.70 * n)]; va = perm[int(0.70 * n):int(0.85 * n)]; te = perm[int(0.85 * n):]
    print(f"[INFO] SVT in train={int(svt[tr].sum())} val={int(svt[va].sum())} test={int(svt[te].sum())}")

    out = os.path.join(out_dir, 'case1_merged_svtmask.npz')
    np.savez(out,
             X_train=X[tr], y_train=y[tr], svt_train=svt[tr],
             X_val=X[va],   y_val=y[va],   svt_val=svt[va],
             X_test=X[te],  y_test=y[te],  svt_test=svt[te])
    print(f"[INFO] saved: {out}")


if __name__ == '__main__':
    main()
