"""Georgia 12-lead (PhysioNet 2020, Emory G12EC) — Preprocess for 4-class ECG.

10,344 12-lead records, 500 Hz, 10s (5000 samples), WFDB .hea/.mat.
Labels: SNOMED-CT codes in the `#Dx:` line. Same taxonomy as Ningbo.
Different acquisition system (Emory, USA) -> a far-transfer test distinct from
PTB-XL (Germany). No JS overlap with Chapman -> no leakage cut needed.

Pipeline (identical numeric path to ningbo_preprocess.py / Chapman):
  Lead II (channel 1) -> resample 500->250 Hz (5000->2500) float64
  -> z-score per record -> float32 last -> <out>/<CLASS>/<rec>.npy

Usage:
    python cross_eval/georgia_preprocess.py --data_dir d:\\Thesis101\\data
"""

import os, glob, argparse
import numpy as np
from scipy.signal import resample
from scipy.io import loadmat
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
    args = ap.parse_args()

    geo_dir = os.path.join(args.data_dir, 'Georgia')
    out_dir = os.path.join(args.data_dir, 'georgia_by_class')
    for c in CLASS_NAMES:
        os.makedirs(os.path.join(out_dir, c), exist_ok=True)

    heas = sorted(glob.glob(os.path.join(geo_dir, '**', '*.hea'), recursive=True))
    print(f"[INFO] found {len(heas)} .hea records")

    dist = Counter()
    skipped_nolabel = skipped_err = 0
    for i, h in enumerate(heas):
        label = get_label(parse_dx(h))
        if label is None:
            skipped_nolabel += 1
            continue
        try:
            # Georgia .mat is MATLAB v5 (key 'val', shape (12, 5000) int16);
            # the .hea record line carries an extra date field that breaks
            # wfdb's parser, so read the waveform directly. Lead II = row 1.
            # z-score downstream removes the 4880/mV gain -> raw int16 is fine.
            val = loadmat(h[:-4] + '.mat')['val']
            sig = val[LEAD_IDX].astype(np.float64)
            proc = preprocess_signal(sig)
        except Exception:
            skipped_err += 1
            continue
        recname = os.path.splitext(os.path.basename(h))[0]
        np.save(os.path.join(out_dir, CLASS_NAMES[label], recname + '.npy'), proc)
        dist[label] += 1
        if (i + 1) % 2000 == 0:
            print(f"  ... {i+1}/{len(heas)}  kept={sum(dist.values())}")

    print(f"\n[INFO] written to {out_dir}")
    for c in range(4):
        print(f"  {CLASS_NAMES[c]}: {dist[c]}")
    print(f"  total labeled: {sum(dist.values())}")
    print(f"  skipped (no rhythm label): {skipped_nolabel}")
    print(f"  skipped (load error):      {skipped_err}")


if __name__ == '__main__':
    main()
