"""MIT-BIH Arrhythmia Database — Download & Preprocess

Pipeline:
  1. Download via wfdb (PhysioNet)
  2. Resample 360 Hz → 250 Hz, Lead MLII (channel 0)
  3. Segment into 10s windows (2500 samples @ 250 Hz)
  4. Map AAMI beat labels → 4-class (N/S/V/F/Q → SR/GSVT/GSVT/drop/drop)
  5. Patient-independent split 70/15/15
  6. Save preprocessed dataset as .npz

Class mapping AAMI → Chapman 4-class:
  N (Normal)      → SR   (3): N, L, R, e, j
  S (Supra)       → GSVT (1): A, a, J, S
  V (Ventricular) → GSVT (1): V, E  [merged for 4-class compat, documented in paper]
  F (Fusion)      → drop
  Q (Unclassified)→ drop

Usage:
    python cross_eval/mitbih_preprocess.py --data_dir d:\\Thesis101\\data
"""

import os
import argparse
import numpy as np
from scipy.signal import resample
from collections import defaultdict

# AAMI beat-type to 4-class
# F and Q dropped (ambiguous, cannot map to Chapman 4-class cleanly)
AAMI_TO_4CLASS = {
    # N class → SR (3)
    'N': 3, 'L': 3, 'R': 3, 'e': 3, 'j': 3,
    # S class → GSVT (1)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    # V class → GSVT (1) [merged per AAMI grouping, noted in paper]
    'V': 1, 'E': 1,
    # F, Q → drop
}

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']

MITBIH_RECORDS = [
    '100','101','102','103','104','105','106','107',
    '108','109','111','112','113','114','115','116',
    '117','118','119','121','122','123','124','200',
    '201','202','203','205','207','208','209','210',
    '212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234',
]

ORIG_FS   = 360
TARGET_FS = 250
WIN_LEN   = TARGET_FS * 10   # 2500 samples @ 250 Hz
ORIG_WIN  = ORIG_FS * 10     # 3600 samples @ 360 Hz


def download_mitbih(data_dir):
    import wfdb
    db_dir = os.path.join(data_dir, 'mitbih')
    os.makedirs(db_dir, exist_ok=True)
    print(f"[INFO] Downloading MIT-BIH to {db_dir} ...")
    wfdb.dl_database('mitdb', dl_dir=db_dir)
    print("[INFO] Download complete.")
    return db_dir


def classify_window_beats(annotations, win_start, win_end):
    """Majority beat-label vote within window, returns 4-class label or None."""
    counts = defaultdict(int)
    for i, sample in enumerate(annotations.sample):
        if win_start <= sample < win_end:
            symbol = annotations.symbol[i]
            cls = AAMI_TO_4CLASS.get(symbol)
            if cls is not None:
                counts[cls] += 1
    if not counts:
        return None
    # Dominant class in window
    return max(counts, key=counts.get)


def preprocess_record(record_id, db_dir):
    import wfdb
    path = os.path.join(db_dir, record_id)
    try:
        rec = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, 'atr')
    except Exception as e:
        print(f"[WARN] {record_id}: {e}")
        return [], []

    # Lead MLII is channel 0 for most MIT-BIH records
    sig = rec.p_signal[:, 0].astype(np.float64)
    total_orig_samples = len(sig)

    segments = []
    labels   = []

    # Slide non-overlapping 10s windows
    for win_start_orig in range(0, total_orig_samples - ORIG_WIN + 1, ORIG_WIN):
        win_end_orig = win_start_orig + ORIG_WIN
        win_sig = sig[win_start_orig:win_end_orig]

        # Class label from beat annotations in this window
        cls = classify_window_beats(ann, win_start_orig, win_end_orig)
        if cls is None:
            continue

        # Resample 360 → 250 Hz
        win_resampled = resample(win_sig, WIN_LEN)

        # Z-score normalize (match Chapman pipeline)
        mu    = np.mean(win_resampled)
        sigma = np.std(win_resampled) + 1e-8
        win_norm = (win_resampled - mu) / sigma

        segments.append(win_norm.astype(np.float32))
        labels.append(cls)

    return segments, labels


def patient_split(record_ids, segments_by_rec, labels_by_rec, seed=42):
    """Patient-independent split: 70% train / 15% val / 15% test by record."""
    rng = np.random.default_rng(seed)
    recs = [r for r in record_ids if r in segments_by_rec]
    perm = rng.permutation(len(recs))
    n = len(perm)
    train_recs = [recs[i] for i in perm[:int(0.70 * n)]]
    val_recs   = [recs[i] for i in perm[int(0.70 * n):int(0.85 * n)]]
    test_recs  = [recs[i] for i in perm[int(0.85 * n):]]

    def collect(rec_list):
        segs, labs = [], []
        for r in rec_list:
            segs.extend(segments_by_rec[r])
            labs.extend(labels_by_rec[r])
        return np.array(segs, dtype=np.float32), np.array(labs, dtype=np.int64)

    return (collect(train_recs), collect(val_recs), collect(test_recs),
            train_recs, val_recs, test_recs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=r'd:\Thesis101\data')
    p.add_argument('--skip_download', action='store_true',
                   help='Skip download if already present')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    db_dir  = os.path.join(args.data_dir, 'mitbih')
    out_dir = os.path.join(args.data_dir, 'mitbih_processed')
    os.makedirs(out_dir, exist_ok=True)

    # 1. Download
    if not args.skip_download:
        download_mitbih(args.data_dir)
    else:
        print(f"[INFO] Skipping download, using {db_dir}")

    # 2. Preprocess all records
    segments_by_rec = {}
    labels_by_rec   = {}
    total_segs = 0

    for rec_id in MITBIH_RECORDS:
        segs, labs = preprocess_record(rec_id, db_dir)
        if segs:
            segments_by_rec[rec_id] = segs
            labels_by_rec[rec_id]   = labs
            total_segs += len(segs)
            dist = {CLASS_NAMES[c]: labs.count(c) for c in set(labs)}
            print(f"  {rec_id}: {len(segs)} windows  {dist}")

    print(f"\n[INFO] Total segments: {total_segs}")

    # 3. Patient-independent split
    (X_train, y_train), (X_val, y_val), (X_test, y_test), \
    train_recs, val_recs, test_recs = patient_split(
        MITBIH_RECORDS, segments_by_rec, labels_by_rec, seed=args.seed
    )

    for split, X, y in [('train', X_train, y_train),
                         ('val',   X_val,   y_val),
                         ('test',  X_test,  y_test)]:
        dist = {CLASS_NAMES[c]: int((y == c).sum()) for c in range(4) if (y == c).sum() > 0}
        print(f"  {split}: {len(y)} samples  {dist}")

    # 4. Save
    out_path = os.path.join(out_dir, 'mitbih_dataset.npz')
    np.savez(out_path,
             X_train=X_train, y_train=y_train,
             X_val=X_val,     y_val=y_val,
             X_test=X_test,   y_test=y_test,
             train_recs=train_recs,
             val_recs=val_recs,
             test_recs=test_recs)
    print(f"\n[INFO] Saved → {out_path}")
    print(f"       X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")


if __name__ == '__main__':
    main()
