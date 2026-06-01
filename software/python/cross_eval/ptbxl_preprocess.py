"""PTB-XL Dataset — Preprocess for 4-class ECG Classification

PTB-XL: 21,799 12-lead ECG records, 500 Hz, 10s (5000 samples)
Source: PhysioNet https://physionet.org/content/ptb-xl/

Pipeline:
  1. Read ptbxl_database.csv + scp_statements.csv
  2. Filter records with mappable rhythm label (dominant SCP code)
  3. Load waveform from records500/ (500 Hz) — Lead II = channel 1
  4. Downsample 500 → 250 Hz (5000 → 2500 samples) — match Chapman
  5. Z-score normalize per record
  6. Patient-independent split 70/15/15 (by patient_id)
  7. Save as ptbxl_dataset.npz

Chapman 4-class mapping:
  AFIB (0): AFIB, AFLT
  GSVT (1): STACH, SVTAC, PSVT, SVARR
  SB   (2): SBRAD
  SR   (3): SR

Usage:
    python cross_eval/ptbxl_preprocess.py --data_dir d:\\Thesis101\\data
"""

import os
import ast
import argparse
import numpy as np
import pandas as pd
from scipy.signal import resample
from collections import Counter

SCP_TO_4CLASS = {
    'SR':    3,
    'SBRAD': 2,
    'AFIB':  0,
    'AFLT':  0,
    'STACH': 1,
    'SVTAC': 1,
    'PSVT':  1,
    'SVARR': 1,
}

CLASS_NAMES  = ['AFIB', 'GSVT', 'SB', 'SR']
ORIG_FS      = 500
TARGET_FS    = 250
TARGET_LEN   = TARGET_FS * 10   # 2500
LEAD_IDX     = 1                # Lead II


def get_label(scp_codes_str):
    """Return 4-class label from scp_codes dict string, or None if unmappable."""
    codes = ast.literal_eval(scp_codes_str)
    hits  = {c: v for c, v in codes.items() if c in SCP_TO_4CLASS}
    if not hits:
        return None
    top = max(hits, key=hits.get)
    return SCP_TO_4CLASS[top]


def load_waveform(filename_hr, ptbxl_dir):
    """Load 500 Hz record via wfdb, return Lead II signal (5000 samples)."""
    import wfdb
    path = os.path.join(ptbxl_dir, filename_hr)
    rec  = wfdb.rdrecord(path)
    sig  = rec.p_signal[:, LEAD_IDX].astype(np.float32)
    return sig


def preprocess_signal(sig):
    """Downsample 500→250 Hz, Z-score normalize → float32 (2500,)."""
    down = resample(sig, TARGET_LEN).astype(np.float32)
    mu   = np.mean(down)
    std  = np.std(down) + 1e-8
    return (down - mu) / std


def patient_split(df_labeled, seed=42):
    """Split by patient_id: 70% train / 15% val / 15% test."""
    patients = df_labeled['patient_id'].unique()
    rng      = np.random.default_rng(seed)
    perm     = rng.permutation(len(patients))
    n        = len(perm)

    train_pats = set(patients[perm[:int(0.70 * n)]])
    val_pats   = set(patients[perm[int(0.70 * n):int(0.85 * n)]])
    test_pats  = set(patients[perm[int(0.85 * n):]])

    train_df = df_labeled[df_labeled['patient_id'].isin(train_pats)]
    val_df   = df_labeled[df_labeled['patient_id'].isin(val_pats)]
    test_df  = df_labeled[df_labeled['patient_id'].isin(test_pats)]
    return train_df, val_df, test_df


def build_arrays(df_split, ptbxl_dir, split_name):
    X, y = [], []
    skipped = 0
    for _, row in df_split.iterrows():
        try:
            sig  = load_waveform(row['filename_hr'], ptbxl_dir)
            proc = preprocess_signal(sig)
            X.append(proc)
            y.append(int(row['label4']))
        except Exception as e:
            skipped += 1
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    dist = {CLASS_NAMES[c]: int((y == c).sum()) for c in range(4) if (y == c).sum() > 0}
    print(f"  {split_name}: {len(y)} samples  {dist}  (skipped={skipped})")
    return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=r'd:\Thesis101\data')
    p.add_argument('--seed',     type=int, default=42)
    args = p.parse_args()

    ptbxl_dir = os.path.join(args.data_dir, 'ptbxl')
    out_dir   = os.path.join(args.data_dir, 'ptbxl_processed')
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load metadata
    print("[INFO] Reading ptbxl_database.csv ...")
    df = pd.read_csv(os.path.join(ptbxl_dir, 'ptbxl_database.csv'))
    print(f"       Total records: {len(df)}")

    # 2. Assign 4-class label
    df['label4'] = df['scp_codes'].apply(get_label)
    df_labeled   = df[df['label4'].notna()].copy()
    print(f"       Labeled records (4-class mappable): {len(df_labeled)}")

    dist = Counter(df_labeled['label4'].astype(int))
    for c in range(4):
        print(f"         {CLASS_NAMES[c]}: {dist[c]}")

    # 3. Patient-independent split
    train_df, val_df, test_df = patient_split(df_labeled, seed=args.seed)
    print(f"\n[INFO] Patient split: train={len(train_df['patient_id'].unique())} pts, "
          f"val={len(val_df['patient_id'].unique())} pts, "
          f"test={len(test_df['patient_id'].unique())} pts")

    # 4. Load waveforms + preprocess
    print("\n[INFO] Loading waveforms ...")
    X_train, y_train = build_arrays(train_df, ptbxl_dir, 'train')
    X_val,   y_val   = build_arrays(val_df,   ptbxl_dir, 'val')
    X_test,  y_test  = build_arrays(test_df,  ptbxl_dir, 'test')

    # 5. Save
    out_path = os.path.join(out_dir, 'ptbxl_dataset.npz')
    np.savez(out_path,
             X_train=X_train, y_train=y_train,
             X_val=X_val,     y_val=y_val,
             X_test=X_test,   y_test=y_test)
    print(f"\n[INFO] Saved: {out_path}")
    print(f"       X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")


if __name__ == '__main__':
    main()
