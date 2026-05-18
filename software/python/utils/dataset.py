"""
Chapman ECG Database Loader
============================
Source: https://figshare.com/collections/ChapmanECG/4560497

Data format (after unzipping ECGData.zip):
  - 10,646 CSV files: 12-lead ECG, 5000 samples (500 Hz × 10 s)
    Header: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    Values: float in µV (microvolts)
  - Diagnostics.xlsx: 10,646 rows
    Key columns: FileName, Rhythm, VentricularRate

Pipeline:
  Load CSV → Extract Lead II → Downsample 500→250 Hz → Z-score normalize
  → Output: 2500 samples per record (float32, zero-mean unit-variance)

11 Rhythm labels → 4-class mapping (Zheng et al. 2020):
  AFIB (0): AFIB, AF
  GSVT (1): ST, SVT, AT, AVNRT, AVRT, SAAWR
  SB   (2): SB
  SR   (3): SR, SA
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample

try:
    import openpyxl
except ImportError:
    raise ImportError("openpyxl is required. Install: pip install openpyxl")

# ============================================================
#  Constants
# ============================================================

RHYTHM_TO_4CLASS = {
    'AFIB': 0, 'AF': 0,
    'ST': 1, 'SVT': 1, 'AT': 1,
    'AVNRT': 1, 'AVRT': 1, 'SAAWR': 1,
    'SB': 2,
    'SR': 3, 'SA': 3,
}

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
NUM_CLASSES = 4


# ============================================================
#  Dataset Class
# ============================================================

class ChapmanECGDataset(Dataset):
    """
    Chapman ECG Dataset loader.

    Args:
        data_dir:  Directory containing Diagnostics.xlsx + CSV files
        split:     'train' (80%), 'val' (10%), 'test' (10%), or 'all'
        target_fs: Target sampling rate after resampling (default 250 Hz)
        lead:      Lead index to extract (default 1 = Lead II)
        seed:      Random seed for reproducible train/val/test split
    """

    def __init__(self, data_dir, split='train', target_fs=250, lead=1, seed=42):
        super().__init__()
        self.data_dir  = data_dir
        self.split     = split
        self.target_fs = target_fs
        self.orig_fs   = 500
        self.lead      = lead
        self.seed      = seed
        self.target_len = target_fs * 10  # 2500

        self.records     = []
        self.labels      = []
        self.heart_rates = []

        self._load()

    # ----------------------------------------------------------
    def _load(self):
        diag_path = os.path.join(self.data_dir, 'Diagnostics.xlsx')
        if not os.path.exists(diag_path):
            raise FileNotFoundError(
                f"Diagnostics.xlsx not found in {self.data_dir}\n"
                "Download Chapman ECG from: "
                "https://figshare.com/collections/ChapmanECG/4560497"
            )

        print(f"[INFO] Reading {diag_path} ...")
        wb = openpyxl.load_workbook(diag_path, read_only=True)
        ws = wb.active
        rows = ws.iter_rows(values_only=True)
        next(rows)  # skip header

        all_entries     = []
        skipped_rhythms = {}

        for row in rows:
            fname  = row[0]
            rhythm = row[1]
            hr     = row[5]

            if rhythm in RHYTHM_TO_4CLASS:
                all_entries.append((fname, RHYTHM_TO_4CLASS[rhythm], hr))
            else:
                skipped_rhythms[rhythm] = skipped_rhythms.get(rhythm, 0) + 1

        wb.close()

        if skipped_rhythms:
            print(f"[INFO] Skipped unknown rhythms: {skipped_rhythms}")
        print(f"[INFO] Total labeled entries: {len(all_entries)}")

        # Split 80/10/10
        # Use legacy-compatible numpy random API (np.random.seed + permutation)
        # so that train/val/test indices match the legacy codebase exactly.
        np.random.seed(self.seed)
        indices = np.random.permutation(len(all_entries))
        n       = len(indices)

        if   self.split == 'train': indices = indices[:int(0.8 * n)]
        elif self.split == 'val':   indices = indices[int(0.8 * n):int(0.9 * n)]
        elif self.split == 'test':  indices = indices[int(0.9 * n):]
        # 'all' → keep all

        loaded    = 0
        missing   = 0
        too_short = 0

        for count, i in enumerate(indices):
            fname, label, hr = all_entries[i]
            csv_path = os.path.join(self.data_dir, f"{fname}.csv")

            if not os.path.exists(csv_path):
                missing += 1
                continue

            try:
                ecg_12 = np.loadtxt(csv_path, delimiter=',', skiprows=1,
                                    encoding='utf-8-sig')
                lead_sig = ecg_12[:, self.lead].astype(np.float64)

                # Skip records shorter than 3.85 s
                if len(lead_sig) < int(3.85 * self.orig_fs):
                    too_short += 1
                    continue

                # Downsample 500 → 250 Hz
                lead_sig = resample(lead_sig, self.target_len)

                # Z-score normalization
                mu    = np.mean(lead_sig)
                sigma = np.std(lead_sig) + 1e-8
                lead_sig = (lead_sig - mu) / sigma

                self.records.append(lead_sig.astype(np.float32))
                self.labels.append(label)
                self.heart_rates.append(hr if hr is not None else 0)
                loaded += 1

            except Exception:
                missing += 1

            if (count + 1) % 1000 == 0:
                print(f"[INFO]   Processed {count+1}/{len(indices)} "
                      f"(loaded={loaded}) ...")

        dist = {CLASS_NAMES[c]: self.labels.count(c) for c in range(NUM_CLASSES)}
        print(f"[INFO] {self.split}: {loaded} records "
              f"(missing={missing}, too_short={too_short})")
        print(f"[INFO]   Distribution: {dist}")

    # ----------------------------------------------------------
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        ecg   = torch.from_numpy(self.records[idx])   # (2500,) float32
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        hr    = self.heart_rates[idx]
        return ecg, label, hr


# ============================================================
#  DataLoader factory
# ============================================================

def get_dataloaders(data_dir, batch_size=128, num_workers=2, seed=42):
    """
    Create train / val / test DataLoaders for Chapman ECG.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    print(f"\n{'='*60}")
    print(f"  Chapman ECG Database: {data_dir}")
    print(f"{'='*60}")

    train_ds = ChapmanECGDataset(data_dir, split='train', seed=seed)
    val_ds   = ChapmanECGDataset(data_dir, split='val',   seed=seed)
    test_ds  = ChapmanECGDataset(data_dir, split='test',  seed=seed)

    kw = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)

    print(f"\n  Train: {len(train_ds):5d} records | {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds):5d} records | {len(val_loader)} batches")
    print(f"  Test:  {len(test_ds):5d} records | {len(test_loader)} batches")
    print(f"{'='*60}\n")

    return train_loader, val_loader, test_loader


# ============================================================
#  Standalone test
# ============================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str,
                   default='/home/duc/Thesis/data/Chapman')
    args = p.parse_args()

    ds = ChapmanECGDataset(args.data_dir, split='train')
    print(f"\nDataset size: {len(ds)}")
    if len(ds) > 0:
        ecg, label, hr = ds[0]
        print(f"ECG shape={ecg.shape}, dtype={ecg.dtype}")
        print(f"Label={label} ({CLASS_NAMES[label]}), HR={hr}")
        print(f"ECG range: [{ecg.min():.3f}, {ecg.max():.3f}]")
