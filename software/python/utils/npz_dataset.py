"""
NPZ-backed dataset loader — drop-in replacement for utils.dataset.get_dataloaders.

Loads a pre-split .npz (keys X_train/y_train/X_val/y_val/X_test/y_test), used to
train/prune/quantize on the extended Chapman-Ningbo set (ningbo_dataset.npz)
without touching the Chapman-xlsx loader. Returns the SAME interface as
get_dataloaders: (train_loader, val_loader, test_loader), and each item is the
same 3-tuple (ecg[2500] float32, label long, hr) so train/prune code that
unpacks batch[0], batch[1] works unchanged. hr is a dummy 0 (npz has no HR).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NPZECGDataset(Dataset):
    def __init__(self, npz_path, split='train'):
        d = np.load(npz_path)
        self.X = d[f'X_{split}']          # (N, 2500) float32
        self.y = d[f'y_{split}']          # (N,) int64
        assert self.X.ndim == 2 and self.X.shape[1] == 2500, \
            f"expected (N,2500), got {self.X.shape}"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        ecg   = torch.from_numpy(self.X[idx].astype(np.float32))   # (2500,)
        label = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return ecg, label, 0   # hr dummy, matches Chapman loader 3-tuple


def get_npz_dataloaders(npz_path, batch_size=128, num_workers=2, seed=42):
    """
    Create train / val / test DataLoaders from a pre-split .npz.
    Mirrors utils.dataset.get_dataloaders signature and return.
    """
    print(f"\n{'='*60}")
    print(f"  NPZ ECG dataset: {npz_path}")
    print(f"{'='*60}")

    train_ds = NPZECGDataset(npz_path, split='train')
    val_ds   = NPZECGDataset(npz_path, split='val')
    test_ds  = NPZECGDataset(npz_path, split='test')

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
