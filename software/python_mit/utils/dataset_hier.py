"""
MIT-BIH beat loader — multi-task 5-symbol (N,L,R,A,V)
=====================================================
Keep only the five most common beat symbols and train two INDEPENDENT heads
on a shared backbone:

  Dense 1 (binary)  : normal {N,L,R} vs abnormal {A,V}
  Dense 2 (5-class) : full N / L / R / A / V

(N,L,R are sinus-conducted beats → "normal" per AAMI; A=atrial premature,
V=PVC → "abnormal".) Beats not in {N,L,R,A,V} are dropped. RR features are
kept (class A is near-identical to N in morphology and needs the rhythm cue).
Reuses WFDB reading + RR-feature code from dataset.py.

Per-beat labels returned:
  y5    : 0=N 1=L 2=R 3=A 4=V    (Dense 2 target, full 5-class)
  y_bin : 0=normal {N,L,R}, 1=abnormal {A,V}   (Dense 1 target)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import wfdb
except ImportError:
    raise ImportError("wfdb is required. Install: pip install wfdb")

from dataset import _rr_features, _augment, PRE, POST, WINDOW, N_RR, LEAD

# de Chazal inter-patient split (DS1 train / DS2 test). Both sets contain all
# five symbols N,L,R,A,V in workable counts, so this honest no-patient-leak
# split is usable for the 5-symbol task (not just AAMI grouping).
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# 5 symbols → index
SYM5 = {'N': 0, 'L': 1, 'R': 2, 'A': 3, 'V': 4}
CLASS5  = ['N', 'L', 'R', 'A', 'V']
N5 = 5
# binary: normal = sinus-conducted {N,L,R} (y5 in {0,1,2}); abnormal = {A,V}
NORMAL_Y5 = {0, 1, 2}
BIN_NAMES = ['normal', 'abnormal']


def _extract_record_5sym(data_dir, rec):
    """Return (beats, rr, y5) keeping only N,L,R,A,V beats of one record."""
    path = os.path.join(data_dir, str(rec))
    sig  = wfdb.rdrecord(path).p_signal[:, LEAD].astype(np.float64)
    ann  = wfdb.rdann(path, 'atr')

    r_samples = np.asarray(ann.sample)
    median_rr = np.median(np.diff(r_samples)) if len(r_samples) > 1 else 1.0
    median_rr = median_rr if median_rr > 0 else 1.0

    beats, rr, labels = [], [], []
    n = len(sig)
    for i, (sample, symbol) in enumerate(zip(ann.sample, ann.symbol)):
        cls = SYM5.get(symbol)
        if cls is None:
            continue
        start, end = sample - PRE, sample + POST
        if start < 0 or end > n:
            continue
        w = sig[start:end]
        mu, sigma = w.mean(), w.std() + 1e-8
        beats.append(((w - mu) / sigma).astype(np.float32))
        rr.append(_rr_features(r_samples, i, median_rr))
        labels.append(cls)

    if not beats:
        return (np.empty((0, WINDOW), np.float32),
                np.empty((0, N_RR), np.float32),
                np.empty((0,), np.int64))
    return np.stack(beats), np.stack(rr), np.array(labels, dtype=np.int64)


def _load_records(data_dir, records):
    Xs, Rs, ys = [], [], []
    for rec in records:
        X, R, y = _extract_record_5sym(data_dir, rec)
        if len(X):
            Xs.append(X); Rs.append(R); ys.append(y)
    if not Xs:
        return (np.empty((0, WINDOW), np.float32),
                np.empty((0, N_RR), np.float32),
                np.empty((0,), np.int64))
    return np.concatenate(Xs), np.concatenate(Rs), np.concatenate(ys)


class MITBIH5SymDataset(Dataset):
    """Intra-patient random 80/10/10 over N,L,R,A,V beats."""

    def __init__(self, data_dir, split='train', seed=42, rr_stats=None,
                 aug_minority=None, aug_p=0.0, oversample=None, scheme='intra'):
        super().__init__()
        self.data_dir = data_dir
        self.split    = split
        self.scheme   = scheme        # 'intra' (beat 80/10/10) | 'inter' (DS1/DS2)
        self.seed     = seed
        self.rr_stats = rr_stats
        self.aug_minority = set(aug_minority) if aug_minority else set()
        self.aug_p = aug_p if split == 'train' else 0.0
        # oversample: {y5_idx: factor}, train only. Replicated copies are
        # always jittered on-the-fly (the original stays clean), so the model
        # sees a controlled number of varied views rather than exact dupes.
        self.oversample = oversample if (oversample and split == 'train') else {}
        self.epoch = 0
        self._load()

    def _load(self):
        rng = np.random.RandomState(self.seed)
        if self.scheme == 'inter':
            # No-patient-leak: train/val are disjoint DS1 records, test = DS2.
            ds1 = list(DS1)
            rng.shuffle(ds1)
            n_val = max(1, int(round(0.15 * len(ds1))))
            val_recs, tr_recs = ds1[:n_val], ds1[n_val:]
            recs = (tr_recs if self.split == 'train' else
                    val_recs if self.split == 'val' else list(DS2))
            X, R, y = _load_records(self.data_dir, recs)
        else:
            recs = sorted(int(f[:-4]) for f in os.listdir(self.data_dir)
                          if f.endswith('.dat'))
            X, R, y = _load_records(self.data_dir, recs)
            idx = rng.permutation(len(X))
            n = len(idx)
            if   self.split == 'train': sel = idx[:int(0.8 * n)]
            elif self.split == 'val':   sel = idx[int(0.8 * n):int(0.9 * n)]
            else:                       sel = idx[int(0.9 * n):]
            X, R, y = X[sel], R[sel], y[sel]

        R = np.clip(R, 0.0, 4.0)
        if self.rr_stats is None:
            self.rr_stats = (R.mean(0).astype(np.float32),
                             (R.std(0) + 1e-6).astype(np.float32))
        mu, sd = self.rr_stats
        R = ((R - mu) / sd).astype(np.float32)

        force = np.zeros(len(X), dtype=bool)          # replicated → always jitter
        if self.oversample:
            Xs, Rs, ys, fs = [X], [R], [y], [force]
            for cls, factor in self.oversample.items():
                ci = np.where(y == cls)[0]
                for _ in range(int(factor) - 1):
                    Xs.append(X[ci]); Rs.append(R[ci]); ys.append(y[ci])
                    fs.append(np.ones(len(ci), bool))
            X = np.concatenate(Xs); R = np.concatenate(Rs)
            y = np.concatenate(ys); force = np.concatenate(fs)
            order = rng.permutation(len(X))
            X, R, y, force = X[order], R[order], y[order], force[order]

        self.X, self.R, self.y5, self._force = X, R, y, force
        # binary: 0 if y5 in {N,L,R}, else 1 ({A,V})
        self.y_bin = np.array([0 if v in NORMAL_Y5 else 1 for v in y],
                              dtype=np.int64)
        self._report()

    def set_epoch(self, e):
        self.epoch = int(e)

    def _report(self):
        dist = {CLASS5[c]: int((self.y5 == c).sum()) for c in range(N5)}
        nb = int((self.y_bin == 1).sum())
        print(f"[INFO] 5sym/{self.split}: {len(self.X)} beats  dist={dist}  "
              f"(normal={len(self.X)-nb}, abnormal={nb})")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        beat = self.X[idx]
        rng = np.random.RandomState(self.seed + idx + 100003 * (self.epoch + 1))
        if self._force[idx]:
            beat = _augment(beat, rng)              # replicated copy: always vary
        elif self.aug_p and int(self.y5[idx]) in self.aug_minority:
            if rng.random() < self.aug_p:
                beat = _augment(beat, rng)
        ecg = torch.from_numpy(np.ascontiguousarray(beat))
        rr  = torch.from_numpy(np.ascontiguousarray(self.R[idx]))
        return (ecg, rr,
                torch.tensor(int(self.y_bin[idx]), dtype=torch.long),
                torch.tensor(int(self.y5[idx]),    dtype=torch.long))


def class_weights_bin(ds, temperature=0.5):
    counts = np.array([(ds.y_bin == c).sum() for c in range(2)], dtype=np.float64)
    counts = np.maximum(counts, 1)
    w = (counts.sum() / (2 * counts)) ** temperature
    return torch.tensor((w / w.mean()), dtype=torch.float32)


def class_weights_5(ds, temperature=0.5):
    counts = np.array([(ds.y5 == c).sum() for c in range(N5)], dtype=np.float64)
    counts = np.maximum(counts, 1)
    w = (counts.sum() / (N5 * counts)) ** temperature
    return torch.tensor((w / w.mean()), dtype=torch.float32)


def get_dataloaders(data_dir, batch_size=256, num_workers=2, seed=42,
                    aug_minority=None, aug_p=0.0, oversample=None,
                    scheme='intra'):
    print(f"\n{'='*60}\n  MIT-BIH 5-symbol (N,L,R,A,V) [{scheme}]: {data_dir}"
          f"\n{'='*60}")
    tr = MITBIH5SymDataset(data_dir, 'train', seed=seed,
                           aug_minority=aug_minority, aug_p=aug_p,
                           oversample=oversample, scheme=scheme)
    va = MITBIH5SymDataset(data_dir, 'val',  seed=seed, rr_stats=tr.rr_stats,
                           scheme=scheme)
    te = MITBIH5SymDataset(data_dir, 'test', seed=seed, rr_stats=tr.rr_stats,
                           scheme=scheme)
    kw = dict(num_workers=num_workers, pin_memory=True)
    return (DataLoader(tr, batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(va, batch_size, shuffle=False, **kw),
            DataLoader(te, batch_size, shuffle=False, **kw), tr)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='../../data/mitdb')
    a = p.parse_args()
    ds = MITBIH5SymDataset(a.data_dir, 'train')
    ecg, rr, yb, y5 = ds[0]
    print(f"beat {ecg.shape} rr {rr.shape}  y5={int(y5)} y_bin={int(yb)}")
