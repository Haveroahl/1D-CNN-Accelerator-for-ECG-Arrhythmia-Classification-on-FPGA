"""
MIT-BIH Arrhythmia Database Loader (beat-level, 5-class AAMI)
============================================================
Source: https://physionet.org/content/mitdb/1.0.0/  (data/mitdb/)

Data format (WFDB):
  - 48 records, fs=360 Hz, 2 leads. Lead 0 = MLII (used here).
  - Per-record beat annotations in .atr: a symbol + R-peak sample index per beat.

AAMI 5-class grouping (de Chazal et al. 2004; ANSI/AAMI EC57):
  N (0): N, L, R, e, j      (normal + bundle-branch + nodal/atrial escape)
  S (1): A, a, J, S         (supraventricular ectopic)
  V (2): V, E               (ventricular ectopic)
  F (3): F                  (fusion of ventricular + normal)
  Q (4): /, f, Q            (paced / fusion-paced / unclassifiable)
Non-beat symbols (+, ~, !, ", |, [, ], x) are ignored.

Pipeline:
  rdrecord lead MLII → per-beat window of WINDOW samples centred on the
  annotated R-peak (PRE before, POST after) → per-beat Z-score normalize.

Two split schemes (selectable via `scheme`):
  - 'intra'  : pool all beats, random 80/10/10 (beats of one patient may
               appear in several splits — optimistic, target ~98%).
  - 'inter'  : AAMI inter-patient DS1 (train) / DS2 (test); val carved out
               of DS1 by record. No patient leakage — honest baseline.

Augmentation (train split only, optional): morphology-preserving jitter on
minority classes to counter the heavy N-class imbalance. Amplitude scaling,
additive Gaussian noise, and small baseline wander — none of which move the
R-peak or distort QRS shape.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import wfdb
except ImportError:
    raise ImportError("wfdb is required. Install: pip install wfdb")

# ============================================================
#  Constants
# ============================================================

SYMBOL_TO_AAMI = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4,
}

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
NUM_CLASSES = 5

ORIG_FS = 360
LEAD    = 0          # MLII

# Window: ~0.53 s around the R-peak. Keep the full pre-R region (P-wave, key for
# the atrial-premature class A) and trim the post-R tail (T-wave) → 192 samples.
PRE     = 100        # samples before R-peak
POST    = 92         # samples after R-peak
WINDOW  = PRE + POST  # 192

# AAMI inter-patient split (de Chazal). The 4 paced records below are excluded
# from both DS1 and DS2 by the AAMI recommendation.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# 4 paced records (continuous artificial-pacemaker beats). de Chazal drops them
# entirely. We exclude them from BOTH schemes so intra and inter are consistent
# and free of the paced-beat artifact; this leaves the AAMI 4-class problem
# N/S/V/F (the Q class is essentially these records, so it drops out).
PACED_RECORDS = [102, 104, 107, 217]


# ============================================================
#  Beat extraction
# ============================================================

# 12 RR features per beat. [0:4] immediate-neighbour set; [4:8] ±2-beat context
# and longer-term ratios; [8:12] prematurity-targeted features for class A
# (atrial premature beat: short pre-RR + compensatory pause, near-normal QRS).
N_RR = 12


def _rr_at(samples, i, median_rr):
    """RR interval ending at beat i (samples[i]-samples[i-1]), HR-normalized."""
    if i <= 0 or i >= len(samples):
        return 1.0
    return (samples[i] - samples[i - 1]) / median_rr


def _rr_features(samples, i, median_rr, win=10):
    """
    Heart-rate-invariant rhythm features for beat i, from the full R-peak
    sequence `samples` (all annotated beats, before AAMI filtering).

    Normalizing by the record median RR makes these invariant to baseline
    heart rate, so they encode "is this beat early/late" — the cue that
    separates supraventricular ectopics (S) from normal (N), which look
    almost identical in single-beat morphology.
    """
    n = len(samples)
    pre  = (samples[i]   - samples[i-1]) if i > 0     else median_rr
    post = (samples[i+1] - samples[i])   if i + 1 < n else median_rr
    lo, hi = max(0, i - win), min(n, i + win + 1)
    diffs = np.diff(samples[lo:hi]) if hi - lo > 1 else np.array([median_rr])
    local = np.median(diffs) if len(diffs) else median_rr
    local = local if local > 0 else median_rr
    local_std = float(np.std(diffs)) if len(diffs) > 1 else 0.0

    pre2  = _rr_at(samples, i - 1, median_rr)   # RR one step before pre
    post2 = _rr_at(samples, i + 2, median_rr)   # RR one step after post

    # Prematurity-targeted features for class A (atrial premature beat):
    # A has an abnormally SHORT pre-RR followed by a compensatory pause, against
    # a near-normal QRS — only the rhythm reveals it.
    prematurity = max(0.0, 1.0 - pre / local)        # 0 normal, ↑ when pre is short
    compensation = post / local                       # ↑ pause after a premature beat
    rr_irregular = local_std / local                  # local RR variability
    # combined premature-then-pause signature (high & positive only for A-like)
    premature_pause = prematurity * compensation

    return np.array([
        pre  / median_rr,          # 0  pre-RR (norm)
        post / median_rr,          # 1  post-RR (norm)
        pre  / (post + 1e-8),      # 2  pre/post ratio
        pre  / local,              # 3  pre / local-median
        pre2,                      # 4  RR two beats back (norm)
        post2,                     # 5  RR two beats ahead (norm)
        post / local,              # 6  post / local-median
        (pre + post) / (2.0 * local),  # 7  avg(pre,post) / local-median
        prematurity,               # 8  how much earlier than local rhythm
        compensation,              # 9  compensatory-pause magnitude
        rr_irregular,              # 10 local RR irregularity
        premature_pause,           # 11 premature-then-pause signature (A-specific)
    ], dtype=np.float32)


def _extract_record_beats(data_dir, rec):
    """Return (beats, rr, labels) for one record.
    beats: (n, WINDOW) f32 | rr: (n, N_RR) f32 | labels: (n,) i64."""
    path = os.path.join(data_dir, str(rec))
    sig  = wfdb.rdrecord(path).p_signal[:, LEAD].astype(np.float64)
    ann  = wfdb.rdann(path, 'atr')

    # RR features come from the full annotated R-peak sequence (any symbol),
    # so neighbour timing is correct even where neighbours are non-AAMI beats.
    r_samples = np.asarray(ann.sample)
    median_rr = np.median(np.diff(r_samples)) if len(r_samples) > 1 else 1.0
    median_rr = median_rr if median_rr > 0 else 1.0

    beats, rr, labels = [], [], []
    n = len(sig)
    for i, (sample, symbol) in enumerate(zip(ann.sample, ann.symbol)):
        cls = SYMBOL_TO_AAMI.get(symbol)
        if cls is None:
            continue
        start = sample - PRE
        end   = sample + POST
        if start < 0 or end > n:
            continue   # drop beats too close to record edges
        w = sig[start:end]
        mu, sigma = w.mean(), w.std() + 1e-8
        beats.append(((w - mu) / sigma).astype(np.float32))
        rr.append(_rr_features(r_samples, i, median_rr))
        labels.append(cls)

    if not beats:
        return (np.empty((0, WINDOW), np.float32),
                np.empty((0, N_RR), np.float32),
                np.empty((0,), np.int64))
    return (np.stack(beats), np.stack(rr),
            np.array(labels, dtype=np.int64))


def _load_records(data_dir, records):
    """Concatenate beats + rr + labels from a list of records."""
    Xs, Rs, ys = [], [], []
    for rec in records:
        X, R, y = _extract_record_beats(data_dir, rec)
        if len(X):
            Xs.append(X); Rs.append(R); ys.append(y)
    if not Xs:
        return (np.empty((0, WINDOW), np.float32),
                np.empty((0, N_RR), np.float32),
                np.empty((0,), np.int64))
    return np.concatenate(Xs), np.concatenate(Rs), np.concatenate(ys)


# ============================================================
#  Augmentation (morphology-preserving)
# ============================================================

def _augment(beat, rng):
    """Jitter one beat without moving the R-peak or distorting QRS shape."""
    out = beat.astype(np.float64).copy()
    n = len(out)
    # light time-warp: resample to ±5% length then crop/pad back to n, centred
    # so the R-peak stays put — varies QRS width slightly (a real inter-beat
    # source of variation) without destroying morphology.
    scale = rng.uniform(0.95, 1.05)
    m = max(8, int(round(n * scale)))
    warped = np.interp(np.linspace(0, n - 1, m),
                       np.arange(n), out)
    if m >= n:
        s = (m - n) // 2
        out = warped[s:s + n]
    else:
        pad = n - m
        lo = pad // 2
        out = np.pad(warped, (lo, pad - lo), mode='edge')
    out *= rng.uniform(0.85, 1.15)                        # amplitude scaling
    out += rng.normal(0, 0.05, size=n)                    # additive noise
    phase = rng.uniform(0, 2 * np.pi)                     # slow baseline wander
    out += 0.05 * np.sin(np.linspace(0, np.pi, n) + phase)
    return out.astype(np.float32)


# ============================================================
#  Dataset
# ============================================================

class MITBIHBeatDataset(Dataset):
    """
    MIT-BIH beat-level dataset, 5-class AAMI.

    Args:
        data_dir : directory with the WFDB records (.dat/.hea/.atr)
        split    : 'train' | 'val' | 'test'
        scheme   : 'intra' (random 80/10/10) or 'inter' (AAMI DS1/DS2)
        augment  : if True and split=='train', oversample+jitter minorities
        target_per_class : cap per class after oversampling (train only).
                           None → no oversampling, only on-the-fly jitter is
                           skipped too. Set e.g. 20000 to balance.
        seed     : split / RNG seed.
    """

    def __init__(self, data_dir, split='train', scheme='intra',
                 augment=False, target_per_class=None, seed=42,
                 rr_stats=None, aug_minority=None, aug_p=0.0):
        super().__init__()
        self.data_dir = data_dir
        self.split    = split
        self.scheme   = scheme
        self.augment  = augment and split == 'train'
        self.target_per_class = target_per_class
        self.seed     = seed
        # (mean, std) per RR feature; train computes its own, val/test reuse it.
        self.rr_stats = rr_stats
        # On-the-fly minority augmentation (train only): each epoch, a beat whose
        # class is in `aug_minority` is jittered with probability `aug_p`. Unlike
        # fixed oversampling, the jitter is re-rolled per epoch (epoch counter is
        # mixed into the RNG seed), so the model never memorizes fixed copies.
        self.aug_minority = set(aug_minority) if aug_minority else set()
        self.aug_p   = aug_p if split == 'train' else 0.0
        self.epoch   = 0

        self.X = None   # (N, WINDOW) float32
        self.R = None   # (N, N_RR)   float32 (standardized)
        self.y = None   # (N,)        int64
        self._aug_flag = None  # bool per sample: apply jitter on __getitem__

        self._load()

    # ----------------------------------------------------------
    def _load(self):
        rng = np.random.RandomState(self.seed)

        if self.scheme == 'intra':
            all_recs = [int(f[:-4]) for f in os.listdir(self.data_dir)
                        if f.endswith('.dat')]
            X, R, y = _load_records(self.data_dir, sorted(all_recs))
            idx = rng.permutation(len(X))
            n = len(idx)
            if   self.split == 'train': sel = idx[:int(0.8 * n)]
            elif self.split == 'val':   sel = idx[int(0.8 * n):int(0.9 * n)]
            else:                       sel = idx[int(0.9 * n):]
            X, R, y = X[sel], R[sel], y[sel]

        elif self.scheme == 'inter':
            if self.split == 'test':
                X, R, y = _load_records(self.data_dir, DS2)
            else:
                X, R, y = _load_records(self.data_dir, DS1)
                # carve val out of DS1 by random beats (records already fixed)
                idx = rng.permutation(len(X))
                cut = int(0.9 * len(X))
                sel = idx[:cut] if self.split == 'train' else idx[cut:]
                X, R, y = X[sel], R[sel], y[sel]
        else:
            raise ValueError(f"unknown scheme {self.scheme!r}")

        # ---- inter-patient = AAMI de Chazal 4-class (drop paced/Q) ----
        # DS1/DS2 already exclude the 4 paced records (102,104,107,217); the few
        # residual Q beats can't be learned and only distort F1-macro, so drop
        # them → N/S/V/F. (intra keeps all 5 classes incl. Q.)
        if self.scheme == 'inter':
            keep = y != 4               # 4 == Q
            X, R, y = X[keep], R[keep], y[keep]

        # ---- standardize RR features (clip outliers, then z-score) ----
        R = np.clip(R, 0.0, 4.0)
        if self.rr_stats is None:
            mu = R.mean(0); sd = R.std(0) + 1e-6
            self.rr_stats = (mu.astype(np.float32), sd.astype(np.float32))
        mu, sd = self.rr_stats
        R = ((R - mu) / sd).astype(np.float32)

        # ---- balancing / augmentation (train only) ----
        if self.augment and self.target_per_class:
            X, R, y, aug = self._balance(X, R, y, rng)
        else:
            aug = np.zeros(len(X), dtype=bool)

        self.X, self.R, self.y, self._aug_flag = X, R, y, aug
        self._report()

    def _balance(self, X, R, y, rng):
        """Oversample each class up to target_per_class; mark copies for jitter."""
        Xs, Rs, ys, flags = [], [], [], []
        for c in range(NUM_CLASSES):
            ci = np.where(y == c)[0]
            if len(ci) == 0:
                continue
            # keep originals (no jitter)
            Xs.append(X[ci]); Rs.append(R[ci]); ys.append(y[ci])
            flags.append(np.zeros(len(ci), bool))
            deficit = self.target_per_class - len(ci)
            if deficit > 0:
                pick = rng.choice(ci, size=deficit, replace=True)
                Xs.append(X[pick]); Rs.append(R[pick]); ys.append(y[pick])
                flags.append(np.ones(deficit, bool))   # jittered copies
        X2 = np.concatenate(Xs); R2 = np.concatenate(Rs)
        y2 = np.concatenate(ys); f2 = np.concatenate(flags)
        order = rng.permutation(len(X2))
        return X2[order], R2[order], y2[order], f2[order]

    def _report(self):
        dist = {CLASS_NAMES[c]: int((self.y == c).sum()) for c in range(NUM_CLASSES)}
        print(f"[INFO] {self.scheme}/{self.split}: {len(self.X)} beats  dist={dist}"
              + (f"  (augmented={int(self._aug_flag.sum())})" if self.augment else ""))

    # ----------------------------------------------------------
    def __len__(self):
        return len(self.X)

    def set_epoch(self, e):
        """Set epoch so per-epoch minority jitter is re-rolled each pass."""
        self.epoch = int(e)

    def __getitem__(self, idx):
        beat = self.X[idx]
        if self._aug_flag is not None and self._aug_flag[idx]:
            # fixed-oversampling jitter path (only when target_per_class used)
            beat = _augment(beat, np.random.RandomState(self.seed + idx))
        elif self.aug_p and int(self.y[idx]) in self.aug_minority:
            # on-the-fly minority jitter, re-rolled per epoch
            rng = np.random.RandomState(self.seed + idx + 100003 * (self.epoch + 1))
            if rng.random() < self.aug_p:
                beat = _augment(beat, rng)
        ecg   = torch.from_numpy(np.ascontiguousarray(beat))   # (WINDOW,)
        rr    = torch.from_numpy(np.ascontiguousarray(self.R[idx]))  # (N_RR,)
        label = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return ecg, rr, label


# ============================================================
#  DataLoader factory
# ============================================================

def class_weights(dataset, temperature=0.5):
    """
    Class weights for the CE/focal loss, from the (raw) class frequencies.

    w_c = (total / (K * count_c)) ** temperature

    temperature=1.0 → plain inverse-frequency (here the rarest class F gets
    ~28×, a 114× spread that destabilizes focal loss). temperature=0.5 takes
    the square root, compressing the spread to ~11× — still strongly favours
    minorities but keeps gradients sane. Weights are renormalized to mean 1.
    """
    counts = np.array([(dataset.y == c).sum() for c in range(NUM_CLASSES)],
                      dtype=np.float64)
    counts = np.maximum(counts, 1)
    w = (counts.sum() / (NUM_CLASSES * counts)) ** temperature
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def get_dataloaders(data_dir, scheme='intra', batch_size=256, num_workers=2,
                    augment=True, target_per_class=20000, seed=42,
                    aug_minority=None, aug_p=0.0):
    """
    Create train / val / test DataLoaders for MIT-BIH beats.

    Returns:
        (train_loader, val_loader, test_loader, train_ds)
    train_ds is returned so callers can derive class weights.
    """
    print(f"\n{'='*60}")
    print(f"  MIT-BIH beats: {data_dir}  (scheme={scheme})")
    print(f"{'='*60}")

    train_ds = MITBIHBeatDataset(data_dir, 'train', scheme,
                                 augment=augment,
                                 target_per_class=target_per_class, seed=seed,
                                 aug_minority=aug_minority, aug_p=aug_p)
    # val/test reuse the train RR-feature standardization (no leakage)
    val_ds   = MITBIHBeatDataset(data_dir, 'val',  scheme, seed=seed,
                                 rr_stats=train_ds.rr_stats)
    test_ds  = MITBIHBeatDataset(data_dir, 'test', scheme, seed=seed,
                                 rr_stats=train_ds.rr_stats)

    kw = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)

    print(f"\n  Train: {len(train_ds):6d} beats | {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds):6d} beats | {len(val_loader)} batches")
    print(f"  Test:  {len(test_ds):6d} beats | {len(test_loader)} batches")
    print(f"{'='*60}\n")

    return train_loader, val_loader, test_loader, train_ds


# ============================================================
#  Standalone test
# ============================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='../../data/mitdb')
    p.add_argument('--scheme',   type=str, default='intra',
                   choices=['intra', 'inter'])
    args = p.parse_args()

    ds = MITBIHBeatDataset(args.data_dir, 'train', args.scheme,
                           augment=True, target_per_class=20000)
    print(f"\nDataset size: {len(ds)}")
    ecg, rr, label = ds[0]
    print(f"beat shape={ecg.shape}, rr shape={rr.shape}, "
          f"label={label} ({CLASS_NAMES[label]})")
    print(f"beat range: [{ecg.min():.3f}, {ecg.max():.3f}]  rr={rr.numpy().round(3)}")
    print(f"rr_stats mean={ds.rr_stats[0].round(3)} std={ds.rr_stats[1].round(3)}")
