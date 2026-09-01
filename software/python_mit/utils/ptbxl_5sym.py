"""
PTB-XL beat extractor — cross-check the MIT-BIH 5-symbol model {N,L,R,A,V}
=========================================================================
Goal: take a tiny model trained on MIT-BIH per-beat symbols and ask "does it
generalize zero-shot to PTB-XL beats of the same five kinds?".

PTB-XL is NOT a per-beat database. Its labels are SCP statements per 10-s
record. We map them onto the five MIT-BIH symbols as follows:

  MIT-BIH   PTB-XL SCP   label kind          per-beat ground-truth?
  -------   ----------   ----------------    ----------------------
  N (0)     NORM         diagnostic (whole)  YES  (every beat is normal)
  L (1)     CLBBB        diagnostic (whole)  YES  (every beat is LBBB)
  R (2)     CRBBB        diagnostic (whole)  YES  (every beat is RBBB)
  A (3)     PAC          form  (a beat is)   WEAK (only 1-2 beats are PAC; we
  V (4)     PVC          form  (a beat is)   WEAK  label ALL beats in the strip
                                                   with the record symbol)

So N/L/R beats carry honest per-beat labels; A/V are weak record-level labels
(most beats in a PAC/PVC strip are really N — accept the noise, the user asked
for the 5-class picture). Records that carry more than one of these five codes
are dropped (ambiguous).

Signal handling, matched to MIT-BIH so the trained morphology transfers:
  - lead II (PTB-XL index 1; MIT-BIH used MLII)
  - resample 500 Hz -> 360 Hz (MIT-BIH fs)
  - R-peak detection by a compact Pan-Tompkins (neurokit2 not installed)
  - per-beat window WINDOW (=192) centred on the R-peak, Z-score
  - the same 12 RR features as utils/dataset.py (HR-normalized by median RR)

Returns arrays compatible with ECG_TinyMultiTask: (beats, rr, y5).
"""

import os
import ast
import numpy as np
from scipy.signal import resample_poly, butter, filtfilt

try:
    import wfdb
except ImportError:
    raise ImportError("wfdb is required. Install: pip install wfdb")

# reuse the EXACT MIT-BIH beat geometry + RR features so the model sees the
# same input distribution it was trained on.
from dataset import PRE, POST, WINDOW, N_RR, _rr_features
from dataset_hier import SYM5, CLASS5            # {N,L,R,A,V} index space

PTBXL_FS = 500
TARGET_FS = 360                                   # MIT-BIH sampling rate
LEAD_II   = 1                                     # PTB-XL lead order: I, II, ...

# PTB-XL SCP code -> MIT-BIH 5-symbol index. CLBBB/CRBBB/NORM are whole-record
# diagnostics (clean per-beat label); PAC/PVC are "form" codes (weak label).
SCP_TO_SYM5 = {
    'NORM':  SYM5['N'],
    'CLBBB': SYM5['L'],
    'CRBBB': SYM5['R'],
    'PAC':   SYM5['A'],
    'PVC':   SYM5['V'],
}
WEAK_SYMS = {SYM5['A'], SYM5['V']}                # record-level (noisy) labels


# ------------------------------------------------------------------
#  Pan-Tompkins-lite R-peak detection (lead II, TARGET_FS)
# ------------------------------------------------------------------

def _bandpass(sig, fs, lo=5.0, hi=15.0):
    b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, sig)


def _detect_rpeaks(sig, fs=TARGET_FS):
    """Compact Pan-Tompkins: bandpass -> derivative -> square -> moving-window
    integrate -> adaptive-threshold peak pick. Returns R-peak sample indices."""
    x = _bandpass(sig.astype(np.float64), fs)
    d = np.ediff1d(x, to_begin=0.0)               # derivative
    sq = d * d                                    # square
    w = max(1, int(0.150 * fs))                   # ~150 ms integration window
    integ = np.convolve(sq, np.ones(w) / w, mode='same')

    thr = 0.3 * np.mean(integ) + 0.0001
    min_gap = int(0.25 * fs)                       # 250 ms refractory (=240 bpm cap)
    peaks = []
    i, n = 1, len(integ) - 1
    while i < n:
        if integ[i] > thr and integ[i] >= integ[i - 1] and integ[i] >= integ[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_gap:
                peaks.append(i)
            elif integ[i] > integ[peaks[-1]]:
                peaks[-1] = i
        i += 1
    if not peaks:
        return np.array([], dtype=int)
    # refine each peak to the local maximum of the raw signal (±50 ms)
    r = int(0.05 * fs)
    refined = []
    for p in peaks:
        lo, hi = max(0, p - r), min(len(sig), p + r + 1)
        refined.append(lo + int(np.argmax(sig[lo:hi])))
    return np.unique(refined)


# ------------------------------------------------------------------
#  Per-record beat extraction
# ------------------------------------------------------------------

def _record_codes(scp_codes):
    """Which of our 5 target symbols this record carries (as a set)."""
    return {SCP_TO_SYM5[c] for c in scp_codes if c in SCP_TO_SYM5}


def _extract_record(path, sym):
    """Extract all beats of one PTB-XL record, labelling every beat `sym`.

    path : full path to the .hea/.dat (no extension)
    sym  : the single MIT-BIH 5-symbol index assigned to this record
    """
    rec = wfdb.rdrecord(path)
    sig = rec.p_signal[:, LEAD_II].astype(np.float64)
    # 500 -> 360 Hz (25/36 ratio is exact: 360/500 = 18/25)
    sig = resample_poly(sig, up=18, down=25)
    fs = TARGET_FS

    r_samples = _detect_rpeaks(sig, fs)
    if len(r_samples) < 3:
        return None
    median_rr = np.median(np.diff(r_samples))
    median_rr = median_rr if median_rr > 0 else 1.0

    beats, rr, labels = [], [], []
    n = len(sig)
    for i, sample in enumerate(r_samples):
        start, end = sample - PRE, sample + POST
        if start < 0 or end > n:
            continue
        w = sig[start:end]
        mu, sigma = w.mean(), w.std() + 1e-8
        beats.append(((w - mu) / sigma).astype(np.float32))
        rr.append(_rr_features(r_samples, i, median_rr))
        labels.append(sym)
    if not beats:
        return None
    return (np.stack(beats), np.stack(rr),
            np.array(labels, dtype=np.int64))


def load_ptbxl_5sym(ptbxl_dir, rr_stats, max_per_class=None, sampling='hr',
                    seed=42, verbose=True):
    """Build a PTB-XL beat set labelled in the MIT-BIH 5-symbol space.

    Args:
        ptbxl_dir   : data/ptbxl directory (has ptbxl_database.csv, records500/)
        rr_stats    : (mu, sd) per RR feature from the MIT-BIH TRAIN set — RR
                      features must be standardized with the SAME stats the
                      model trained on, or the rhythm branch sees garbage.
        max_per_class: cap records per symbol (None = all). Keeps runtime sane.
        sampling    : 'hr' (500 Hz, records500) — required, we resample to 360.
        seed        : record-shuffle seed for the per-class cap.

    Returns (beats, rr, y5): arrays ready for the tiny model.
    """
    import pandas as pd
    db = pd.read_csv(os.path.join(ptbxl_dir, 'ptbxl_database.csv'),
                     index_col='ecg_id')
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)

    # assign each record at most one symbol: keep only records carrying exactly
    # one of the five target codes (multi-label records are ambiguous per-beat).
    rng = np.random.RandomState(seed)
    by_sym = {s: [] for s in range(len(CLASS5))}
    for ecg_id, row in db.iterrows():
        syms = _record_codes(row.scp_codes)
        if len(syms) != 1:
            continue
        sym = next(iter(syms))
        fname = row.filename_hr if sampling == 'hr' else row.filename_lr
        by_sym[sym].append(os.path.join(ptbxl_dir, fname))

    Xs, Rs, ys = [], [], []
    for sym in range(len(CLASS5)):
        paths = by_sym[sym]
        rng.shuffle(paths)
        if max_per_class:
            paths = paths[:max_per_class]
        nb = 0
        for p in paths:
            out = _extract_record(p, sym)
            if out is None:
                continue
            X, R, y = out
            Xs.append(X); Rs.append(R); ys.append(y); nb += len(X)
        if verbose:
            print(f"[PTB-XL] {CLASS5[sym]:1s}: {len(paths):4d} records -> "
                  f"{nb:6d} beats")

    X = np.concatenate(Xs); R = np.concatenate(Rs); y = np.concatenate(ys)
    # standardize RR with the MIT-BIH train stats (NOT PTB-XL's own).
    R = np.clip(R, 0.0, 4.0)
    mu, sd = rr_stats
    R = ((R - mu) / sd).astype(np.float32)
    if verbose:
        dist = {CLASS5[c]: int((y == c).sum()) for c in range(len(CLASS5))}
        print(f"[PTB-XL] total {len(X)} beats  dist={dist}")
    return X, R, y
