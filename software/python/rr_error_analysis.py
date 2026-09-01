"""RR-interval analysis of the misclassified samples.

Question: are the remaining errors a model defect, or a clinical boundary the
input window cannot resolve? SB and SR are morphologically identical and are
separated only by rate (the 60 bpm threshold), so if the errors concentrate
around 60 bpm the model is hitting a definition boundary, not failing to learn.

R peaks are detected on the (z-scored) test signal; HR = 60 * fs / median(RR),
RMSSD measures beat-to-beat irregularity (high in AFIB).

Usage:
  python rr_error_analysis.py --npz ../../data/ningba_processed/ningbo_dataset_clip16.npz \
      --pred results/ningba/int8_eval/ningba_int8_argmax.npy \
      --out results/ningba/rr_analysis
"""
import argparse
import json
import os

import numpy as np
from scipy.signal import find_peaks

CLASS = ['AFIB', 'GSVT', 'SB', 'SR']
FS = 250.0          # ningba preprocessing: 500 Hz -> 250 Hz, 10 s -> 2500 samples


def rr_features(sig):
    """Detect R peaks; return (HR bpm, RMSSD ms, n_beats). NaN if too few."""
    s = sig - np.median(sig)
    sd = s.std()
    if sd < 1e-6:
        return np.nan, np.nan, 0
    # Signed peaks (not |s|) so the T wave and the negative deflection are not
    # counted as beats, plus a height floor. Calibrated against the clinical
    # definition: this yields median 59.8 bpm on SB (defined <60) and 78.3 on
    # SR, whereas an |s|-based detector reports 114 bpm on SB (double-counting).
    peaks, _ = find_peaks(s, distance=int(0.30 * FS),
                          height=1.0 * sd, prominence=1.0 * sd)
    if len(peaks) < 3:
        return np.nan, np.nan, len(peaks)
    rr = np.diff(peaks) / FS                      # seconds
    hr = 60.0 / np.median(rr)
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000.0)
    return float(hr), rmssd, len(peaks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--clip', type=float, default=16.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.npz)
    X = np.clip(d['X_test'], -args.clip, args.clip).astype(np.float32)
    y = d['y_test'].astype(np.int64)
    pred = np.load(args.pred).astype(np.int64)
    assert len(pred) == len(y), f'{len(pred)} preds vs {len(y)} labels'

    hr = np.empty(len(X)); rmssd = np.empty(len(X)); nb = np.empty(len(X), int)
    for i in range(len(X)):
        hr[i], rmssd[i], nb[i] = rr_features(X[i])
    ok = np.isfinite(hr)
    correct = pred == y
    print(f"[INFO] {len(X)} samples, R-peak detection succeeded on {ok.sum()} "
          f"({ok.mean()*100:.1f}%)")

    res = {'n': int(len(X)), 'n_rr_ok': int(ok.sum()), 'fs': FS}

    # --- 1. HR distribution: correct vs wrong -------------------------------
    def stat(mask):
        v = hr[mask & ok]
        return dict(n=int(mask.sum()), n_rr=int(len(v)),
                    hr_mean=float(np.mean(v)) if len(v) else None,
                    hr_median=float(np.median(v)) if len(v) else None,
                    hr_p25=float(np.percentile(v, 25)) if len(v) else None,
                    hr_p75=float(np.percentile(v, 75)) if len(v) else None,
                    near60_pct=float(np.mean((v >= 50) & (v <= 70)) * 100) if len(v) else None)

    res['overall'] = {'correct': stat(correct), 'wrong': stat(~correct)}
    print("\n=== HR of correct vs misclassified ===")
    for k, v in res['overall'].items():
        print(f"{k:<9} n={v['n']:>5}  HR median={v['hr_median']:.1f}  "
              f"IQR[{v['hr_p25']:.1f},{v['hr_p75']:.1f}]  "
              f"in 50-70 bpm: {v['near60_pct']:.1f}%")

    # --- 2. the SB<->SR pair specifically ------------------------------------
    sb, sr = 2, 3
    pair = ((y == sb) | (y == sr))
    swap = pair & (((y == sb) & (pred == sr)) | ((y == sr) & (pred == sb)))
    res['sb_sr'] = {
        'n_pair': int(pair.sum()),
        'n_swapped': int(swap.sum()),
        'swap_rate_pct': float(swap.sum() / pair.sum() * 100),
        'hr_swapped': stat(swap),
        'hr_pair_correct': stat(pair & correct),
    }
    print(f"\n=== SB <-> SR confusion ===")
    print(f"pairs={pair.sum()}  swapped={swap.sum()} "
          f"({res['sb_sr']['swap_rate_pct']:.2f}%)")
    print(f"  swapped   HR median={res['sb_sr']['hr_swapped']['hr_median']}, "
          f"in 50-70 bpm {res['sb_sr']['hr_swapped']['near60_pct']:.1f}%")
    print(f"  correct   HR median={res['sb_sr']['hr_pair_correct']['hr_median']}, "
          f"in 50-70 bpm {res['sb_sr']['hr_pair_correct']['near60_pct']:.1f}%")

    # --- 3. error rate binned by HR ------------------------------------------
    edges = [0, 40, 50, 55, 60, 65, 70, 80, 100, 120, 300]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = ok & (hr >= lo) & (hr < hi)
        if mk.sum() == 0:
            continue
        bins.append(dict(lo=lo, hi=hi, n=int(mk.sum()),
                         err_pct=float((~correct[mk]).mean() * 100)))
    res['hr_bins'] = bins
    print("\n=== Error rate by HR band ===")
    print(f"{'band (bpm)':<14}{'n':>7}{'err %':>9}")
    for b in bins:
        print(f"{str(b['lo'])+'-'+str(b['hi']):<14}{b['n']:>7}{b['err_pct']:>9.2f}")

    # --- 4. AFIB irregularity (RMSSD) ----------------------------------------
    af = (y == 0) & ok
    res['afib_rmssd'] = dict(
        correct_median=float(np.median(rmssd[af & correct])),
        wrong_median=float(np.median(rmssd[af & ~correct])),
        n_correct=int((af & correct).sum()), n_wrong=int((af & ~correct).sum()))
    print(f"\n=== AFIB beat-to-beat irregularity (RMSSD, ms) ===")
    print(f"correctly classified AFIB: median {res['afib_rmssd']['correct_median']:.1f} "
          f"(n={res['afib_rmssd']['n_correct']})")
    print(f"missed AFIB              : median {res['afib_rmssd']['wrong_median']:.1f} "
          f"(n={res['afib_rmssd']['n_wrong']})")

    np.savez(os.path.join(args.out, 'rr_features.npz'),
             hr=hr, rmssd=rmssd, n_beats=nb, y=y, pred=pred)
    with open(os.path.join(args.out, 'rr_analysis.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\n[OK] -> {args.out}/rr_analysis.json")


if __name__ == '__main__':
    main()
