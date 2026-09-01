"""
RR-fusion probe — simulate HPS hybrid post-processing on PTB-XL zero-shot.
===========================================================================
Goal: fix the SB/SR cross-dataset confusion WITHOUT touching the FPGA core.

Pipeline (mirrors what the Cortex-A9 HPS would do):
  1. FPGA accelerator  -> logits[4] / argmax        (unchanged, bit-exact)
  2. HPS R-peak detect -> HR_bpm  (count peaks / 10s window * 6)
  3. Decision fusion: if argmax in {SB,SR}: class = SB if HR<thr else SR
                      else: keep argmax  (AFIB/GSVT untouched)

Reports baseline (pure argmax) vs fused, overall + per-class, and a small
threshold sweep so we can see how sensitive the gain is to the bpm cutoff.

Read-only: loads the QAT checkpoint + the preprocessed PTB-XL npz. No training.
"""
import os, sys, argparse
import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptbxl_eval import ECG_CNN, load_qat_checkpoint, CLASS_NAMES   # reuse exact model

SB, SR = 2, 3
FS = 250            # Hz after preprocessing
WIN_S = 10.0        # 2500 samples = 10 s


def detect_hr_bpm(sig, fs=FS):
    """R-peak count -> bpm. sig is z-scored (per-record), so threshold on std.
    Mirrors a lightweight HPS detector: rectify, find prominent peaks, count."""
    # min RR 0.33 s (=180 bpm ceiling) -> min distance between peaks
    peaks, _ = find_peaks(np.abs(sig), distance=int(0.33 * fs),
                          prominence=np.std(sig) * 1.2)
    if len(peaks) < 2:
        return np.nan
    # robust rate: median instantaneous, converted to bpm
    rr = np.diff(peaks) / fs            # seconds between beats
    inst_bpm = 60.0 / rr
    return float(np.median(inst_bpm))


def fuse(argmax, hr_bpm, thr):
    """Override only SB/SR pairs using HR; leave AFIB/GSVT as-is."""
    out = argmax.copy()
    sbsr = np.isin(argmax, [SB, SR])
    have_hr = ~np.isnan(hr_bpm)
    m = sbsr & have_hr
    out[m] = np.where(hr_bpm[m] < thr, SB, SR)
    return out


def metrics(y, pred):
    return (accuracy_score(y, pred),
            f1_score(y, pred, average='macro', zero_division=0),
            f1_score(y, pred, average=None, labels=[0,1,2,3], zero_division=0),
            confusion_matrix(y, pred, labels=[0,1,2,3]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',  default=os.path.join(os.path.dirname(__file__), '..',
                   'results', 'qat_int8', 'model_qat_int8.pth'))
    p.add_argument('--ptbxl', default=os.path.join(os.path.dirname(__file__), '..', '..', '..',
                   'data', 'ptbxl_processed', 'ptbxl_dataset.npz'))
    args = p.parse_args()
    device = 'cpu'

    print(f"[INFO] loading {os.path.basename(args.ptbxl)} (val + test splits)")
    d = np.load(args.ptbxl)
    Xv, yv = d['X_val'].astype(np.float32),  d['y_val'].astype(int)
    X,  y  = d['X_test'].astype(np.float32), d['y_test'].astype(int)
    print(f"[INFO] val  n={len(yv)}  dist={np.bincount(yv, minlength=4).tolist()}")
    print(f"[INFO] test n={len(y)}   dist={np.bincount(y,  minlength=4).tolist()} (AFIB,GSVT,SB,SR)")

    # ---- logits / argmax from the exact cross-eval model (== C2) ----
    model = load_qat_checkpoint(args.ckpt, device).eval()
    with torch.no_grad():
        argmax_v = model(torch.from_numpy(Xv)).numpy().argmax(1)
        argmax   = model(torch.from_numpy(X)).numpy().argmax(1)

    acc0, f10, f1p0, cm0 = metrics(y, argmax)
    print(f"\n=== BASELINE on TEST (pure argmax, == C2) ===")
    print(f"acc={acc0:.4f}  f1-macro={f10:.4f}")
    print(f"per-class F1: " + "  ".join(f"{n}={f:.3f}" for n, f in zip(CLASS_NAMES, f1p0)))

    # ---- HR from R-peak detection (what HPS computes), both splits ----
    print(f"\n[INFO] detecting HR (R-peaks) ...")
    hr_v = np.array([detect_hr_bpm(Xv[i]) for i in range(len(Xv))])
    hr   = np.array([detect_hr_bpm(X[i])  for i in range(len(X))])
    print(f"[INFO] test HR detected {len(X)-np.isnan(hr).sum()}/{len(X)}, "
          f"median {np.nanmedian(hr):.1f} bpm")
    for c in range(4):
        h = hr[(y == c) & ~np.isnan(hr)]
        if len(h):
            print(f"   true {CLASS_NAMES[c]:<5}: median HR = {np.median(h):5.1f} bpm  (n={len(h)})")

    # ---- calibrate THR on VAL (select by F1-macro), report on TEST ----
    print(f"\n=== THR calibration on VAL split (select by F1-macro) ===")
    print(f"{'thr':>5}{'val_acc':>9}{'val_f1':>9}")
    best_val = None
    for thr in [50, 52, 54, 55, 56, 58, 60, 62, 65]:
        pred_v = fuse(argmax_v, hr_v, thr)
        accv, f1v, _, _ = metrics(yv, pred_v)
        print(f"{thr:>5}{accv:>9.4f}{f1v:>9.4f}")
        if best_val is None or f1v > best_val[1]:
            best_val = (thr, f1v)
    thr = best_val[0]
    print(f"\n[selected on VAL]  THR = {thr} bpm  (val f1-macro={best_val[1]:.4f})")

    # ---- apply locked THR to TEST ----
    pred = fuse(argmax, hr, thr)
    accB, f1B, f1pB, cmB = metrics(y, pred)
    f1p = f1pB
    print(f"\n=== FUSION on TEST (THR={thr}, locked from val) vs baseline ===")
    print(f"acc      : {acc0:.4f} -> {accB:.4f}   (d {(accB-acc0)*100:+.2f} pp)")
    print(f"f1-macro : {f10:.4f} -> {f1B:.4f}   (d {(f1B-f10)*100:+.2f} pp)")
    print(f"per-class F1:")
    for c in range(4):
        print(f"  {CLASS_NAMES[c]:<5} {f1p0[c]:.3f} -> {f1pB[c]:.3f}   (d {(f1pB[c]-f1p0[c])*100:+.2f})")

    print(f"\nconfusion BEFORE (rows=true):")
    print("        " + "".join(f"{n:>7}" for n in CLASS_NAMES))
    for c in range(4):
        print(f"  {CLASS_NAMES[c]:<5}" + "".join(f"{v:>7}" for v in cm0[c]))
    print(f"\nconfusion AFTER fusion:")
    print("        " + "".join(f"{n:>7}" for n in CLASS_NAMES))
    for c in range(4):
        print(f"  {CLASS_NAMES[c]:<5}" + "".join(f"{v:>7}" for v in cmB[c]))


if __name__ == '__main__':
    main()
