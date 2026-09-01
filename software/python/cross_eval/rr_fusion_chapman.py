"""
RR-fusion on Chapman (IN-DISTRIBUTION sanity check).
=====================================================
Before investing in RTL, verify the SB/SR HR-fusion does NOT hurt accuracy on
the source dataset where the CNN already works (94.65%). Reports:
  - baseline argmax (== C1)
  - fusion using DETECTED HR (what HW/HPS computes from the signal)
  - fusion using GROUND-TRUTH HR (Chapman VentricularRate column) — upper bound

THR calibrated on Chapman VAL split (F1-macro), reported on Chapman TEST.
Read-only: loads QAT checkpoint + Chapman dataset.
"""
import os, sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptbxl_eval import ECG_CNN, load_qat_checkpoint, CLASS_NAMES
from rr_fusion_probe import detect_hr_bpm, fuse, SB, SR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.dataset import ChapmanECGDataset

CKPT = os.path.join(os.path.dirname(__file__), '..', 'results', 'qat_int8', 'model_qat_int8.pth')
DATA = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'Chapman')


def metrics(y, pred):
    return (accuracy_score(y, pred),
            f1_score(y, pred, average='macro', zero_division=0),
            f1_score(y, pred, average=None, labels=[0,1,2,3], zero_division=0),
            confusion_matrix(y, pred, labels=[0,1,2,3]))


def load_split(split):
    ds = ChapmanECGDataset(DATA, split=split, seed=42)
    X = np.stack([ds.records[i] for i in range(len(ds))]).astype(np.float32)
    y = np.array(ds.labels)
    hr_gt = np.array([h if h not in (None, 0) else np.nan
                      for h in ds.heart_rates], dtype=float)
    return X, y, hr_gt


def calibrate_thr(argmax, hr, y):
    best = None
    for thr in [50, 52, 54, 55, 56, 58, 60, 62, 65]:
        _, f1, _, _ = metrics(y, fuse(argmax, hr, thr))
        if best is None or f1 > best[1]:
            best = (thr, f1)
    return best[0]


def report(tag, y, argmax, hr, thr):
    pred = fuse(argmax, hr, thr)
    acc, f1, f1p, cm = metrics(y, pred)
    acc0, f10, f1p0, _ = metrics(y, argmax)
    print(f"\n--- {tag} (THR={thr}) ---")
    print(f"acc      : {acc0:.4f} -> {acc:.4f}   (d {(acc-acc0)*100:+.2f} pp)")
    print(f"f1-macro : {f10:.4f} -> {f1:.4f}   (d {(f1-f10)*100:+.2f} pp)")
    for c in range(4):
        print(f"  {CLASS_NAMES[c]:<5} {f1p0[c]:.3f} -> {f1p[c]:.3f}  (d {(f1p[c]-f1p0[c])*100:+.2f})")
    return cm, acc0


def main():
    device = 'cpu'
    model = load_qat_checkpoint(CKPT, device).eval()

    print("[INFO] loading Chapman val + test ...")
    Xv, yv, hrv_gt = load_split('val')
    Xt, yt, hrt_gt = load_split('test')
    print(f"[INFO] val n={len(yv)}  test n={len(yt)}")

    with torch.no_grad():
        amv = model(torch.from_numpy(Xv)).numpy().argmax(1)
        amt = model(torch.from_numpy(Xt)).numpy().argmax(1)

    acc0, f10, f1p0, cm0 = metrics(yt, amt)
    print(f"\n=== BASELINE Chapman TEST (argmax, == C1) ===")
    print(f"acc={acc0:.4f}  f1-macro={f10:.4f}")
    print("per-class F1: " + "  ".join(f"{n}={f:.3f}" for n,f in zip(CLASS_NAMES,f1p0)))

    # detected HR (what HW does)
    print("\n[INFO] detecting HR from signal (val+test) ...")
    hrv_det = np.array([detect_hr_bpm(Xv[i]) for i in range(len(Xv))])
    hrt_det = np.array([detect_hr_bpm(Xt[i]) for i in range(len(Xt))])

    # HR sanity vs ground truth on test
    ok = ~np.isnan(hrt_det) & ~np.isnan(hrt_gt)
    mae = np.mean(np.abs(hrt_det[ok] - hrt_gt[ok]))
    print(f"[INFO] detected vs GT HR: MAE={mae:.1f} bpm on {ok.sum()} records")
    for c in range(4):
        m = (yt==c) & ~np.isnan(hrt_gt)
        print(f"   {CLASS_NAMES[c]:<5} GT median HR = {np.median(hrt_gt[m]):5.1f} bpm (n={m.sum()})")

    # ---- fusion with DETECTED HR (calibrate THR on val-detected) ----
    thr_det = calibrate_thr(amv, hrv_det, yv)
    cm_det, _ = report(f"FUSION detected-HR", yt, amt, hrt_det, thr_det)

    # ---- fusion with GROUND-TRUTH HR (upper bound; calibrate on val-GT) ----
    thr_gt = calibrate_thr(amv, hrv_gt, yv)
    cm_gt, _ = report(f"FUSION groundtruth-HR (upper bound)", yt, amt, hrt_gt, thr_gt)

    print(f"\n=== confusion (rows=true) ===")
    for name, cm in [("BASELINE", cm0), (f"detected THR={thr_det}", cm_det),
                     (f"GT THR={thr_gt}", cm_gt)]:
        print(f"\n{name}:")
        print("        " + "".join(f"{n:>7}" for n in CLASS_NAMES))
        for c in range(4):
            print(f"  {CLASS_NAMES[c]:<5}" + "".join(f"{v:>7}" for v in cm[c]))


if __name__ == '__main__':
    main()
