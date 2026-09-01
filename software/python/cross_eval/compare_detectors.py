"""
Compare HR detectors (scipy vs Pan-Tompkins) for fusion, on BOTH datasets.
===========================================================================
Decides whether a better R-peak detector makes SB/SR fusion safe in-distribution
(Chapman) while keeping the cross-dataset gain (PTB-XL).

For each dataset: THR_BPM calibrated on VAL (F1-macro), reported on TEST.
Chapman has ground-truth HR (VentricularRate) -> also report MAE + GT-fusion.
"""
import os, sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptbxl_eval import ECG_CNN, load_qat_checkpoint, CLASS_NAMES
from rr_fusion_probe import detect_hr_bpm as scipy_hr, fuse, SB, SR
from pan_tompkins import pan_tompkins_hr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.dataset import ChapmanECGDataset

CKPT = os.path.join(os.path.dirname(__file__), '..', 'results', 'qat_int8', 'model_qat_int8.pth')
CHAP = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'Chapman')
PTBXL = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'ptbxl_processed', 'ptbxl_dataset.npz')


def af1(y, pred):
    return accuracy_score(y, pred), f1_score(y, pred, average='macro', zero_division=0)


def pt_hr(sig):
    bpm, _ = pan_tompkins_hr(sig)
    return bpm


def calib(argmax, hr, y):
    best = None
    for thr in [48, 50, 52, 54, 55, 56, 58, 60, 62, 65]:
        _, f1 = af1(y, fuse(argmax, hr, thr))
        if best is None or f1 > best[1]:
            best = (thr, f1)
    return best[0]


def run_dataset(name, Xv, yv, Xt, yt, model, hrt_gt=None):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    with torch.no_grad():
        amv = model(torch.from_numpy(Xv)).numpy().argmax(1)
        amt = model(torch.from_numpy(Xt)).numpy().argmax(1)
    acc0, f10 = af1(yt, amt)
    print(f"baseline argmax:        acc={acc0:.4f}  f1={f10:.4f}")

    for label, fn in [("scipy", scipy_hr), ("pan-tompkins", pt_hr)]:
        hv = np.array([fn(Xv[i]) for i in range(len(Xv))])
        ht = np.array([fn(Xt[i]) for i in range(len(Xt))])
        thr = calib(amv, hv, yv)
        acc, f1 = af1(yt, fuse(amt, ht, thr))
        line = f"fusion {label:<13} THR={thr:>3}  acc={acc:.4f}  f1={f1:.4f}   (d {(acc-acc0)*100:+.2f}pp)"
        if hrt_gt is not None:
            ok = ~np.isnan(ht) & ~np.isnan(hrt_gt)
            mae = np.mean(np.abs(ht[ok] - hrt_gt[ok]))
            line += f"   MAE={mae:.1f}bpm"
        print(line)

    if hrt_gt is not None:
        hv_gt = HR_GT_VAL
        thr = calib(amv, hv_gt, yv)
        acc, f1 = af1(yt, fuse(amt, hrt_gt, thr))
        print(f"fusion ground-truth   THR={thr:>3}  acc={acc:.4f}  f1={f1:.4f}   (d {(acc-acc0)*100:+.2f}pp)  [upper bound]")


def main():
    model = load_qat_checkpoint(CKPT, 'cpu').eval()

    # ---- Chapman ----
    def load_chap(split):
        ds = ChapmanECGDataset(CHAP, split=split, seed=42)
        X = np.stack([ds.records[i] for i in range(len(ds))]).astype(np.float32)
        y = np.array(ds.labels)
        hr = np.array([h if h not in (None,0) else np.nan for h in ds.heart_rates], dtype=float)
        return X, y, hr
    Xv, yv, hrv = load_chap('val')
    Xt, yt, hrt = load_chap('test')
    global HR_GT_VAL
    HR_GT_VAL = hrv
    run_dataset("CHAPMAN (in-distribution)", Xv, yv, Xt, yt, model, hrt_gt=hrt)

    # ---- PTB-XL ----
    d = np.load(PTBXL)
    run_dataset("PTB-XL (cross-dataset zero-shot)",
                d['X_val'].astype(np.float32), d['y_val'].astype(int),
                d['X_test'].astype(np.float32), d['y_test'].astype(int), model)


if __name__ == '__main__':
    main()
