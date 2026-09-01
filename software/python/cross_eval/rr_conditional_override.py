"""
Conditional (dead-band) HR override for SB/SR — zero-shot, no training.
=======================================================================
Prior study (RR_FUSION_FINDINGS.md) showed UNCONDITIONAL HR override fixes
PTB-XL zero-shot (+13pp) but HURTS Chapman in-dist (-3.6pp), because R-peak
error (~8 bpm) flips records the CNN already had right, near the 60-bpm boundary.

This tests a CONDITIONAL variant that was NOT tried: only override when HR is
UNAMBIGUOUS — far from the boundary. Two thresholds (lo, hi) define a dead-band:
    HR < lo            -> force SB
    HR > hi            -> force SR
    lo <= HR <= hi     -> keep CNN argmax  (dead-band: don't trust HR here)
AFIB/GSVT always kept. lo==hi recovers the old unconditional single-threshold.

Question: does a dead-band exist that recovers PTB-XL while leaving Chapman
(near-)untouched? Reports a joint sweep over both datasets so the trade-off is
visible in one place. Zero-shot: no parameter is trained on either dataset.

Read-only: QAT checkpoint + Chapman CSVs + PTB-XL npz.
"""
import os, sys, argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ptbxl_eval import load_qat_checkpoint, CLASS_NAMES
from rr_fusion_probe import detect_hr_bpm           # scipy detector (same as prior study)
from pan_tompkins import pan_tompkins_hr

SB, SR = 2, 3

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, '..', 'results', 'qat_int8', 'model_qat_int8.pth')
CHAP = os.path.join(HERE, '..', '..', '..', 'data', 'Chapman')
PTBX = os.path.join(HERE, '..', '..', '..', 'data', 'ptbxl_processed', 'ptbxl_dataset.npz')


def metrics(y, pred):
    return (accuracy_score(y, pred),
            f1_score(y, pred, average='macro', zero_division=0))


def cond_override(argmax, hr, lo, hi):
    """Dead-band override of SB/SR. lo<=hi. NaN HR -> keep CNN."""
    out = argmax.copy()
    sbsr = np.isin(argmax, [SB, SR]) & ~np.isnan(hr)
    out[sbsr & (hr < lo)] = SB
    out[sbsr & (hr > hi)] = SR
    return out


def detect_hr_pt(sig):
    bpm, _ = pan_tompkins_hr(sig)
    return bpm


def load_chapman_test():
    from utils.dataset import ChapmanECGDataset
    ds = ChapmanECGDataset(CHAP, split='test', seed=42)
    X = np.stack(ds.records).astype(np.float32)
    y = np.array(ds.labels)
    return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--detector', choices=['scipy', 'pan'], default='pan')
    args = p.parse_args()
    device = 'cpu'
    det = detect_hr_pt if args.detector == 'pan' else detect_hr_bpm
    print(f"[INFO] detector={args.detector}")

    model = load_qat_checkpoint(CKPT, device).eval()

    print("[INFO] loading Chapman test ...")
    Xc, yc = load_chapman_test()
    print("[INFO] loading PTB-XL test ...")
    d = np.load(PTBX)
    Xp, yp = d['X_test'].astype(np.float32), d['y_test'].astype(int)

    with torch.no_grad():
        am_c = model(torch.from_numpy(Xc)).numpy().argmax(1)
        am_p = model(torch.from_numpy(Xp)).numpy().argmax(1)

    print("[INFO] detecting HR (both sets) ...")
    hr_c = np.array([det(Xc[i]) for i in range(len(Xc))])
    hr_p = np.array([det(Xp[i]) for i in range(len(Xp))])

    bc_a, bc_f = metrics(yc, am_c)
    bp_a, bp_f = metrics(yp, am_p)
    print(f"\nBASELINE (zero-shot argmax)")
    print(f"  Chapman: acc={bc_a:.4f} f1={bc_f:.4f}")
    print(f"  PTB-XL : acc={bp_a:.4f} f1={bp_f:.4f}")

    # ---- joint sweep over dead-band (lo, hi) ----
    print(f"\nDEAD-BAND SWEEP  (override SB if HR<lo, SR if HR>hi, else keep CNN)")
    print(f"{'lo':>4}{'hi':>4} | {'chap_acc':>9}{'(d)':>8} | {'ptb_acc':>9}{'(d)':>8} | {'ptb_f1':>8}")
    print("-" * 64)
    grid = [45, 50, 55, 60, 65, 70, 75]
    best = None
    for lo in grid:
        for hi in grid:
            if hi < lo:
                continue
            pc = cond_override(am_c, hr_c, lo, hi)
            pp = cond_override(am_p, hr_p, lo, hi)
            ca, _ = metrics(yc, pc)
            pa, pf = metrics(yp, pp)
            dc = (ca - bc_a) * 100
            dp = (pa - bp_a) * 100
            mark = ""
            # interesting = chapman barely hurt (>= -0.5pp) AND ptb improved (>= +3pp)
            if dc >= -0.5 and dp >= 3.0:
                mark = "  <== safe gain"
                if best is None or dp > best[0]:
                    best = (dp, lo, hi, ca, pa)
            print(f"{lo:>4}{hi:>4} | {ca:>9.4f}{dc:>+8.2f} | {pa:>9.4f}{dp:>+8.2f} | {pf:>8.4f}{mark}")

    print("\n" + "=" * 64)
    if best:
        dp, lo, hi, ca, pa = best
        print(f"BEST SAFE dead-band: lo={lo} hi={hi}")
        print(f"  Chapman {bc_a:.4f} -> {ca:.4f}  ({(ca-bc_a)*100:+.2f} pp)")
        print(f"  PTB-XL  {bp_a:.4f} -> {pa:.4f}  ({(pa-bp_a)*100:+.2f} pp)")
    else:
        print("NO dead-band recovers PTB-XL >=+3pp while keeping Chapman >=-0.5pp.")
        print("Conditional override does not escape the prior trade-off.")
    print("=" * 64)


if __name__ == '__main__':
    main()
