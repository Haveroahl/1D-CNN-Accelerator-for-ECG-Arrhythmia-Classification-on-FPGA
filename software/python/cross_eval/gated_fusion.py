"""
Gated fusion — only override SB/SR when the CNN is UNSURE.
==========================================================
Insight from compare_detectors.py:
  fusion helps when CNN is wrong (PTB-XL, +13pp) but hurts when CNN is right
  (Chapman, -3.6pp even with Pan-Tompkins). So gate the override on CNN
  confidence: only let HR decide when |logit_SB - logit_SR| < GATE.

  Chapman: CNN confident on SB/SR -> large gap -> gate rarely fires -> keep 94.5%
  PTB-XL : CNN unsure             -> small gap -> gate fires       -> keep +13pp

Uses Pan-Tompkins HR. THR_BPM fixed at 60 (clinical). GATE calibrated on each
dataset's VAL split by F1-macro, reported on TEST. Reports gate-fire rate.
"""
import os, sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptbxl_eval import ECG_CNN, load_qat_checkpoint, CLASS_NAMES
from pan_tompkins import pan_tompkins_hr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.dataset import ChapmanECGDataset

CKPT  = os.path.join(os.path.dirname(__file__), '..', 'results', 'qat_int8', 'model_qat_int8.pth')
CHAP  = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'Chapman')
PTBXL = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'ptbxl_processed', 'ptbxl_dataset.npz')
SB, SR = 2, 3
THR_BPM = 60.0


def af1(y, p):
    return accuracy_score(y, p), f1_score(y, p, average='macro', zero_division=0)


def pt_hr(sig):
    return pan_tompkins_hr(sig)[0]


def gated_fuse(logits, hr, gate, thr_bpm=THR_BPM):
    """Override class to {SB,SR} by HR only when argmax in {SB,SR} AND the
    SB-vs-SR logit gap is below `gate` (CNN unsure). Else keep argmax."""
    argmax = logits.argmax(1)
    out = argmax.copy()
    gap = np.abs(logits[:, SB] - logits[:, SR])
    fire = np.isin(argmax, [SB, SR]) & ~np.isnan(hr) & (gap < gate)
    out[fire] = np.where(hr[fire] < thr_bpm, SB, SR)
    return out, fire


def calib_gate(logits, hr, y):
    best = None
    # gate=0 -> never fire (==baseline); large gate -> always fire (==plain fusion)
    for gate in [0, 50, 100, 200, 400, 800, 1600, 3200, 1e9]:
        pred, _ = gated_fuse(logits, hr, gate)
        _, f1 = af1(y, pred)
        if best is None or f1 > best[1]:
            best = (gate, f1)
    return best[0]


def run(name, Xv, yv, Xt, yt, model):
    print(f"\n{'='*64}\n  {name}\n{'='*64}")
    with torch.no_grad():
        lv = model(torch.from_numpy(Xv)).numpy()
        lt = model(torch.from_numpy(Xt)).numpy()
    amt = lt.argmax(1)
    acc0, f10 = af1(yt, amt)

    hv = np.array([pt_hr(Xv[i]) for i in range(len(Xv))])
    ht = np.array([pt_hr(Xt[i]) for i in range(len(Xt))])

    # confidence gap diagnostics on SB/SR-predicted samples
    gap_t = np.abs(lt[:, SB] - lt[:, SR])
    sbsr = np.isin(amt, [SB, SR])
    print(f"baseline:            acc={acc0:.4f}  f1={f10:.4f}")
    print(f"SB/SR logit-gap (predicted SB/SR): median={np.median(gap_t[sbsr]):.0f}  "
          f"p25={np.percentile(gap_t[sbsr],25):.0f}  p75={np.percentile(gap_t[sbsr],75):.0f}")

    # plain fusion (gate=inf)
    pred_p, _ = gated_fuse(lt, ht, 1e9)
    accp, f1p = af1(yt, pred_p)
    print(f"plain fusion:        acc={accp:.4f}  f1={f1p:.4f}   (d {(accp-acc0)*100:+.2f}pp)")

    # gated fusion (gate calibrated on val)
    gate = calib_gate(lv, hv, yv)
    pred_g, fire = gated_fuse(lt, ht, gate)
    accg, f1g = af1(yt, pred_g)
    fire_rate = 100.0 * fire.sum() / len(yt)
    print(f"gated fusion GATE={gate:<6}: acc={accg:.4f}  f1={f1g:.4f}   "
          f"(d {(accg-acc0)*100:+.2f}pp)  gate fired on {fire.sum()}/{len(yt)} ({fire_rate:.1f}%)")


def main():
    model = load_qat_checkpoint(CKPT, 'cpu').eval()

    def load_chap(s):
        ds = ChapmanECGDataset(CHAP, split=s, seed=42)
        X = np.stack([ds.records[i] for i in range(len(ds))]).astype(np.float32)
        return X, np.array(ds.labels)
    Xv, yv = load_chap('val'); Xt, yt = load_chap('test')
    run("CHAPMAN (in-distribution)", Xv, yv, Xt, yt, model)

    d = np.load(PTBXL)
    run("PTB-XL (cross-dataset zero-shot)",
        d['X_val'].astype(np.float32), d['y_val'].astype(int),
        d['X_test'].astype(np.float32), d['y_test'].astype(int), model)


if __name__ == '__main__':
    main()
