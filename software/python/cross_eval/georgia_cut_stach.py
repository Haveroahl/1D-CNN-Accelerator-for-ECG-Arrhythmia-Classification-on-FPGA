"""Georgia cross-check after randomly cutting 50 Sinus-Tachycardia (STach) records.

Objective composition change ONLY: drop 50 STach records chosen at random
(seed=42), WITHOUT looking at any prediction. STach (SNOMED 427084000) is the
dominant GSVT subclass on Georgia (96%), so this measures whether thinning it
changes the GSVT->AFIB confusion. Labels untouched; no record is selected by
its prediction outcome.

Maps recname -> STach via the original .hea #Dx line (georgia_by_class collapses
subclass into GSVT, so we re-read headers to know which GSVT files are STach).

Usage:
    python cross_eval/georgia_cut_stach.py --cut 50
"""

import os, sys, glob, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ptbxl_eval import ECG_CNN

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
TH1_MODEL = r'software/python/results/case_study/case1_model_float32.pth'
ROOT      = r'data/georgia_by_class'
GEO_RAW   = r'data/Georgia'
OUTDIR    = r'software/python/results/cross_eval'
STACH_CODE = '427084000'


def stach_recnames():
    """Return set of recnames whose #Dx contains the STach SNOMED code."""
    names = set()
    for h in glob.glob(os.path.join(GEO_RAW, '**', '*.hea'), recursive=True):
        with open(h) as f:
            for line in f:
                if line.startswith('#Dx:'):
                    codes = [c.strip() for c in line.split(':', 1)[1].strip().split(',')]
                    if STACH_CODE in codes:
                        names.add(os.path.splitext(os.path.basename(h))[0])
                    break
    return names


def load_byclass_with_names(root):
    X, y, names = [], [], []
    for label, c in enumerate(CLASS_NAMES):
        for f in sorted(glob.glob(os.path.join(root, c, '*.npy'))):
            X.append(np.load(f)); y.append(label)
            names.append(os.path.splitext(os.path.basename(f))[0])
    return np.asarray(X, np.float32), np.asarray(y, np.int64), names


def report(model, X, y, device, tag):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=256, shuffle=False)
    logits = []
    with torch.no_grad():
        for xb, _ in loader:
            logits.append(model(xb.to(device)).cpu().numpy())
    logits = np.concatenate(logits, 0)
    pred = logits.argmax(1)
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    acc = accuracy_score(y, pred)
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=[0,1,2,3], zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y, pred, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(np.eye(4)[y], probs, average='macro', multi_class='ovr')
    except ValueError:
        auc = float('nan')
    cm = confusion_matrix(y, pred, labels=[0,1,2,3])
    print(f"\n=== {tag}  n={len(y)} ===")
    print(f"acc={acc:.4f}  macro-P={pm:.4f}  macro-R={rm:.4f}  macro-F1={fm:.4f}  macro-AUC={auc:.4f}")
    for c in range(4):
        print(f"  {CLASS_NAMES[c]:<5} P={p[c]:.4f} R={r[c]:.4f} F1={f[c]:.4f} n={sup[c]}")
    print("  CM:", cm.tolist())
    return {'acc': float(acc), 'macro_p': float(pm), 'macro_r': float(rm),
            'macro_f1': float(fm), 'macro_auc': float(auc),
            'per_class': {CLASS_NAMES[c]: {'precision': float(p[c]), 'recall': float(r[c]),
                          'f1': float(f[c]), 'support': int(sup[c])} for c in range(4)},
            'confusion_matrix': cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cut', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ECG_CNN().to(device)
    model.load_state_dict(torch.load(TH1_MODEL, map_location=device))

    X, y, names = load_byclass_with_names(ROOT)
    print(f"[INFO] Georgia: {dict(zip(CLASS_NAMES, np.bincount(y, minlength=4)))}")

    stach = stach_recnames()
    stach_pos = [i for i, nm in enumerate(names) if nm in stach]
    print(f"[INFO] STach records found in GSVT: {len(stach_pos)}")

    res = {}
    res['baseline'] = report(model, X, y, device, "TH1 -> Georgia (full)")

    # randomly drop `cut` STach records (no prediction used)
    rng = np.random.default_rng(args.seed)
    drop = set(rng.choice(stach_pos, size=min(args.cut, len(stach_pos)), replace=False).tolist())
    keep = [i for i in range(len(y)) if i not in drop]
    Xc, yc = X[keep], y[keep]
    print(f"\n[INFO] dropped {len(drop)} random STach records (seed={args.seed})")
    print(f"[INFO] Georgia after cut: {dict(zip(CLASS_NAMES, np.bincount(yc, minlength=4)))}")
    res['after_cut_stach'] = report(model, Xc, yc, device,
                                    f"TH1 -> Georgia (cut {len(drop)} random STach)")

    with open(os.path.join(OUTDIR, 'georgia_cut_stach.json'), 'w') as fp:
        json.dump(res, fp, indent=2)
    print(f"\n[INFO] saved: georgia_cut_stach.json")


if __name__ == '__main__':
    main()
