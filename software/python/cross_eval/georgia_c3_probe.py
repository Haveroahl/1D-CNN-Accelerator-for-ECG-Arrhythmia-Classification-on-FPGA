"""Georgia C3 — linear probe (freeze conv from Chapman, retrain FC on Georgia).

Loads the Chapman QAT-INT8 checkpoint as float, freezes all conv layers, and
retrains ONLY the FC layer on a Georgia train split. Reports zero-shot (C2) vs
linear-probe (C3) on the SAME held-out Georgia test split, so the precision/F1
gain is measured on data the FC never saw (no leakage, no label change).

Split: record-level 70/15/15 (seed=42) over data/georgia_by_class.

Usage:
    python cross_eval/georgia_c3_probe.py
"""

import os, sys, glob, json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ptbxl_eval import load_qat_checkpoint, finetune

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
CKPT   = r'software/python/results/qat_int8/model_qat_int8.pth'
ROOT   = r'data/georgia_by_class'
OUTDIR = r'software/python/results/cross_eval'


def load_byclass(root):
    X, y = [], []
    for label, c in enumerate(CLASS_NAMES):
        for f in sorted(glob.glob(os.path.join(root, c, '*.npy'))):
            X.append(np.load(f)); y.append(label)
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


def split_7015(X, y, seed=42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y)); n = len(perm)
    tr, va, te = perm[:int(.70*n)], perm[int(.70*n):int(.85*n)], perm[int(.85*n):]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


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
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    print(f"[INFO] device={device}")

    X, y = load_byclass(ROOT)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split_7015(X, y)
    print(f"[INFO] Georgia split: train={len(ytr)} val={len(yva)} test={len(yte)}")
    print(f"[INFO] test class dist: {dict(zip(CLASS_NAMES, np.bincount(yte, minlength=4)))}")

    res = {}
    # ── C2 zero-shot on the SAME test split (fair baseline) ─────────────
    model_c2 = load_qat_checkpoint(CKPT, device)
    res['C2_zeroshot_testsplit'] = report(model_c2, Xte, yte, device,
                                           "C2 zero-shot (Georgia test split)")

    # ── C3 linear probe: freeze conv, retrain FC on train split ─────────
    model_c3 = load_qat_checkpoint(CKPT, device)
    model_c3.freeze_conv()
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=128, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                    batch_size=128, shuffle=False)
    print("\n[INFO] training linear probe (FC only) on Georgia ...")
    model_c3 = finetune(model_c3, tr, va, device, epochs=30, lr=1e-3, label='georgia_probe')
    res['C3_linear_probe_testsplit'] = report(model_c3, Xte, yte, device,
                                              "C3 linear probe (Georgia test split)")

    torch.save(model_c3.fc.state_dict(), os.path.join(OUTDIR, 'georgia_fc_adapter.pth'))
    with open(os.path.join(OUTDIR, 'georgia_c3_probe.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\n[INFO] saved: georgia_c3_probe.json + georgia_fc_adapter.pth")


if __name__ == '__main__':
    main()
