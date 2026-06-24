"""Georgia C2 zero-shot with AFIB random-upsampled (Chapman QAT-INT8, no fine-tune).

WHAT-IF / SANITY ONLY. This is zero-shot: the model is NOT retrained, so duplicating
AFIB test records gives the model no new information — every original record's
prediction is identical. AFIB precision/recall therefore stay ~unchanged (only
the duplicated copies repeat the same TP/FP ratio); only macro-F1 shifts slightly
because the class proportions change. It does NOT test "is AFIB weak due to too
few samples" (that would require oversampling at TRAIN time). Reported with that
caveat. Labels untouched — duplicates carry their true AFIB label.

Usage:
    python cross_eval/georgia_c2_upsample.py --afib_target 1400
"""

import os, sys, glob, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ptbxl_eval import load_qat_checkpoint, ECG_CNN

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
CKPT = r'software/python/results/qat_int8/model_qat_int8.pth'
TH1_MODEL = r'software/python/results/case_study/case1_model_float32.pth'
ROOT = r'data/georgia_by_class'
OUTDIR = r'software/python/results/cross_eval'


def load_byclass(root):
    X, y = [], []
    for label, c in enumerate(CLASS_NAMES):
        for f in sorted(glob.glob(os.path.join(root, c, '*.npy'))):
            X.append(np.load(f)); y.append(label)
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


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
    ap.add_argument('--afib_target', type=int, default=1400)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--model', choices=['chapman_qat', 'th1'], default='chapman_qat',
                    help='chapman_qat = Chapman-only QAT-INT8; th1 = Chapman+Ningbo float32')
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device={device}  model={args.model}")

    X, y = load_byclass(ROOT)
    print(f"[INFO] Georgia original: {dict(zip(CLASS_NAMES, np.bincount(y, minlength=4)))}")

    if args.model == 'th1':
        model = ECG_CNN().to(device)
        model.load_state_dict(torch.load(TH1_MODEL, map_location=device))
    else:
        model = load_qat_checkpoint(CKPT, device)
    res = {}
    res['baseline'] = report(model, X, y, device, "C2 zero-shot ORIGINAL (no upsample)")

    # random upsample AFIB (class 0) with replacement up to afib_target
    rng = np.random.default_rng(args.seed)
    afib_idx = np.where(y == 0)[0]
    n_have = len(afib_idx)
    n_add = max(0, args.afib_target - n_have)
    dup = rng.choice(afib_idx, size=n_add, replace=True)
    Xu = np.concatenate([X, X[dup]], axis=0)
    yu = np.concatenate([y, y[dup]], axis=0)
    print(f"\n[INFO] AFIB upsampled {n_have} -> {n_have + n_add} (+{n_add} random duplicates)")
    print(f"[INFO] Georgia upsampled: {dict(zip(CLASS_NAMES, np.bincount(yu, minlength=4)))}")
    res['afib_upsampled'] = report(model, Xu, yu, device,
                                   f"C2 zero-shot AFIB upsampled to {args.afib_target}")

    with open(os.path.join(OUTDIR, 'georgia_c2_upsample.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\n[INFO] saved: georgia_c2_upsample.json")


if __name__ == '__main__':
    main()
