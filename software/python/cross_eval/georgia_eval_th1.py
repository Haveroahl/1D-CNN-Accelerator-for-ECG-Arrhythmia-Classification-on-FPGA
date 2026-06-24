"""Eval TH1 model (float32, trained on merged Chapman+Ningbo) zero-shot on Georgia.

Compares against the Chapman-only QAT-INT8 C2 baseline (90.2%) to see whether
adding Ningbo at TRAIN time improves Georgia transfer / AFIB precision.
Loads case1_model_float32.pth (the model produced by case1_train.py).
Labels untouched; full Georgia by-class tree (5,606).

Usage:
    python cross_eval/georgia_eval_th1.py
"""

import os, sys, glob, json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ptbxl_eval import ECG_CNN

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
TH1_MODEL = r'software/python/results/case_study/case1_model_float32.pth'
ROOT      = r'data/georgia_by_class'
OUTDIR    = r'software/python/results/cross_eval'
FIGDIR    = r'software/python/results/figures'


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
            'confusion_matrix': cm.tolist()}, probs, cm, acc


def plot_figures(y, probs, cm, acc):
    os.makedirs(FIGDIR, exist_ok=True)
    y_oh = np.eye(4)[y]
    auc_c = []
    for c in range(4):
        try: auc_c.append(roc_auc_score(y_oh[:, c], probs[:, c]))
        except ValueError: auc_c.append(float('nan'))
    auc_m = roc_auc_score(y_oh, probs, average='macro', multi_class='ovr')

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Georgia TH1 (Chapman+Ningbo) Zero-shot (acc={acc:.3f})')
    thr = cm.max()/2
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i][j], ha='center', va='center',
                    color='white' if cm[i][j] > thr else 'black', fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'georgia_th1_confusion_matrix.png'), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for c in range(4):
        fpr, tpr, _ = roc_curve(y_oh[:, c], probs[:, c])
        ax.plot(fpr, tpr, label=f'{CLASS_NAMES[c]} (AUC={auc_c[c]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Georgia TH1 ROC (macro-AUC={auc_m:.3f})')
    ax.legend(loc='lower right', fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'georgia_th1_roc.png'), dpi=150); plt.close(fig)
    print(f"[INFO] saved figures -> {FIGDIR}/georgia_th1_confusion_matrix.png, georgia_th1_roc.png")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device={device}")

    model = ECG_CNN().to(device)
    model.load_state_dict(torch.load(TH1_MODEL, map_location=device))
    print(f"[INFO] loaded TH1 model (trained on merged Chapman+Ningbo): {TH1_MODEL}")

    X, y = load_byclass(ROOT)
    print(f"[INFO] Georgia: {dict(zip(CLASS_NAMES, np.bincount(y, minlength=4)))}")

    res = {'mode': 'TH1_chapman+ningbo_float32 -> Georgia zero-shot'}
    metrics, probs, cm, acc = report(model, X, y, device,
                           "TH1 (train Chapman+Ningbo) -> Georgia zero-shot")
    res['result'] = metrics
    plot_figures(y, probs, cm, acc)
    with open(os.path.join(OUTDIR, 'georgia_th1_eval.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\n[INFO] saved: georgia_th1_eval.json")


if __name__ == '__main__':
    main()
