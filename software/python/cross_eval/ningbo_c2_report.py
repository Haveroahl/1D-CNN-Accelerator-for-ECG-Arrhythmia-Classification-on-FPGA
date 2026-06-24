"""Ningbo C2 zero-shot — full evaluation report.

Loads the Chapman QAT-INT8 checkpoint, runs zero-shot over the FULL Ningbo-only
by-class tree (Chapman half already excluded in preprocess), and reports:
Accuracy, per-class + macro/weighted Precision/Recall/F1, one-vs-rest AUC
(per-class + macro), and the confusion matrix. Saves CM + ROC figures and a
JSON/Markdown summary.

Usage:
    python cross_eval/ningbo_c2_report.py
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
from ptbxl_eval import load_qat_checkpoint   # reuse exact checkpoint loader

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
CKPT   = r'software/python/results/qat_int8/model_qat_int8.pth'
ROOT   = r'data/ningba_by_class'
OUTDIR = r'software/python/results/cross_eval'
FIGDIR = r'software/python/results/figures'


def load_byclass(root):
    X, y = [], []
    for label, c in enumerate(CLASS_NAMES):
        for f in sorted(glob.glob(os.path.join(root, c, '*.npy'))):
            X.append(np.load(f)); y.append(label)
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device={device}")

    X, y = load_byclass(ROOT)
    print(f"[INFO] Ningbo (Chapman half excluded): {len(y)} records")
    dist = {CLASS_NAMES[c]: int((y == c).sum()) for c in range(4)}
    print(f"[INFO] class dist: {dist}")

    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=128, shuffle=False)

    model = load_qat_checkpoint(CKPT, device)
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(yb.tolist())
    logits = np.concatenate(all_logits, axis=0)
    y_true = np.asarray(all_labels)
    y_pred = logits.argmax(axis=1)
    probs  = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    # ── Scalar + per-class metrics ──────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    p_c, r_c, f_c, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1,2,3], zero_division=0)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)

    # ── One-vs-rest AUC ─────────────────────────────────────────────────
    y_onehot = np.eye(4)[y_true]
    auc_c = []
    for c in range(4):
        try:
            auc_c.append(roc_auc_score(y_onehot[:, c], probs[:, c]))
        except ValueError:
            auc_c.append(float('nan'))
    auc_macro = roc_auc_score(y_onehot, probs, average='macro', multi_class='ovr')
    auc_w     = roc_auc_score(y_onehot, probs, average='weighted', multi_class='ovr')

    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])

    # ── Print table ─────────────────────────────────────────────────────
    print("\n" + "="*68)
    print(f"NINGBO  C2 ZERO-SHOT  (Chapman QAT-INT8, no fine-tune)   n={len(y_true)}")
    print("="*68)
    print(f"Accuracy: {acc:.4f}\n")
    hdr = f"{'Class':<8}{'Precision':>11}{'Recall':>10}{'F1':>10}{'AUC':>9}{'Support':>10}"
    print(hdr); print("-"*len(hdr))
    for c in range(4):
        print(f"{CLASS_NAMES[c]:<8}{p_c[c]:>11.4f}{r_c[c]:>10.4f}"
              f"{f_c[c]:>10.4f}{auc_c[c]:>9.4f}{sup[c]:>10d}")
    print("-"*len(hdr))
    print(f"{'macro':<8}{p_macro:>11.4f}{r_macro:>10.4f}{f_macro:>10.4f}{auc_macro:>9.4f}{sup.sum():>10d}")
    print(f"{'weighted':<8}{p_w:>11.4f}{r_w:>10.4f}{f_w:>10.4f}{auc_w:>9.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("            " + "".join(f"{n:>8}" for n in CLASS_NAMES))
    for c in range(4):
        print(f"  {CLASS_NAMES[c]:<8}" + "".join(f"{cm[c][j]:>8d}" for j in range(4)))

    # ── Figure: confusion matrix ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Ningbo C2 Zero-shot (acc={acc:.3f})')
    thr = cm.max() / 2
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i][j], ha='center', va='center',
                    color='white' if cm[i][j] > thr else 'black', fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cm_path = os.path.join(FIGDIR, 'ningbo_c2_confusion_matrix.png')
    fig.savefig(cm_path, dpi=150); plt.close(fig)

    # ── Figure: ROC one-vs-rest ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for c in range(4):
        fpr, tpr, _ = roc_curve(y_onehot[:, c], probs[:, c])
        ax.plot(fpr, tpr, label=f'{CLASS_NAMES[c]} (AUC={auc_c[c]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Ningbo C2 ROC (macro-AUC={auc_macro:.3f})')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    roc_path = os.path.join(FIGDIR, 'ningbo_c2_roc.png')
    fig.savefig(roc_path, dpi=150); plt.close(fig)

    # ── Save JSON ───────────────────────────────────────────────────────
    report = {
        'mode': 'C2_zeroshot_ningbo',
        'n': int(len(y_true)),
        'class_dist': dist,
        'accuracy': float(acc),
        'per_class': {CLASS_NAMES[c]: {
            'precision': float(p_c[c]), 'recall': float(r_c[c]),
            'f1': float(f_c[c]), 'auc': float(auc_c[c]),
            'support': int(sup[c])} for c in range(4)},
        'macro':    {'precision': float(p_macro), 'recall': float(r_macro),
                     'f1': float(f_macro), 'auc': float(auc_macro)},
        'weighted': {'precision': float(p_w), 'recall': float(r_w),
                     'f1': float(f_w), 'auc': float(auc_w)},
        'confusion_matrix': cm.tolist(),
    }
    json_path = os.path.join(OUTDIR, 'ningbo_c2_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n[INFO] saved: {json_path}")
    print(f"[INFO] saved: {cm_path}")
    print(f"[INFO] saved: {roc_path}")


if __name__ == '__main__':
    main()
