"""Georgia C2 zero-shot — full evaluation report (Chapman QAT-INT8, no fine-tune).

Same report format as ningbo_c2_report.py so the three datasets compare directly:
accuracy, per-class + macro Precision/Recall/F1, one-vs-rest AUC, confusion matrix.
Saves CM + ROC figures and a JSON summary.

Usage:
    python cross_eval/georgia_c2_report.py
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
from ptbxl_eval import load_qat_checkpoint

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
CKPT   = r'software/python/results/qat_int8/model_qat_int8.pth'
ROOT   = r'data/georgia_by_class'
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
    print(f"[INFO] Georgia: {len(y)} records")
    dist = {CLASS_NAMES[c]: int((y == c).sum()) for c in range(4)}
    print(f"[INFO] class dist: {dist}")

    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=128, shuffle=False)
    model = load_qat_checkpoint(CKPT, device)
    model.eval()

    all_logits = []
    with torch.no_grad():
        for xb, _ in loader:
            all_logits.append(model(xb.to(device)).cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    y_pred = logits.argmax(axis=1)
    probs  = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    acc = accuracy_score(y, y_pred)
    p_c, r_c, f_c, sup = precision_recall_fscore_support(y, y_pred, labels=[0,1,2,3], zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y, y_pred, average='macro', zero_division=0)
    y_oh = np.eye(4)[y]
    auc_c = []
    for c in range(4):
        try: auc_c.append(roc_auc_score(y_oh[:, c], probs[:, c]))
        except ValueError: auc_c.append(float('nan'))
    auc_m = roc_auc_score(y_oh, probs, average='macro', multi_class='ovr')
    cm = confusion_matrix(y, y_pred, labels=[0,1,2,3])

    print("\n" + "="*68)
    print(f"GEORGIA  C2 ZERO-SHOT  (Chapman QAT-INT8, no fine-tune)   n={len(y)}")
    print("="*68)
    print(f"Accuracy: {acc:.4f}   macro-F1: {f_m:.4f}   macro-AUC: {auc_m:.4f}\n")
    hdr = f"{'Class':<8}{'Precision':>11}{'Recall':>10}{'F1':>10}{'AUC':>9}{'Support':>10}"
    print(hdr); print("-"*len(hdr))
    for c in range(4):
        print(f"{CLASS_NAMES[c]:<8}{p_c[c]:>11.4f}{r_c[c]:>10.4f}{f_c[c]:>10.4f}{auc_c[c]:>9.4f}{sup[c]:>10d}")
    print("-"*len(hdr))
    print(f"{'macro':<8}{p_m:>11.4f}{r_m:>10.4f}{f_m:>10.4f}{auc_m:>9.4f}{sup.sum():>10d}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("          " + "".join(f"{n:>8}" for n in CLASS_NAMES))
    for c in range(4):
        print(f"  {CLASS_NAMES[c]:<8}" + "".join(f"{cm[c][j]:>8d}" for j in range(4)))

    # figures
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Georgia C2 Zero-shot (acc={acc:.3f})')
    thr = cm.max()/2
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i][j], ha='center', va='center',
                    color='white' if cm[i][j] > thr else 'black', fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'georgia_c2_confusion_matrix.png'), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for c in range(4):
        fpr, tpr, _ = roc_curve(y_oh[:, c], probs[:, c])
        ax.plot(fpr, tpr, label=f'{CLASS_NAMES[c]} (AUC={auc_c[c]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Georgia C2 ROC (macro-AUC={auc_m:.3f})')
    ax.legend(loc='lower right', fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'georgia_c2_roc.png'), dpi=150); plt.close(fig)

    report = {
        'mode': 'C2_zeroshot_georgia', 'n': int(len(y)), 'class_dist': dist,
        'accuracy': float(acc),
        'per_class': {CLASS_NAMES[c]: {'precision': float(p_c[c]), 'recall': float(r_c[c]),
                      'f1': float(f_c[c]), 'auc': float(auc_c[c]), 'support': int(sup[c])} for c in range(4)},
        'macro': {'precision': float(p_m), 'recall': float(r_m), 'f1': float(f_m), 'auc': float(auc_m)},
        'confusion_matrix': cm.tolist(),
    }
    with open(os.path.join(OUTDIR, 'georgia_c2_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] saved: {os.path.join(OUTDIR, 'georgia_c2_report.json')}")


if __name__ == '__main__':
    main()
