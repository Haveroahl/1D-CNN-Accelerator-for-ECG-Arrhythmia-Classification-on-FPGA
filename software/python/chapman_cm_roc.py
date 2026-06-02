"""Chapman test-set Confusion Matrix + ROC/AUC for the QAT-INT8 model (Table 5 / Fig.).

Loads the existing A2 power-of-2 QAT-INT8 checkpoint (results/qat_int8/model_qat_int8.pth)
and runs the bit-exact INT8 forward (int8_forward from qat_int8.py) on the Chapman test set.
Does NOT retrain or re-run qat_int8.py — pure evaluation.

Outputs (results/figures/):
  chapman_confusion_matrix.png   — 4x4 CM (counts + row-normalized)
  chapman_roc.png                — one-vs-rest ROC for 4 classes + macro-average AUC
  chapman_cm_roc.json            — CM, per-class AUC, macro AUC, accuracy, F1

Usage:
    cd software/python
    python chapman_cm_roc.py \\
        --checkpoint ./results/qat_int8/model_qat_int8.pth \\
        --output_dir ./results/figures
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantization.qat_int8 import ECG_1DCNN_QAT, int8_forward
from utils.dataset import get_dataloaders, CLASS_NAMES


def collect_logits(qat_model, loader, w_int8, b_int8, w_shift, nb, input_shift):
    """Run INT8 forward over the loader; return logits, preds, labels."""
    qat_model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0]
            y = batch[1]
            logits = int8_forward(qat_model, x, w_int8, b_int8, w_shift, nb, input_shift)
            all_logits.append(logits.cpu().numpy())
            all_labels.extend(y.numpy())
    return np.concatenate(all_logits, axis=0), np.array(all_labels)


def plot_confusion_matrix(cm, class_names, out_path):
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Chapman Test Set — QAT-INT8 Confusion Matrix')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            txt = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
            ax.text(j, i, txt, ha='center', va='center',
                    color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Row-normalized')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc(y_true, probs, class_names, out_path):
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average AUC (interpolated)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], lw=1.8,
                label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    ax.plot(all_fpr, mean_tpr, 'k--', lw=2,
            label=f"macro-avg (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], color='gray', lw=0.8, linestyle=':')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Chapman Test Set — QAT-INT8 ROC (one-vs-rest)')
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {str(i): float(roc_auc[i]) for i in range(n_classes)}, float(macro_auc)


def main():
    p = argparse.ArgumentParser(description='Chapman CM + ROC/AUC for QAT-INT8 (eval only)')
    p.add_argument('--checkpoint', type=str, default='./results/qat_int8/model_qat_int8.pth')
    p.add_argument('--data_dir',   type=str, default='../../data/Chapman')
    p.add_argument('--output_dir', type=str, default='./results/figures')
    p.add_argument('--batch_size', type=int, default=128)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cpu')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    qat_model = ECG_1DCNN_QAT(
        c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
        c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
    ).to(device)
    qat_model.load_state_dict(ckpt['model_state_dict'])
    qat_model.eval()

    w_int8  = {k: np.array(v, dtype=np.int8)    for k, v in ckpt['w_int8'].items()}
    b_int8  = {k: np.array(v, dtype=np.float64) for k, v in ckpt['b_int8'].items()}
    w_shift = ckpt['w_shift']
    nb      = ckpt['nb']
    input_shift = ckpt['input_shift_bits']

    _, _, test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=2)

    logits, labels = collect_logits(qat_model, test_loader, w_int8, b_int8, w_shift, nb, input_shift)
    preds = logits.argmax(axis=1)
    probs = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()

    acc = (preds == labels).mean()
    f1_macro = f1_score(labels, preds, average='macro')
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))

    print(f"\n  Chapman test accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-macro              : {f1_macro:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    cm_path  = os.path.join(args.output_dir, 'chapman_confusion_matrix.png')
    roc_path = os.path.join(args.output_dir, 'chapman_roc.png')
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)
    per_class_auc, macro_auc = plot_roc(labels, probs, CLASS_NAMES, roc_path)

    print(f"  Per-class AUC         : "
          + ", ".join(f"{CLASS_NAMES[int(i)]}={a:.3f}" for i, a in per_class_auc.items()))
    print(f"  Macro-average AUC     : {macro_auc:.4f}")
    print(f"\n  Saved: {cm_path}")
    print(f"  Saved: {roc_path}")

    summary = {
        'source': 'A2 power-of-2 QAT-INT8 (model_qat_int8.pth), eval only',
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'confusion_matrix': cm.tolist(),
        'class_names': CLASS_NAMES,
        'per_class_auc': {CLASS_NAMES[int(i)]: a for i, a in per_class_auc.items()},
        'macro_auc': macro_auc,
    }
    json_path = os.path.join(args.output_dir, 'chapman_cm_roc.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
