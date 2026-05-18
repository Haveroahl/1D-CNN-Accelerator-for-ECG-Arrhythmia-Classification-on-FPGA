"""
Evaluation Utilities
====================
Metrics: Accuracy, F1-macro, Confusion Matrix, per-class Recall/Precision/F1
"""

import numpy as np
import torch


def evaluate_model(model, dataloader, device):
    """
    Run model inference on a dataloader.

    Returns:
        preds:  (N,) int array of predicted classes
        labels: (N,) int array of ground-truth classes
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for ecg, labels, _ in dataloader:
            ecg    = ecg.unsqueeze(1).to(device)   # (B, 1, 2500)
            labels = labels.to(device)

            logits    = model(ecg)
            predicted = logits.argmax(dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels, class_names):
    """
    Compute Accuracy, F1-macro, confusion matrix, per-class metrics.

    Args:
        preds, labels: (N,) int arrays
        class_names:   list of str

    Returns:
        dict with keys: accuracy, f1_macro, confusion_matrix, per_class
    """
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true][pred] += 1

    accuracy  = np.trace(cm) / np.sum(cm)
    f1_scores = []
    per_class = {}

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * recall * precision / (recall + precision)
              if (recall + precision) > 0 else 0.0)

        per_class[class_names[i]] = {
            'recall':    recall,
            'precision': precision,
            'f1':        f1,
            'support':   int(np.sum(cm[i, :])),
        }
        f1_scores.append(f1)

    return {
        'accuracy':         float(accuracy),
        'f1_macro':         float(np.mean(f1_scores)),
        'confusion_matrix': cm,
        'per_class':        per_class,
    }


def print_classification_report(metrics, title=""):
    """Print a formatted classification report."""
    if title:
        print(f"\n  {'─'*50}")
        print(f"  {title}")
        print(f"  {'─'*50}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1-macro : {metrics['f1_macro']:.4f}")
    print()
    header = f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    print(header)
    print("  " + "-" * 50)
    for cls, m in metrics['per_class'].items():
        print(f"  {cls:<10} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10d}")
    print()
    print("  Confusion Matrix (rows=true, cols=pred):")
    cm      = metrics['confusion_matrix']
    classes = list(metrics['per_class'].keys())
    header  = "  " + " " * 8 + "".join(f"{c:>8}" for c in classes)
    print(header)
    for i, cls in enumerate(classes):
        row = "  " + f"{cls:<8}" + "".join(f"{cm[i,j]:>8d}" for j in range(len(classes)))
        print(row)
