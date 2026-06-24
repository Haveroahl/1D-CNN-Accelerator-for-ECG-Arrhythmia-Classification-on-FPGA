"""Combined cross-check — pool ALL Ningbo + ALL PTB-XL into one external test set.

Zero-shot (Chapman QAT-INT8, no fine-tune) over the union of both by-class trees.
No subsampling, no balancing — every record from both datasets is included.
Ground-truth labels untouched (folder index = class). Reports a single combined
accuracy / per-class P/R/F1 / macro-AUC / confusion matrix, plus a per-source
breakdown (Ningbo vs PTB-XL) so the pooled number stays interpretable.

Usage:
    python cross_eval/combined_crosscheck.py
"""

import os, sys, glob, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ptbxl_eval import load_qat_checkpoint   # reuse exact checkpoint loader

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
CKPT    = r'software/python/results/qat_int8/model_qat_int8.pth'
SOURCES = {'ningbo': r'data/ningba_by_class',
           'ptbxl':  r'data/ptbxl_by_class'}
OUTDIR  = r'software/python/results/cross_eval'


def load_byclass(root, drop_sr_frac=0.0, sr_keep_n=None, seed=42):
    """Load every <root>/<CLASS>/*.npy. Returns X, y, src_count_per_class.

    Objective SR (class 3) subsampling of THIS source only (applied to every SR
    file regardless of any prediction; ground-truth labels untouched), to reduce
    class skew. Either:
      sr_keep_n     : keep exactly this many SR records (takes precedence), or
      drop_sr_frac  : random-drop this fraction of SR records.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    per_class = {}
    for label, c in enumerate(CLASS_NAMES):
        files = sorted(glob.glob(os.path.join(root, c, '*.npy')))
        if c == 'SR':
            if sr_keep_n is not None:
                keep_n = min(sr_keep_n, len(files))
            elif drop_sr_frac > 0.0:
                keep_n = int(round(len(files) * (1.0 - drop_sr_frac)))
            else:
                keep_n = len(files)
            if keep_n < len(files):
                keep_idx = np.sort(rng.choice(len(files), size=keep_n, replace=False))
                files = [files[i] for i in keep_idx]
        per_class[c] = len(files)
        for f in files:
            X.append(np.load(f)); y.append(label)
    return np.asarray(X, np.float32), np.asarray(y, np.int64), per_class


def metrics_block(y_true, y_pred, probs, n_label):
    acc = accuracy_score(y_true, y_pred)
    p_c, r_c, f_c, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_label)), zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    try:
        y_onehot = np.eye(n_label)[y_true]
        auc_m = roc_auc_score(y_onehot, probs, average='macro', multi_class='ovr')
    except ValueError:
        auc_m = float('nan')
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_label)))
    return {'acc': float(acc), 'p_c': p_c, 'r_c': r_c, 'f_c': f_c, 'sup': sup,
            'p_macro': float(p_m), 'r_macro': float(r_m), 'f_macro': float(f_m),
            'auc_macro': float(auc_m), 'cm': cm}


def print_block(title, m):
    print("\n" + "=" * 68)
    print(f"{title}   n={int(m['sup'].sum())}")
    print("=" * 68)
    print(f"Accuracy: {m['acc']:.4f}   macro-F1: {m['f_macro']:.4f}   "
          f"macro-AUC: {m['auc_macro']:.4f}\n")
    hdr = f"{'Class':<8}{'Precision':>11}{'Recall':>10}{'F1':>10}{'Support':>10}"
    print(hdr); print("-" * len(hdr))
    for c in range(4):
        print(f"{CLASS_NAMES[c]:<8}{m['p_c'][c]:>11.4f}{m['r_c'][c]:>10.4f}"
              f"{m['f_c'][c]:>10.4f}{m['sup'][c]:>10d}")
    print("-" * len(hdr))
    print(f"{'macro':<8}{m['p_macro']:>11.4f}{m['r_macro']:>10.4f}{m['f_macro']:>10.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("          " + "".join(f"{n:>8}" for n in CLASS_NAMES))
    for c in range(4):
        print(f"  {CLASS_NAMES[c]:<8}" + "".join(f"{m['cm'][c][j]:>8d}" for j in range(4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drop_sr_ptbxl', type=float, default=0.0,
                    help='Random-drop this fraction of PTB-XL SR records (e.g. 0.6).')
    ap.add_argument('--keep_sr_ptbxl', type=int, default=None,
                    help='Keep exactly this many PTB-XL SR records (overrides --drop_sr_ptbxl).')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device={device}  drop_sr_ptbxl={args.drop_sr_ptbxl}  "
          f"keep_sr_ptbxl={args.keep_sr_ptbxl}")

    # ── Load both trees, tag each record with its source ────────────────────
    Xs, ys, srcs = [], [], []
    dist = {}
    for name, root in SOURCES.items():
        drop    = args.drop_sr_ptbxl if name == 'ptbxl' else 0.0
        keep_sr = args.keep_sr_ptbxl if name == 'ptbxl' else None
        X, y, per_class = load_byclass(root, drop_sr_frac=drop,
                                       sr_keep_n=keep_sr, seed=args.seed)
        dist[name] = per_class
        Xs.append(X); ys.append(y)
        srcs.append(np.full(len(y), name, dtype=object))
        print(f"[INFO] {name}: {len(y)} records  {per_class}")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    src = np.concatenate(srcs, axis=0)
    print(f"[INFO] POOLED total: {len(y)} records")

    # ── Forward pass (single model, single pass over the whole pool) ────────
    model = load_qat_checkpoint(CKPT, device)
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=128, shuffle=False)
    all_logits = []
    with torch.no_grad():
        for xb, _ in loader:
            all_logits.append(model(xb.to(device)).cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    y_pred = logits.argmax(axis=1)
    probs  = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    # ── Combined metrics ────────────────────────────────────────────────────
    m_all = metrics_block(y, y_pred, probs, 4)
    print_block("COMBINED  (ALL Ningbo + ALL PTB-XL, zero-shot)", m_all)

    # ── Per-source breakdown (so the pooled number is interpretable) ────────
    per_source = {}
    for name in SOURCES:
        mask = src == name
        ms = metrics_block(y[mask], y_pred[mask], probs[mask], 4)
        print_block(f"  source = {name}", ms)
        per_source[name] = {'acc': ms['acc'], 'f1_macro': ms['f_macro'],
                            'auc_macro': ms['auc_macro'], 'n': int(mask.sum()),
                            'confusion_matrix': ms['cm'].tolist()}

    # ── Save JSON ────────────────────────────────────────────────────────────
    report = {
        'mode': 'combined_zeroshot_ningbo_plus_ptbxl',
        'sources': dist,
        'n_total': int(len(y)),
        'combined': {
            'accuracy': m_all['acc'],
            'macro': {'precision': m_all['p_macro'], 'recall': m_all['r_macro'],
                      'f1': m_all['f_macro'], 'auc': m_all['auc_macro']},
            'per_class': {CLASS_NAMES[c]: {
                'precision': float(m_all['p_c'][c]), 'recall': float(m_all['r_c'][c]),
                'f1': float(m_all['f_c'][c]), 'support': int(m_all['sup'][c])}
                for c in range(4)},
            'confusion_matrix': m_all['cm'].tolist(),
        },
        'per_source': per_source,
    }
    out = os.path.join(OUTDIR, 'combined_crosscheck.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] saved: {out}")


if __name__ == '__main__':
    main()
