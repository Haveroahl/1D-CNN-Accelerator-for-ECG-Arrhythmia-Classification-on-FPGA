"""Train on merged Chapman+Ningbo with SVT subclass reduced 50% in TRAIN ONLY.

Loads case1_merged_svtmask.npz. Baseline: train on full train split. Treatment:
randomly drop 50% of SVT (SNOMED 426761007) records from the TRAIN split only
(seed=42); val + test untouched. Both models evaluated on the SAME held-out
test split so the effect of thinning SVT at train time is isolated. Labels never
changed; test composition never changed.

Usage:
    python cross_eval/svt_reduce_train.py --drop_frac 0.5
"""

import os, sys, json, glob, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ptbxl_eval import ECG_CNN

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
NPZ        = r'data/case_study/case1_merged_svtmask.npz'
OUTDIR     = r'software/python/results/case_study'
GEORGIA    = r'data/georgia_by_class'


def load_georgia(root):
    X, y = [], []
    for label, c in enumerate(CLASS_NAMES):
        for f in sorted(glob.glob(os.path.join(root, c, '*.npy'))):
            X.append(np.load(f)); y.append(label)
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


def metrics(model, X, y, device, bs=256):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=bs, shuffle=False)
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
    return {'acc': float(acc), 'f1_macro': float(fm), 'auc_macro': float(auc),
            'p_macro': float(pm), 'r_macro': float(rm),
            'per_class': {CLASS_NAMES[c]: {'precision': float(p[c]), 'recall': float(r[c]),
                          'f1': float(f[c]), 'support': int(sup[c])} for c in range(4)},
            'confusion_matrix': cm.tolist(), 'n': int(len(y))}


def show(tag, m):
    print(f"\n=== {tag}  n={m['n']} ===")
    print(f"acc={m['acc']:.4f}  macro-P={m['p_macro']:.4f}  macro-R={m['r_macro']:.4f}  "
          f"macro-F1={m['f1_macro']:.4f}  macro-AUC={m['auc_macro']:.4f}")
    for c in CLASS_NAMES:
        pc = m['per_class'][c]
        print(f"  {c:<5} P={pc['precision']:.4f} R={pc['recall']:.4f} F1={pc['f1']:.4f} n={pc['support']}")
    print("  CM:", m['confusion_matrix'])


def train(Xtr, ytr, Xva, yva, device, epochs, lr, seed):
    torch.manual_seed(seed)
    model = ECG_CNN().to(device)
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                    batch_size=128, shuffle=False, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_acc, best_sd = 0.0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for xb, yb in va:
                pred = model(xb.to(device)).argmax(1).cpu()
                correct += (pred == yb).sum().item(); total += len(yb)
        va_acc = correct / total
        if va_acc > best_acc:
            best_acc = va_acc
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}/{epochs}  val_acc={va_acc:.4f}  (best={best_acc:.4f})")
    if best_sd: model.load_state_dict(best_sd)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drop_frac', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device={device}")

    d = np.load(NPZ)
    Xtr, ytr, svt_tr = d['X_train'], d['y_train'], d['svt_train']
    Xva, yva = d['X_val'], d['y_val']
    Xte, yte = d['X_test'], d['y_test']
    print(f"[INFO] train={len(ytr)} val={len(yva)} test={len(yte)}")
    print(f"[INFO] SVT in train={int(svt_tr.sum())}  test={int(d['svt_test'].sum())}")
    print(f"[INFO] test dist: {dict(zip(CLASS_NAMES, np.bincount(yte, minlength=4)))}")

    Xg, yg = load_georgia(GEORGIA)
    print(f"[INFO] Georgia: {dict(zip(CLASS_NAMES, np.bincount(yg, minlength=4)))}")

    res = {}

    # ── baseline: full train ────────────────────────────────────────────
    print("\n[INFO] === BASELINE: train on FULL merged (all SVT) ===")
    m_base = train(Xtr, ytr, Xva, yva, device, args.epochs, args.lr, args.seed)
    res['baseline_fullSVT'] = metrics(m_base, Xte, yte, device)
    show("BASELINE (full SVT) -> test split", res['baseline_fullSVT'])
    res['baseline_fullSVT_georgia'] = metrics(m_base, Xg, yg, device)
    show("BASELINE (full SVT) -> GEORGIA zero-shot", res['baseline_fullSVT_georgia'])
    torch.save(m_base.state_dict(), os.path.join(OUTDIR, 'svt_baseline_model.pth'))

    # ── treatment: drop drop_frac of SVT from TRAIN only ────────────────
    rng = np.random.default_rng(args.seed)
    svt_idx = np.where(svt_tr == 1)[0]
    n_drop = int(round(args.drop_frac * len(svt_idx)))
    drop = set(rng.choice(svt_idx, size=n_drop, replace=False).tolist())
    keep = np.array([i for i in range(len(ytr)) if i not in drop])
    Xtr2, ytr2 = Xtr[keep], ytr[keep]
    print(f"\n[INFO] === TREATMENT: dropped {n_drop}/{len(svt_idx)} SVT from train "
          f"({args.drop_frac:.0%}) -> train n={len(ytr2)} ===")
    m_red = train(Xtr2, ytr2, Xva, yva, device, args.epochs, args.lr, args.seed)
    res[f'reduced_SVT_{args.drop_frac}'] = metrics(m_red, Xte, yte, device)
    show(f"REDUCED SVT {args.drop_frac:.0%} -> test split", res[f'reduced_SVT_{args.drop_frac}'])
    res[f'reduced_SVT_{args.drop_frac}_georgia'] = metrics(m_red, Xg, yg, device)
    show(f"REDUCED SVT {args.drop_frac:.0%} -> GEORGIA zero-shot",
         res[f'reduced_SVT_{args.drop_frac}_georgia'])
    torch.save(m_red.state_dict(), os.path.join(OUTDIR, 'svt_reduced_model.pth'))

    res['meta'] = {'svt_in_train': int(svt_tr.sum()), 'svt_dropped': n_drop,
                   'drop_frac': args.drop_frac, 'svt_in_test': int(d['svt_test'].sum())}
    with open(os.path.join(OUTDIR, 'svt_reduce_train.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\n[INFO] saved: svt_reduce_train.json")


if __name__ == '__main__':
    main()
