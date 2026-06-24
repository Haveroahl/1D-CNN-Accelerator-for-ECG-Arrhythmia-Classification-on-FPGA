"""TH1 train+eval — train float32 on merged Chapman+Ningbo, test on PTB-XL.

Train: data/case_study/case1_merged.npz (train/val splits).
In-dist test: case1_merged test split.
External test: PTB-XL by-class tree, two configs:
    (a) full SR (19,952)
    (b) SR subsampled to --keep_sr_ptbxl (default 2567)  -> 5,776

Outputs results + per-source breakdown to results/case_study/case1_result.json.

Usage:
    python cross_eval/case1_train.py
"""

import os, sys, glob, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ptbxl_eval import ECG_CNN          # reuse model def (4,4,8,8)

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
PTBXL_ROOT  = r'data/ptbxl_by_class'
NPZ         = r'data/case_study/case1_merged.npz'
OUTDIR      = r'software/python/results/case_study'


def loaders_from_npz(npz, bs=128):
    d = np.load(npz)
    def mk(X, y, sh):
        return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                          batch_size=bs, shuffle=sh, num_workers=0)
    return (mk(d['X_train'], d['y_train'], True),
            mk(d['X_val'],   d['y_val'],   False),
            mk(d['X_test'],  d['y_test'],  False))


def load_ptbxl(root, keep_sr=None, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, c in enumerate(CLASS_NAMES):
        files = sorted(glob.glob(os.path.join(root, c, '*.npy')))
        if c == 'SR' and keep_sr is not None and keep_sr < len(files):
            idx = np.sort(rng.choice(len(files), size=keep_sr, replace=False))
            files = [files[i] for i in idx]
        for f in files:
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
    print(f"acc={m['acc']:.4f}  macro-F1={m['f1_macro']:.4f}  macro-AUC={m['auc_macro']:.4f}")
    for c in CLASS_NAMES:
        pc = m['per_class'][c]
        print(f"  {c:<5} P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} n={pc['support']}")
    print("  CM:", m['confusion_matrix'])


def train(model, tr, va, device, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_acc, best_sd = 0.0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        # val
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
    if best_sd:
        model.load_state_dict(best_sd)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--keep_sr_ptbxl', type=int, default=2567)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    print(f"[INFO] device={device}")

    tr, va, te = loaders_from_npz(NPZ)
    model = ECG_CNN().to(device)
    print("[INFO] training float32 on merged Chapman+Ningbo ...")
    model = train(model, tr, va, device, args.epochs, args.lr)

    res = {}
    # in-dist test split (need raw arrays for metrics())
    d = np.load(NPZ)
    res['indist_test'] = metrics(model, d['X_test'], d['y_test'], device)
    show("IN-DIST test (Chapman+Ningbo held-out)", res['indist_test'])

    # external PTB-XL full
    Xf, yf = load_ptbxl(PTBXL_ROOT, keep_sr=None)
    res['ptbxl_full'] = metrics(model, Xf, yf, device)
    show("EXTERNAL PTB-XL full SR", res['ptbxl_full'])

    # external PTB-XL SR=keep
    Xk, yk = load_ptbxl(PTBXL_ROOT, keep_sr=args.keep_sr_ptbxl, seed=args.seed)
    res[f'ptbxl_sr{args.keep_sr_ptbxl}'] = metrics(model, Xk, yk, device)
    show(f"EXTERNAL PTB-XL SR={args.keep_sr_ptbxl}", res[f'ptbxl_sr{args.keep_sr_ptbxl}'])

    torch.save(model.state_dict(), os.path.join(OUTDIR, 'case1_model_float32.pth'))
    with open(os.path.join(OUTDIR, 'case1_result.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\n[INFO] saved model + case1_result.json to {OUTDIR}")


if __name__ == '__main__':
    main()
