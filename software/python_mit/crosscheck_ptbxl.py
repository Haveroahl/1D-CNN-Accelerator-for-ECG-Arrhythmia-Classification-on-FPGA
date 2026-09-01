"""
Cross-check the tiny MIT-BIH 5-symbol model on PTB-XL (zero-shot)
================================================================
Loads a model trained by train_hier.py (scheme=intra, {N,L,R,A,V}) and runs it
on PTB-XL beats relabelled into the same 5-symbol space (utils/ptbxl_5sym.py).

N/L/R carry honest per-beat labels (whole-record diagnostics); A/V are weak
record-level labels — read their F1 with that caveat. The point is to see how
much MIT-BIH-learned beat morphology + rhythm transfers to a different cohort.

The RR features are standardized with the SAME stats the model trained on. We
recompute those stats from the MIT-BIH intra TRAIN split (same seed as training)
since train_hier.py does not store them in the checkpoint.

Usage:
  python crosscheck_ptbxl.py \
      --ckpt results/hier_5sym_intra/best_model.pth \
      --mit_dir ../../data/mitdb --ptbxl_dir ../../data/ptbxl \
      --max_per_class 150
"""

import os, sys, argparse, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from dataset_hier import MITBIH5SymDataset, CLASS5, N5
from utils.ptbxl_5sym import load_ptbxl_5sym
from model.model_hier import build_model


def confusion(y_true, y_pred, K):
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def report(cm, names, title):
    K = cm.shape[0]
    acc = np.trace(cm) / max(cm.sum(), 1)
    print(f"\n  {title}")
    print(f"  acc={acc:.4f}")
    print(f"  {'cls':<6}{'P':>8}{'R':>8}{'F1':>8}{'sup':>8}")
    f1s = []
    for i in range(K):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        r = tp / (tp + fn) if (tp + fn) else 0.0
        p = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if cm[i, :].sum() > 0:
            f1s.append(f1)
        print(f"  {names[i]:<6}{p:>8.3f}{r:>8.3f}{f1:>8.3f}{int(cm[i,:].sum()):>8d}")
    f1m = float(np.mean(f1s)) if f1s else 0.0
    print(f"  F1-macro (present classes) = {f1m:.4f}")
    print(f"  confusion (rows=true, cols=pred):")
    print("        " + "".join(f"{n:>7}" for n in names))
    for i in range(K):
        print(f"  {names[i]:<5}" + "".join(f"{int(cm[i,j]):>7d}" for j in range(K)))
    return float(acc), f1m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/hier_5sym_intra/best_model.pth')
    ap.add_argument('--mit_dir', default='../../data/mitdb')
    ap.add_argument('--ptbxl_dir', default='../../data/ptbxl')
    ap.add_argument('--max_per_class', type=int, default=150,
                    help='cap PTB-XL records per symbol (runtime control)')
    ap.add_argument('--n_rr', type=int, default=8)
    ap.add_argument('--rr_hidden', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='results/hier_5sym_intra/ptbxl_crosscheck.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) recover the MIT-BIH intra-train RR stats the model trained on
    print("[1/4] recovering MIT-BIH train RR-stats (same seed as training) ...")
    tr = MITBIH5SymDataset(args.mit_dir, 'train', seed=args.seed, scheme='intra')
    rr_stats = tr.rr_stats

    # 2) build PTB-XL beats in the 5-symbol space
    print("[2/4] extracting PTB-XL beats (resample 500->360, Pan-Tompkins) ...")
    X, R, y = load_ptbxl_5sym(args.ptbxl_dir, rr_stats,
                              max_per_class=args.max_per_class, seed=args.seed)

    # 3) load model
    print("[3/4] loading model ...")
    model = build_model(n_rr=args.n_rr, rr_hidden=args.rr_hidden).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    # 4) inference (use the 5-class head)
    print("[4/4] inference ...")
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            xb = torch.from_numpy(X[i:i+1024]).to(device)
            rb = torch.from_numpy(R[i:i+1024]).to(device)
            _, o5 = model(xb, rb)
            preds.append(o5.argmax(1).cpu().numpy())
    preds = np.concatenate(preds)

    cm5 = confusion(y, preds, N5)
    acc5, f1m5 = report(cm5, CLASS5,
                        "PTB-XL zero-shot 5-class {N,L,R,A,V}  (A,V = weak record-label)")

    # clean view: only the honestly-labelled N/L/R subset (drop A,V weak beats)
    mask = np.isin(y, [0, 1, 2])
    cm3 = confusion(y[mask], preds[mask], N5)[:3, :3]
    acc3, f1m3 = report(cm3, CLASS5[:3],
                        "PTB-XL zero-shot N/L/R only (honest per-beat labels)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(
        five_class=dict(acc=acc5, f1_macro=f1m5, confusion=cm5.tolist(),
                        note="A,V are weak record-level labels"),
        nlr_only=dict(acc=acc3, f1_macro=f1m3, confusion=cm3.tolist(),
                      note="honest per-beat labels (whole-record diagnostics)"),
        n_beats=int(len(y)),
    ), open(args.out, 'w'), indent=2)
    print(f"\n[DONE] saved {args.out}")


if __name__ == "__main__":
    main()
