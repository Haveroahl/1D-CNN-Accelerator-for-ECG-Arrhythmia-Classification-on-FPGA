"""TH2 train+eval — train float32 on PTB-XL (SR subsampled), test Chapman+Ningbo.

Train: data/case_study/case2_ptbxl_sr<N>.npz (train/val).
In-dist test: PTB-XL test split.
External test: merged Chapman+Ningbo (X_all/y_all/src_all from case1_merged.npz),
  reported combined + per-source (Chapman-half src=0 / Ningbo-half src=1).

Usage:
    python cross_eval/case2_train.py
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ptbxl_eval import ECG_CNN
from case1_train import metrics, show, train     # reuse identical helpers

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
MERGED = r'data/case_study/case1_merged.npz'
OUTDIR = r'software/python/results/case_study'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep_sr', type=int, default=2567)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    print(f"[INFO] device={device}")

    npz = f'data/case_study/case2_ptbxl_sr{args.keep_sr}.npz'
    d = np.load(npz)
    bs = 128
    tr = DataLoader(TensorDataset(torch.from_numpy(d['X_train']), torch.from_numpy(d['y_train'])),
                    batch_size=bs, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(d['X_val']), torch.from_numpy(d['y_val'])),
                    batch_size=bs, shuffle=False)

    model = ECG_CNN().to(device)
    print(f"[INFO] training float32 on PTB-XL (SR={args.keep_sr}) ...")
    model = train(model, tr, va, device, args.epochs, args.lr)

    res = {}
    res['ptbxl_indist_test'] = metrics(model, d['X_test'], d['y_test'], device)
    show("IN-DIST test (PTB-XL held-out)", res['ptbxl_indist_test'])

    # external = merged Chapman+Ningbo full
    m = np.load(MERGED)
    Xa, ya, sa = m['X_all'], m['y_all'], m['src_all']
    res['external_combined'] = metrics(model, Xa, ya, device)
    show("EXTERNAL combined (Chapman+Ningbo merged)", res['external_combined'])

    res['external_chapman_half'] = metrics(model, Xa[sa == 0], ya[sa == 0], device)
    show("EXTERNAL Chapman-half (src=0)", res['external_chapman_half'])

    res['external_ningbo_half'] = metrics(model, Xa[sa == 1], ya[sa == 1], device)
    show("EXTERNAL Ningbo-half (src=1)", res['external_ningbo_half'])

    torch.save(model.state_dict(), os.path.join(OUTDIR, f'case2_model_sr{args.keep_sr}.pth'))
    with open(os.path.join(OUTDIR, 'case2_result.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\n[INFO] saved model + case2_result.json to {OUTDIR}")


if __name__ == '__main__':
    main()
