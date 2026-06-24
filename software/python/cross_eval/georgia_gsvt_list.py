"""List Georgia GSVT records by prediction outcome (TH1 model, zero-shot).

Pure diagnostic report: runs the TH1 (Chapman+Ningbo float32) model over every
Georgia GSVT record and lists which were predicted GSVT (correct) vs misclassified
(and into which class). NOTHING is changed — no labels touched, no records dropped.

Usage:
    python cross_eval/georgia_gsvt_list.py
"""

import os, sys, glob, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ptbxl_eval import ECG_CNN

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
TH1_MODEL = r'software/python/results/case_study/case1_model_float32.pth'
ROOT      = r'data/georgia_by_class'
OUTDIR    = r'software/python/results/cross_eval'


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ECG_CNN().to(device)
    model.load_state_dict(torch.load(TH1_MODEL, map_location=device))
    model.eval()

    files = sorted(glob.glob(os.path.join(ROOT, 'GSVT', '*.npy')))
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    X = np.asarray([np.load(f) for f in files], np.float32)

    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device)).cpu().numpy()
    pred = logits.argmax(1)

    n_true = int((pred == 1).sum())
    print(f"Georgia GSVT total: {len(names)}   True(correct)={n_true}   False(wrong)={len(names)-n_true}\n")

    # full per-record True/False table (True = predicted GSVT correctly)
    rows = []
    print(f"{'record':<10}{'result':<8}pred_as")
    print("-" * 30)
    for i in range(len(names)):
        ok = pred[i] == 1
        rows.append({'record': names[i], 'true': bool(ok),
                     'predicted_as': CLASS_NAMES[pred[i]]})
        print(f"{names[i]:<10}{str(ok):<8}{CLASS_NAMES[pred[i]]}")

    csv_path = os.path.join(OUTDIR, 'georgia_gsvt_truefalse.csv')
    with open(csv_path, 'w') as f:
        f.write("record,true,predicted_as\n")
        for r in rows:
            f.write(f"{r['record']},{r['true']},{r['predicted_as']}\n")
    with open(os.path.join(OUTDIR, 'georgia_gsvt_list.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\n[INFO] saved: {csv_path} + georgia_gsvt_list.json")


if __name__ == '__main__':
    main()
