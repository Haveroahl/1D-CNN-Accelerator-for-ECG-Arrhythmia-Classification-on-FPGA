"""List Georgia AFIB records by prediction outcome (TH1 model, zero-shot).

Pure diagnostic report: runs the TH1 (Chapman+Ningbo float32) model over every
Georgia AFIB record and lists which were predicted AFIB (correct) vs misclassified
(and into which class). NOTHING is changed — no labels touched, no records dropped.

Usage:
    python cross_eval/georgia_afib_list.py
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

    files = sorted(glob.glob(os.path.join(ROOT, 'AFIB', '*.npy')))
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    X = np.asarray([np.load(f) for f in files], np.float32)

    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device)).cpu().numpy()
    pred = logits.argmax(1)

    correct = [names[i] for i in range(len(names)) if pred[i] == 0]
    wrong   = {c: [names[i] for i in range(len(names)) if pred[i] == c]
               for c in (1, 2, 3)}

    print(f"Georgia AFIB total: {len(names)}")
    print(f"  predicted AFIB (correct): {len(correct)}")
    for c in (1, 2, 3):
        print(f"  predicted {CLASS_NAMES[c]:<4} (wrong):   {len(wrong[c])}")

    print(f"\n=== AFIB correctly predicted ({len(correct)}) ===")
    print(", ".join(correct))

    for c in (1, 2, 3):
        if wrong[c]:
            print(f"\n=== AFIB -> {CLASS_NAMES[c]} (misclassified, {len(wrong[c])}) ===")
            print(", ".join(wrong[c]))

    with open(os.path.join(OUTDIR, 'georgia_afib_list.json'), 'w') as f:
        json.dump({'correct_AFIB': correct,
                   'wrong_GSVT': wrong[1], 'wrong_SB': wrong[2], 'wrong_SR': wrong[3]},
                  f, indent=2)
    print(f"\n[INFO] saved: georgia_afib_list.json")


if __name__ == '__main__':
    main()
