"""Ve lai hinh ROC voi nhan tieng Viet cho Chuong 4.

Dung lai dung ham int8_forward_bitexact cua int8_eval_batch.py nen so AUC
bao dam trung khop voi Bang/CM da co trong luan van (khong ve lai tu nguon khac).

  python plot_roc_vn.py --checkpoint results/ningba/qat_int8/model_qat_int8.pth \
      --npz ../../data/ningba_processed/ningbo_dataset_clip16.npz \
      --tag chapman --out results/figures_vn
  python plot_roc_vn.py --checkpoint results/ningba/qat_int8/model_qat_int8.pth \
      --byclass ../../data/georgia_by_class --tag georgia --out results/figures_vn
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from int8_eval_batch import int8_forward_bitexact, load_data, CLASS_NAMES
from quantization.qat_int8 import ECG_1DCNN_QAT


def plot_roc_vn(y, logits, out_png):
    """ROC mot-doi-con-lai cho 4 lop + duong trung binh macro.

    Giu nguyen cach tinh cua int8_eval_batch.plot_roc (softmax tren float32,
    macro = auc cua duong TPR trung binh) de so AUC trung khop tuyet doi voi
    cac bang da co trong luan van. Chi doi nhan sang tieng Viet.
    """
    prob = torch.softmax(torch.from_numpy(logits.astype(np.float32)), dim=1).numpy()
    yb = np.eye(4)[y]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))

    aucs = []
    all_fpr = np.unique(np.concatenate(
        [roc_curve(yb[:, i], prob[:, i])[0] for i in range(4)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(yb[:, i], prob[:, i])
        a = auc(fpr, tpr)
        aucs.append(a)
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        ax.plot(fpr, tpr, lw=1.6, label=f'{name} (AUC = {a:.3f})')
    mean_tpr /= 4
    macro = float(auc(all_fpr, mean_tpr))
    ax.plot(all_fpr, mean_tpr, 'k--', lw=2.0,
            label=f'Trung bình macro (AUC = {macro:.3f})')

    ax.plot([0, 1], [0, 1], ls=':', c='gray', lw=1.0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Tỉ lệ dương tính giả (FPR)')
    ax.set_ylabel('Tỉ lệ dương tính thật (TPR)')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return aucs, macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--npz')
    ap.add_argument('--byclass')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--clip', type=float, default=16.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    m = ECG_1DCNN_QAT(c1_out=ck['c1_out'], c2_out=ck['c2_out'],
                      c3_out=ck['c3_out'], c4_out=ck['c4_out'])
    m.load_state_dict(ck['model_state_dict'])
    m.eval()

    w8 = {k: np.array(v, dtype=np.int8) for k, v in ck['w_int8'].items()}
    b8 = {k: np.array(v, dtype=np.float32) for k, v in ck['b_int8'].items()}

    X, y = load_data(args)
    logits = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256])
            logits.append(int8_forward_bitexact(
                m, xb, w8, b8, ck['nb'], ck['w_shift'],
                ck['input_shift_bits']).numpy())
    logits = np.concatenate(logits)

    out_png = os.path.join(args.out, f'roc_int8_{args.tag}.png')
    aucs, macro = plot_roc_vn(y, logits, out_png)
    print(f'[OK] {out_png}')
    for n, a in zip(CLASS_NAMES, aucs):
        print(f'   {n:<5} AUC = {a:.4f}')
    print(f'   macro AUC = {macro:.4f}   (n = {len(y)})')


if __name__ == '__main__':
    main()
