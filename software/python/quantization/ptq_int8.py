"""Power-of-2 PTQ (A0) — Post-Training Quantization baseline for Table 4 (C1).

This is the QAT-vs-PTQ comparison: SAME power-of-2 INT8 scheme as A2
(qat_int8.py), SAME bit-exact int8_forward, but WITHOUT fake-quant training.

PTQ pipeline:
  1. Load float pruned model (best_model_pruned.pth) — already trained in float32.
  2. Copy weights into the QAT module shell (NO training, NO fake-quant fine-tune).
  3. convert_to_int8: calibrate power-of-2 shifts from the train set (same as A2).
  4. evaluate_int8: bit-exact INT8 forward (== RTL path).

Purpose: quantify why QAT is necessary — PTQ on the heavily-pruned model is
expected to collapse (PROJECT.md notes ~22%), because Conv1-3 have no ReLU
(keep negative activations) and pruning leaves the model quant-sensitive.

Usage:
    cd software/python
    python quantization/ptq_int8.py \\
        --checkpoint ./results/best_model_pruned.pth \\
        --output_dir ./results/ablation_quant/a0_ptq_p2 \\
        --data_dir   ../../data/Chapman
"""

import os
import sys
import argparse
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from prune_finetune import ECG_1DCNN_Pruned
from model.model import ECG_1DCNN
from utils.dataset import get_dataloaders, CLASS_NAMES
from utils.evaluate import compute_metrics, print_classification_report

from quantization.qat_int8 import (
    ECG_1DCNN_QAT, build_qat_model, convert_to_int8, evaluate_int8,
)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] PTQ (A0): power-of-2 INT8, NO fake-quant training")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    is_pruned = 'c1_out' in ckpt

    if is_pruned:
        print(f"[INFO] Pruned model (c1={ckpt['c1_out']}, c2={ckpt['c2_out']}, "
              f"c3={ckpt['c3_out']}, c4={ckpt['c4_out']})")
        base_model = ECG_1DCNN_Pruned(
            c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
            c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
        )
    else:
        base_model = ECG_1DCNN(num_classes=4)
    base_model.load_state_dict(ckpt['model_state_dict'])
    base_model = base_model.to(device)
    base_model.eval()

    train_loader, _, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=2
    )

    # ---- Build QAT shell, copy float weights, NO training ----
    qat_model = build_qat_model(base_model).to(device)
    qat_model.eval()

    # ---- Power-of-2 INT8 conversion (calibrate from train set) ----
    print(f"\n{'='*60}")
    print(f"  Power-of-2 PTQ Conversion (calibrate shifts, no fine-tune)")
    print(f"{'='*60}")
    w_int8, b_int8, w_shift, nb, input_shift = convert_to_int8(
        qat_model, train_loader, device, n_cal_batches=20
    )
    print(f"\n  input_shift = {input_shift}")
    print(f"  w_shift = {w_shift}")
    print(f"  nb      = {nb}")

    # ---- Bit-exact INT8 eval ----
    print(f"\n{'='*60}")
    print(f"  Evaluation (bit-exact int8_forward)")
    print(f"{'='*60}")
    acc, preds, labels = evaluate_int8(
        qat_model, test_loader, w_int8, b_int8, w_shift, nb, input_shift, device
    )
    metrics = compute_metrics(preds, labels, CLASS_NAMES)
    print(f"\n  PTQ INT8 accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-macro          : {metrics['f1_macro']:.4f}")
    print_classification_report(metrics)

    results = {
        'variant': 'A0_ptq_p2',
        'int8_acc': float(acc),
        'fq_acc': float(acc),
        'acc_drop_pct': 0.0,
        'dsp_extra_rescale': 0,
        'f1_macro': float(metrics['f1_macro']),
        'per_class_f1': {k: float(v['f1']) for k, v in metrics['per_class'].items()},
        'nb': nb,
        'w_shift': w_shift,
        'input_shift_bits': int(input_shift),
        'note': ('Power-of-2 PTQ (no fake-quant training) on float pruned model. '
                 'Same INT8 scheme + bit-exact path as A2 — isolates the effect of '
                 'QAT training vs post-training calibration.'),
    }
    out_path = os.path.join(args.output_dir, 'results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    print(f"\n{'='*60}")
    print(f"  Table 4 row — A0 PTQ (power-of-2, no QAT):")
    print(f"  Acc (INT8 sim) : {acc*100:.2f}%")
    print(f"  F1-macro       : {metrics['f1_macro']:.4f}")
    print(f"  vs A2 QAT      : QAT trains away the PTQ accuracy collapse")
    print(f"{'='*60}")


def main():
    p = argparse.ArgumentParser(description='Power-of-2 PTQ (A0) baseline for Table 4')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Float32 pruned checkpoint (best_model_pruned.pth)')
    p.add_argument('--output_dir', type=str, default='./results/ablation_quant/a0_ptq_p2')
    p.add_argument('--data_dir',   type=str, default='../../data/Chapman')
    p.add_argument('--batch_size', type=int, default=128)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
