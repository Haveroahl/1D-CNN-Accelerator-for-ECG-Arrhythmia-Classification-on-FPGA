"""Phase A' — Run all quantization ablation variants for Table 4 (C1).

Variants:
  A1: Float32 baseline (no quantization)
  A2: Power-of-2 QAT round-half-up (ours)  — qat_int8.py
  A3: General-scale INT8 QAT (round)        — qat_int8_general.py --rescale-mode round
  A4: Power-of-2 QAT floor truncation       — qat_int8_floor.py

Usage:
    cd software/python
    python run_ablation_quant.py \\
        --pruned_checkpoint ./results/best_model_pruned.pth \\
        --a2_checkpoint     ./results/qat_int8/model_qat_int8.pth \\
        --data_dir          D:/Thesis101/data/Chapman \\
        --output_dir        ./results/ablation_quant

    # Skip training for A2 (already trained), train A3+A4 from scratch:
    python run_ablation_quant.py \\
        --pruned_checkpoint ./results/best_model_pruned.pth \\
        --a2_checkpoint     ./results/qat_int8/model_qat_int8.pth \\
        --data_dir          D:/Thesis101/data/Chapman \\
        --output_dir        ./results/ablation_quant \\
        --skip_a1 --skip_a2 --epochs 30

Output:
    results/ablation_quant/
      a1_float32/results.json
      a2_p2_round/results.json
      a3_general_round/results.json
      a4_p2_floor/results.json
      table4.json      ← final Table 4 summary
      table4.txt       ← human-readable table
"""

import os
import sys
import argparse
import json
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.model import ECG_1DCNN
from prune_finetune import ECG_1DCNN_Pruned
from utils.dataset import get_dataloaders, CLASS_NAMES
from utils.evaluate import compute_metrics, print_classification_report


def eval_float32(ckpt_path, data_dir, output_dir, batch_size=128):
    """A1: Evaluate float32 pruned model — upper bound."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    is_pruned = 'c1_out' in ckpt

    if is_pruned:
        model = ECG_1DCNN_Pruned(
            c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
            c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
        )
    else:
        model = ECG_1DCNN(num_classes=4)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    _, _, test_loader = get_dataloaders(data_dir, batch_size=batch_size, num_workers=2)

    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            logits = model(x)
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(batch[1].numpy())

    preds  = np.array(preds)
    labels = np.array(labels)
    acc = (preds == labels).mean()
    metrics = compute_metrics(preds, labels, CLASS_NAMES)

    print(f"\n  [A1] Float32 accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print_classification_report(metrics)

    results = {
        'variant': 'A1_float32',
        'int8_acc': float(acc),
        'fq_acc': float(acc),
        'acc_drop_pct': 0.0,
        'dsp_extra_rescale': 'N/A',
        'f1_macro': float(metrics['f1_macro']),
        'per_class_f1': {k: float(v['f1']) for k, v in metrics['per_class'].items()},
        'note': 'Float32 upper bound — no quantization',
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def load_a2_results(a2_checkpoint, data_dir, output_dir, batch_size=128):
    """Load or eval A2 from existing QAT-INT8 checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    existing = os.path.join(output_dir, 'results.json')
    if os.path.exists(existing):
        with open(existing) as f:
            return json.load(f)

    # Re-evaluate using qat_int8 INT8 simulation
    from quantization.qat_int8 import (
        ECG_1DCNN_QAT, int8_forward, evaluate_int8
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(a2_checkpoint, map_location=device, weights_only=False)
    is_pruned = 'c1_out' in ckpt

    if is_pruned:
        qat_model = ECG_1DCNN_QAT(
            c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
            c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
        ).to(device)
    else:
        qat_model = ECG_1DCNN_QAT().to(device)
    qat_model.load_state_dict(ckpt['model_state_dict'])
    qat_model.eval()

    w_int8   = {k: np.array(v, dtype=np.int8)    for k, v in ckpt['w_int8'].items()}
    b_int8   = {k: np.array(v, dtype=np.float64) for k, v in ckpt['b_int8'].items()}
    w_shift  = ckpt['w_shift']
    nb       = ckpt['nb']
    input_shift = ckpt['input_shift_bits']

    _, _, test_loader = get_dataloaders(data_dir, batch_size=batch_size, num_workers=2)

    acc, preds, labels = evaluate_int8(
        qat_model, test_loader, w_int8, b_int8, w_shift, nb, input_shift, device
    )
    metrics = compute_metrics(preds, labels, CLASS_NAMES)

    print(f"\n  [A2] Power-of-2 round INT8 accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print_classification_report(metrics)

    results = {
        'variant': 'A2_p2_round',
        'int8_acc': float(acc),
        'fq_acc': float(acc),
        'acc_drop_pct': 0.0,
        'dsp_extra_rescale': 0,
        'f1_macro': float(metrics['f1_macro']),
        'per_class_f1': {k: float(v['f1']) for k, v in metrics['per_class'].items()},
        'nb': nb,
        'w_shift': w_shift,
        'input_shift_bits': int(input_shift),
        'note': 'Power-of-2 QAT round-half-up (ours) — hardware: shift + add, 0 DSP extra',
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


def run_subprocess(script, extra_args, label):
    """Run a quantization script as subprocess."""
    cmd = [sys.executable, script] + extra_args
    print(f"\n{'='*60}")
    print(f"  Running {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {ret.returncode}). Aborting ablation run so "
            f"Table 4 is not built from incomplete results."
        )


def build_table4(results_dir, variants):
    """Collect all results.json and build Table 4."""
    rows = []
    a1_acc = None

    for v in variants:
        path = os.path.join(results_dir, v['dir'], 'results.json')
        if not os.path.exists(path):
            print(f"  [WARNING] Missing results: {path}")
            continue
        with open(path) as f:
            r = json.load(f)
        rows.append({
            'variant': v['label'],
            'int8_acc': r.get('int8_acc', 0),
            'f1_macro': r.get('f1_macro', 0),
            'dsp_extra': r.get('dsp_extra_rescale', 'N/A'),
            'per_class_f1': r.get('per_class_f1', {}),
        })
        if v['label'].startswith('A1'):
            a1_acc = r.get('int8_acc', 0)

    # Compute acc drop vs A1
    for row in rows:
        if a1_acc and a1_acc > 0:
            row['acc_drop_vs_a1'] = (a1_acc - row['int8_acc']) * 100
        else:
            row['acc_drop_vs_a1'] = None

    return rows


def print_table4(rows):
    header = f"{'Variant':<30} {'Acc':>7} {'F1':>7} {'ΔAcc(A1)':>10} {'DSP+extra':>10}"
    print(f"\n{'='*70}")
    print("  TABLE 4 — Quantization Ablation (Phase A')")
    print(f"{'='*70}")
    print(header)
    print("-" * 70)
    for r in rows:
        drop_str = f"{r['acc_drop_vs_a1']:+.2f}%" if r['acc_drop_vs_a1'] is not None else "  N/A"
        dsp_str = str(r['dsp_extra']) if r['dsp_extra'] != 'N/A' else "N/A"
        print(f"  {r['variant']:<28} {r['int8_acc']*100:>6.2f}%"
              f" {r['f1_macro']:>7.4f} {drop_str:>10}  +{dsp_str} DSP18")
    print(f"{'='*70}")
    print("  DSP+extra: additional DSP18 for rescale vs A2 baseline")
    print("  ΔAcc(A1): accuracy drop vs float32 upper bound")


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    quant_dir  = os.path.join(script_dir, 'quantization')

    common_args = [
        '--checkpoint', args.pruned_checkpoint,
        '--data_dir', args.data_dir,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
    ]

    # ---- A1: Float32 ----
    a1_dir = os.path.join(args.output_dir, 'a1_float32')
    if not args.skip_a1:
        print(f"\n[Phase A'] Running A1 (float32 eval)...")
        eval_float32(args.pruned_checkpoint, args.data_dir, a1_dir, args.batch_size)
    else:
        print(f"\n[Phase A'] Skipping A1 (--skip_a1)")

    # ---- A0: Power-of-2 PTQ (no fake-quant training) ----
    a0_dir = os.path.join(args.output_dir, 'a0_ptq_p2')
    if not args.skip_a0:
        run_subprocess(
            os.path.join(quant_dir, 'ptq_int8.py'),
            ['--checkpoint', args.pruned_checkpoint,
             '--data_dir', args.data_dir,
             '--batch_size', str(args.batch_size),
             '--output_dir', a0_dir],
            'A0 power-of-2 PTQ (no QAT)'
        )
    else:
        print(f"\n[Phase A'] Skipping A0 (--skip_a0)")

    # ---- A2: Power-of-2 round (use existing checkpoint if provided) ----
    a2_dir = os.path.join(args.output_dir, 'a2_p2_round')
    if not args.skip_a2:
        if args.a2_checkpoint and os.path.exists(args.a2_checkpoint):
            print(f"\n[Phase A'] Loading A2 from {args.a2_checkpoint}...")
            load_a2_results(args.a2_checkpoint, args.data_dir, a2_dir, args.batch_size)
        else:
            run_subprocess(
                os.path.join(quant_dir, 'qat_int8.py'),
                common_args + ['--output_dir', a2_dir],
                'A2 power-of-2 round-half-up'
            )
    else:
        print(f"\n[Phase A'] Skipping A2 (--skip_a2)")

    # ---- A3: General-scale round ----
    a3_dir = os.path.join(args.output_dir, 'a3_general_round')
    if not args.skip_a3:
        run_subprocess(
            os.path.join(quant_dir, 'qat_int8_general.py'),
            common_args + ['--output_dir', a3_dir, '--rescale-mode', 'round'],
            'A3 general-scale INT8 (round)'
        )
    else:
        print(f"\n[Phase A'] Skipping A3 (--skip_a3)")

    # ---- A4: Power-of-2 floor ----
    a4_dir = os.path.join(args.output_dir, 'a4_p2_floor')
    if not args.skip_a4:
        run_subprocess(
            os.path.join(quant_dir, 'qat_int8_floor.py'),
            common_args + ['--output_dir', a4_dir],
            'A4 power-of-2 floor truncation'
        )
    else:
        print(f"\n[Phase A'] Skipping A4 (--skip_a4)")

    # ---- Build Table 4 ----
    variants = [
        {'dir': 'a1_float32',      'label': 'A1 Float32 baseline'},
        {'dir': 'a0_ptq_p2',       'label': 'A0 P2-PTQ (no QAT)'},
        {'dir': 'a2_p2_round',     'label': 'A2 P2-QAT round-half-up (ours)'},
        {'dir': 'a3_general_round','label': 'A3 General-scale (round)'},
        {'dir': 'a4_p2_floor',     'label': 'A4 P2-QAT floor (no round)'},
    ]
    rows = build_table4(args.output_dir, variants)
    if rows:
        print_table4(rows)
        table4_path = os.path.join(args.output_dir, 'table4.json')
        with open(table4_path, 'w') as f:
            json.dump(rows, f, indent=2)

        # Human-readable table
        txt_path = os.path.join(args.output_dir, 'table4.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("TABLE 4 - Quantization Ablation for C1 (Phase A')\n")
            f.write("="*70 + "\n")
            f.write(f"{'Variant':<30} {'Acc%':>7} {'F1':>7} {'dAcc(A1)':>10} {'DSP+extra':>10}\n")
            f.write("-"*70 + "\n")
            for r in rows:
                drop_str = f"{r['acc_drop_vs_a1']:+.2f}%" if r['acc_drop_vs_a1'] is not None else "N/A"
                dsp_str = str(r['dsp_extra'])
                f.write(f"  {r['variant']:<28} {r['int8_acc']*100:>6.2f}%"
                        f" {r['f1_macro']:>7.4f} {drop_str:>10}  +{dsp_str}\n")
            f.write("="*70 + "\n")
        print(f"\n  Table 4 saved: {table4_path}")
        print(f"  Text table  : {txt_path}")


def main():
    p = argparse.ArgumentParser(description="Phase A' — run all quantization ablation variants")
    p.add_argument('--pruned_checkpoint', type=str, required=True,
                   help='Float32 pruned model checkpoint')
    p.add_argument('--a2_checkpoint',     type=str, default='',
                   help='Existing A2 QAT-INT8 checkpoint (skip A2 training if provided)')
    p.add_argument('--data_dir',          type=str, default='../../data/Chapman')
    p.add_argument('--output_dir',        type=str, default='./results/ablation_quant')
    p.add_argument('--epochs',            type=int, default=50)
    p.add_argument('--batch_size',        type=int, default=128)
    p.add_argument('--skip_a0',           action='store_true')
    p.add_argument('--skip_a1',           action='store_true')
    p.add_argument('--skip_a2',           action='store_true')
    p.add_argument('--skip_a3',           action='store_true')
    p.add_argument('--skip_a4',           action='store_true')
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
