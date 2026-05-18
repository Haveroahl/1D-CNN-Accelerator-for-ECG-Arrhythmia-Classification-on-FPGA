"""Evaluate quantized models against float baseline.

Compares accuracy of different quantization and pruning formats:
- Float32 (baseline)
- Q8.8 (16-bit signed fixed-point)
- Q4.4 (8-bit signed fixed-point)
- INT8 (post-training quantization)
- QAT-INT8 (quantization-aware training)
- Pruned Float32 (after channel pruning)
- Pruned INT8 (pruned + INT8 quantized)

Usage:
    # Full model comparison
    python evaluate_quantized.py \\
        --float_ckpt ./results/best_model.pth \\
        --q88_ckpt   ./results/q88/model_q88.pth \\
        --q44_ckpt   ./results/q44/model_q44.pth \\
        --int8_ckpt  ./results/int8/model_int8.pth \\
        --qat_ckpt   ./results/qat_int8/model_qat_int8.pth

    # Option 2: Pruned + quantized comparison
    python evaluate_quantized.py \\
        --float_ckpt       ./results/best_model.pth \\
        --pruned_ckpt      ./results/best_model_pruned.pth \\
        --pruned_int8_ckpt ./results/int8_pruned/model_int8.pth
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from model.model import ECG_1DCNN
from prune_finetune import ECG_1DCNN_Pruned
from quantize_int8 import ECG_1DCNN_INT8_fphys
from utils.dataset import get_dataloaders, CLASS_NAMES
from utils.evaluate import evaluate_model, compute_metrics, print_classification_report


# ── Model loader ──────────────────────────────────────────────

def load_model(ckpt_path: str, device):
    """Load model from checkpoint.

    Supports:
      - float32 baseline (ECG_1DCNN or ECG_1DCNN_Pruned)
      - INT8-fphys quantized (ECG_1DCNN_INT8_fphys) — runs fully-integer forward
      - Legacy INT8 checkpoints — runs float forward with quantized weights

    Returns:
        model:    loaded model in eval mode
        ckpt:     raw checkpoint dict
        n_params: parameter count
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    quant = ckpt.get('quantization', '')

    # ── INT8-fphys: restore fully-integer model ──────────────────
    if quant == 'INT8-fphys':
        if 'c1_out' in ckpt:
            base = ECG_1DCNN_Pruned(
                c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
                c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
            )
        else:
            base = ECG_1DCNN(num_classes=4)

        model = ECG_1DCNN_INT8_fphys(base, from_checkpoint=True)
        model.model.load_state_dict(ckpt['model_state_dict'])
        model.w_scales = ckpt.get('w_scales', {})
        model.input_shift_bits = ckpt.get('input_shift_bits', 0)
        model.input_scale = ckpt.get('input_scale', 1.0)
        model.nb = ckpt.get('nb', {})
        model = model.to(device)
        model.eval()
        n_params = sum(p.numel() for p in model.model.parameters()
                       if p.requires_grad)
        return model, ckpt, n_params

    # ── Float or legacy INT8: plain model ───────────────────────
    if 'c1_out' in ckpt:
        model = ECG_1DCNN_Pruned(
            c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
            c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
        )
    else:
        model = ECG_1DCNN(num_classes=4)

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, ckpt, n_params


# ── Per-model evaluation ──────────────────────────────────────

def evaluate_and_report(label: str, model, test_loader, device) -> dict:
    """Run test evaluation and print per-class report."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    preds, labels = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(preds, labels, CLASS_NAMES)
    print_classification_report(metrics)
    return metrics


# ── Memory estimate ───────────────────────────────────────────

_BIT_WIDTH = {
    'float32': 32,
    'q88':     16,
    'q44':      8,
    'int8':     8,
    'qat':      8,
    'pruned_float32': 32,
    'pruned_int8':     8,
}


def memory_kb(n_params: int, bits: int) -> float:
    return round(n_params * bits / 8 / 1024, 2)


# ── Main ──────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Evaluate and compare quantized / pruned model variants'
    )
    p.add_argument('--float_ckpt',       type=str, required=True,
                   help='Float32 baseline checkpoint')
    p.add_argument('--q88_ckpt',         type=str, default=None,
                   help='Q8.8 quantized checkpoint (optional)')
    p.add_argument('--q44_ckpt',         type=str, default=None,
                   help='Q4.4 quantized checkpoint (optional)')
    p.add_argument('--int8_ckpt',        type=str, default=None,
                   help='INT8 (PTQ) checkpoint (optional)')
    p.add_argument('--qat_ckpt',         type=str, default=None,
                   help='QAT-INT8 checkpoint (optional)')
    p.add_argument('--pruned_ckpt',      type=str, default=None,
                   help='Pruned float32 checkpoint from prune_finetune.py (optional)')
    p.add_argument('--pruned_int8_ckpt', type=str, default=None,
                   help='Pruned + INT8 quantized checkpoint (optional)')
    p.add_argument('--data_dir',   type=str,
                   default='/home/duc/Thesis/data/Chapman')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers',type=int, default=2)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}\n")

    _, _, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # (key, label, ckpt_path, bit_key)
    configs = [
        ('float32',       'Float32 (Baseline)',          args.float_ckpt,       'float32'),
        ('q88',           'Q8.8 (16-bit fixed-point)',   args.q88_ckpt,         'q88'),
        ('q44',           'Q4.4 (8-bit fixed-point)',    args.q44_ckpt,         'q44'),
        ('int8',          'INT8 (PTQ)',                  args.int8_ckpt,        'int8'),
        ('qat',           'INT8 (QAT)',                  args.qat_ckpt,         'qat'),
        ('pruned_float32','Pruned Float32',              args.pruned_ckpt,      'pruned_float32'),
        ('pruned_int8',   'Pruned INT8 (PTQ)',           args.pruned_int8_ckpt, 'pruned_int8'),
    ]

    results   = {}   # key → metrics dict
    meta      = {}   # key → {n_params, mem_kb, label}
    base_acc  = None

    for key, label, ckpt_path, bit_key in configs:
        if not ckpt_path:
            continue

        print(f"[INFO] Loading: {label}  ({ckpt_path})")
        model, ckpt, n_params = load_model(ckpt_path, device)
        bits = _BIT_WIDTH[bit_key]
        mem  = memory_kb(n_params, bits)

        metrics = evaluate_and_report(label, model, test_loader, device)
        results[key] = metrics
        meta[key]    = {'n_params': n_params, 'mem_kb': mem, 'label': label}

        if key == 'float32':
            base_acc = metrics['accuracy']

    # ── Summary table ─────────────────────────────────────────
    if not results:
        print("[WARNING] No checkpoints evaluated.")
        return

    if base_acc is None:
        base_acc = next(iter(results.values()))['accuracy']

    print(f"\n{'='*72}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*72}")
    header = f"  {'Method':<26} {'Accuracy':>9}  {'ΔAcc':>7}  {'Params':>7}  {'Memory':>8}  {'Status'}"
    print(header)
    print(f"  {'-'*68}")

    for key, label, _, _ in configs:
        if key not in results:
            continue
        acc      = results[key]['accuracy']
        delta    = (acc - base_acc) * 100
        n_params = meta[key]['n_params']
        mem      = meta[key]['mem_kb']
        if key == 'float32':
            status = '— baseline'
        elif abs(delta) < 0.3:
            status = '✓ no loss'
        elif delta >= -1.0:
            status = '✓ acceptable'
        else:
            status = '⚠ degraded'
        print(f"  {label:<26} {acc:>8.2%}  {delta:>+6.2f}%  {n_params:>7}  {mem:>6.1f} KB  {status}")

    print(f"  {'-'*68}")

    # Show compression vs float32 baseline
    base_params = meta['float32']['n_params']
    base_mem    = meta['float32']['mem_kb']
    print(f"\n  {'Compression vs float32 baseline':}")
    for key, label, _, _ in configs:
        if key == 'float32' or key not in meta:
            continue
        comp = base_mem / meta[key]['mem_kb'] if meta[key]['mem_kb'] > 0 else 0
        print(f"    {label:<26}  {comp:.1f}× memory reduction")

    print(f"{'='*72}")


if __name__ == "__main__":
    main()
