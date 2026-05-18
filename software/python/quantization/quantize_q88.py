"""Q8.8 Quantization (fixed-point, scale=1/256)

Format: Q8.8 (16-bit signed fixed-point)
  - Integer part: 8 bits, fractional part: 8 bits
  - Scale: fixed 1/256
  - Range: [-128.0, +127.99609375]
  - Precision: 1/256 ≈ 0.0039

Hardware pipeline:
  Input(Q8.8) → Conv(int16×int16→int32) → >>8 → clip → MaxPool → ...
  Rescaling: acc >> 8 (fixed, no calibration needed)

Usage:
    python quantize_q88.py --checkpoint ./results/best_model.pth \\
                           --output_dir ./results/q88
"""

import os
import sys
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from model.model import ECG_1DCNN, ECG_1DCNN_Q88
from prune_finetune import ECG_1DCNN_Pruned


def quantize_checkpoint(checkpoint_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'c1_out' in ckpt:
        print(f"[INFO] Detected pruned model "
              f"(c1={ckpt['c1_out']}, c2={ckpt['c2_out']}, "
              f"c3={ckpt['c3_out']}, c4={ckpt['c4_out']})")
        float_model = ECG_1DCNN_Pruned(
            c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
            c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
        )
    else:
        float_model = ECG_1DCNN(num_classes=4)

    float_model.load_state_dict(ckpt['model_state_dict'])
    float_model = float_model.to(device)

    max_w = max(p.data.abs().max().item() for p in float_model.parameters())
    print(f"[INFO] Max weight magnitude: {max_w:.4f}  "
          f"({'within Q8.8 range' if max_w < 128 else 'WARNING: exceeds Q8.8 range [-128, 128)'})")

    print("[INFO] Quantizing to Q8.8 (fixed scale=1/256, range [-128, 127.99])")
    q88_model = ECG_1DCNN_Q88(float_model)
    q88_model = q88_model.to(device)

    q88_ckpt = {
        'model_state_dict': q88_model.model.state_dict(),
        'quantization': 'Q8.8',
        'q88_scale': 256,
        'bit_width': 16,
        'original_epoch': ckpt.get('epoch', '?'),
        'original_val_acc': ckpt.get('val_acc', None),
    }
    for key in ('c1_out', 'c2_out', 'c3_out', 'c4_out'):
        if key in ckpt:
            q88_ckpt[key] = ckpt[key]

    output_path = os.path.join(output_dir, 'model_q88.pth')
    torch.save(q88_ckpt, output_path)
    print(f"[DONE] Saved Q8.8 quantized model: {output_path}")


def main():
    p = argparse.ArgumentParser(
        description='Quantize ECG_1DCNN to Q8.8 (16-bit signed fixed-point)'
    )
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to trained checkpoint (.pth)')
    p.add_argument('--output_dir', type=str, default='./results/q88',
                   help='Output directory for quantized model')
    args = p.parse_args()
    quantize_checkpoint(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
