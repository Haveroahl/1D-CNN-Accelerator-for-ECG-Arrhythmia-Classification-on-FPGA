"""Q4.4 Quantization with per-layer dynamic scaling

Format: Q4.4-style (8-bit signed, per-layer scaling)
Range: [-128, 127] mapped to [-8, 8] via per-layer scale
Precision: 1/16 = 0.0625

Usage:
    python quantize_q44.py --checkpoint ./results/best_model.pth \\
                           --output_dir ./results/q44
"""

import os
import sys
import argparse
import torch
import numpy as np
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from model.model import ECG_1DCNN


class ECG_1DCNN_Q44(torch.nn.Module):
    """Q4.4-style quantized ECG_1DCNN model with per-layer dynamic scaling."""

    def __init__(self, float_model: ECG_1DCNN):
        super().__init__()
        self.float_model = copy.deepcopy(float_model)
        self.scales = {}

        # Quantize each layer with its own scale
        with torch.no_grad():
            for name, param in self.float_model.named_parameters():
                if param.requires_grad:
                    w_np = param.data.cpu().numpy()
                    w_min = np.min(w_np)
                    w_max = np.max(w_np)

                    # Calculate scale to fit into [-128, 127]
                    # Then the range becomes [-128/scale, 127/scale]
                    abs_max = max(abs(w_min), abs(w_max))
                    if abs_max > 0:
                        scale = abs_max / 127.0
                    else:
                        scale = 1.0

                    self.scales[name] = scale

                    # Quantize
                    w_quantized = np.round(w_np / scale)
                    w_quantized = np.clip(w_quantized, -128, 127)

                    # Dequantize back to float
                    param.data = torch.tensor(w_quantized * scale, dtype=torch.float32)

    def forward(self, x):
        return self.float_model(x)

    def get_bit_width(self) -> int:
        return 8


def quantize_checkpoint(checkpoint_path: str, output_dir: str):
    """Load checkpoint and create Q4.4 quantized model."""
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create float model and load weights
    float_model = ECG_1DCNN(num_classes=4)
    float_model.load_state_dict(ckpt['model_state_dict'])
    float_model = float_model.to(device)

    # Quantize
    print("[INFO] Quantizing to Q4.4-style (8-bit signed, per-layer dynamic scaling)")
    q44_model = ECG_1DCNN_Q44(float_model)
    q44_model = q44_model.to(device)

    # Print per-layer scales
    print("\n[INFO] Per-layer quantization scales:")
    for name, scale in q44_model.scales.items():
        print(f"       {name:30s} scale={scale:.6f}")

    # Save quantized model with scales
    q44_ckpt = {
        'model_state_dict': q44_model.float_model.state_dict(),
        'quantization': 'Q4.4-dynamic',
        'scales': q44_model.scales,
        'bit_width': 8,
        'original_epoch': ckpt.get('epoch', '?'),
        'original_val_acc': ckpt.get('val_acc', None),
    }
    output_path = os.path.join(output_dir, 'model_q44.pth')
    torch.save(q44_ckpt, output_path)
    print(f"[DONE] Saved Q4.4 quantized model: {output_path}")


def main():
    p = argparse.ArgumentParser(
        description='Quantize ECG_1DCNN to Q4.4 (8-bit signed fixed-point)'
    )
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to trained checkpoint (.pth)')
    p.add_argument('--output_dir', type=str, default='./results/q44',
                   help='Output directory for quantized model')
    args = p.parse_args()

    quantize_checkpoint(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
