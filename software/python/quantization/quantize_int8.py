"""INT8 Quantization following fphys (Liu et al., 2023) method.

Method (Eq.10-12 from paper):
  Weight quantization : w_int8 = floor(w / |w|_max * 127 + 0.5)
                        range [-127, 127] (symmetric)
  Input quantization  : same formula with |x|_max from dataset
  Activation rescale  : O'[l] = O[l] >> nb  (arithmetic right shift)
                        nb = ceil(log2(max_dataset |O[l]| / 127))
                        where O[l] is raw conv output (before ReLU/pool)
  Clip                : clamp O'[l] to [-127, 127]

Pipeline:
  Input(INT8) -> Conv -> [ReLU] -> >>nb -> clip[-127,127] -> MaxPool -> ...
  (ReLU only after Conv4 in this model; Conv1-3 have no ReLU)

Hardware: no float multiply needed for rescaling — pure arithmetic right shift.

Usage:
    python quantize_int8.py --checkpoint ./results/best_model.pth \\
                            --output_dir ./results/int8 \\
                            --data_dir   /home/duc/Thesis/data/Chapman
"""

import os
import sys
import argparse
import math
import copy

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from model.model import ECG_1DCNN
from prune_finetune import ECG_1DCNN_Pruned
from utils.dataset import get_dataloaders


CONV_LAYERS = ['conv1', 'conv2', 'conv3', 'conv4']
LAYER_ORDER = CONV_LAYERS + ['fc']


# ── Quantization helpers ─────────────────────────────────────────

def quantize_to_int8(arr: np.ndarray) -> tuple:
    """Power-of-2 symmetric quantization (hardware-friendly).

    Finds largest N such that max(|w|) * 2^N <= 127,
    then quantizes: w_q = round(w * 2^N).
    Scale = 2^N (power-of-2), dequantize via: w = w_q / 2^N = w_q >> N
    """
    abs_max = max(abs(float(arr.min())), abs(float(arr.max())))
    if abs_max == 0:
        return np.zeros_like(arr, dtype=np.float32), 0

    qmax = 127  # 8-bit signed: [-127, 127]
    # N = floor(log2(qmax / max_val))
    shift_bits = int(np.floor(np.log2(qmax / abs_max)))
    shift_bits = max(0, min(shift_bits, 15))

    scale = 2.0 ** shift_bits
    q = np.round(arr * scale)
    return np.clip(q, -127, 127).astype(np.float32), shift_bits


def _rescale_clip(t: torch.Tensor, nb: int) -> torch.Tensor:
    """Simulate hardware arithmetic right shift by nb bits, clip [-127, 127].

    In hardware: acc_int32 >>> nb → clamp[-127,127] → INT8
    In Python:   floor(t / 2^nb)  → clamp[-127,127]
    """
    if nb == 0:
        return torch.clamp(t, -127, 127)
    return torch.clamp(torch.floor(t / (2 ** nb)), -127, 127)


# ── fphys INT8 model ─────────────────────────────────────────────

class ECG_1DCNN_INT8_fphys(torch.nn.Module):
    """Fully-integer ECG CNN following fphys quantization method.

    Forward pass simulates integer hardware pipeline:
      - All weights stored as INT8 integers (as float tensors)
      - Input quantized to INT8 on entry
      - After each conv: arithmetic right shift (>>nb) + clip [-127,127]
      - ReLU applied after conv4 only (integer comparison >0)
      - MaxPool on integer values
      - FC output = raw integer logits (argmax for prediction)
    """

    def __init__(self, float_model, from_checkpoint=False, pre_scales=None):
        super().__init__()
        self.model = copy.deepcopy(float_model)
        self.w_scales = {}    # per-layer: shift_bits (power-of-2 scale = 2^shift_bits)
        self.input_scale = 1.0
        self.nb = {}          # per-layer activation shift bits

        # If loading from checkpoint, scales are set externally (don't quantize here)
        if from_checkpoint:
            return

        # Quantize weights to INT8 integers (power-of-2 scale: w_q = round(w * 2^N))
        with torch.no_grad():
            for name in LAYER_ORDER:
                layer = getattr(self.model, name, None)
                if layer is None:
                    continue
                w_np = layer.weight.data.cpu().numpy()

                # Use pre-computed shift_bits if provided, else compute from weights
                if pre_scales and name in pre_scales:
                    shift_bits = pre_scales[name]
                    scale = 2.0 ** shift_bits
                    w_int = np.round(w_np * scale)
                    w_int = np.clip(w_int, -127, 127).astype(np.float32)
                else:
                    w_int, shift_bits = quantize_to_int8(w_np)

                self.w_scales[name] = shift_bits
                layer.weight.data = torch.tensor(w_int)

                # Bias: quantize with same scale
                if layer.bias is not None:
                    b_np = layer.bias.data.cpu().numpy()
                    scale = 2.0 ** shift_bits
                    b_int = np.round(b_np * scale)
                    b_int = np.clip(b_int, -127, 127).astype(np.float32)
                    layer.bias.data = torch.tensor(b_int)

    # ── Calibration ───────────────────────────────────────────────

    def calibrate(self, dataloader, device, n_batches: int = 20):
        """Two-phase calibration following fphys with power-of-2 scale.

        Phase 1: compute input_shift_bits (power-of-2 input scale = 2^N)
        Phase 2: compute nb_l = ceil(log2(max_dataset|O_l| / 127))
                 O_l = raw output of conv_l (before ReLU for conv4, before pool)
        """
        # ── Phase 1: input shift bits (power-of-2 scale) ──────────────────
        max_input = 0.0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break
                x = batch[0]  # unpack only ECG, ignore label and HR
                v = x.abs().max().item()
                if v > max_input:
                    max_input = v

        # Compute shift_bits: N = floor(log2(127 / max_input))
        qmax = 127
        if max_input > 0:
            self.input_shift_bits = int(np.floor(np.log2(qmax / max_input)))
            self.input_shift_bits = max(0, min(self.input_shift_bits, 15))
        else:
            self.input_shift_bits = 0
        self.input_scale = 2.0 ** self.input_shift_bits

        # ── Phase 2: activation shift bits ───────────────────────
        # Hook raw conv outputs (before ReLU for conv4, before pool for all)
        max_acts = {n: 0.0 for n in CONV_LAYERS}
        hooks = []

        for name in CONV_LAYERS:
            layer = getattr(self.model, name)
            apply_relu = (name == 'conv4')

            def make_hook(n, relu):
                def hook(mod, inp, out):
                    t = F.relu(out) if relu else out
                    v = t.detach().abs().max().item()
                    if v > max_acts[n]:
                        max_acts[n] = v
                return hook

            hooks.append(layer.register_forward_hook(make_hook(name, apply_relu)))

        # Run forward with FLOAT32 inputs (not quantized!)
        # We need to measure activations with original float inputs
        # to get correct nb values
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break
                x = batch[0].to(device)  # unpack only ECG, ignore label and HR
                # Use FLOAT32 input for calibration, not INT8!
                self._raw_forward(x)  # runs with float inputs

        for h in hooks:
            h.remove()

        # Eq.11: nb_l = ceil(log2(max|O_l| / 127))
        for name in CONV_LAYERS:
            m = max_acts[name]
            self.nb[name] = math.ceil(math.log2(m / 127)) if m > 127 else 0

        return self.nb

    # ── Forward passes ────────────────────────────────────────────

    def _raw_forward(self, x):
        """Float model forward (used during calibration)."""
        return self.model(x)

    def forward(self, x):
        """Fully-integer simulation forward (used for accuracy evaluation).

        x: float tensor (original ECG values)
        Returns integer-valued logits (argmax = predicted class)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Quantize input to INT8 integers (power-of-2 scale: x_q = round(x * 2^N))
        x = torch.round(x * self.input_scale)
        x = torch.clamp(x, -127, 127)

        # Conv1 → rescale → clip → MaxPool  (no ReLU)
        x = _rescale_clip(self.model.conv1(x), self.nb.get('conv1', 0))
        x = self.model.pool1(x)

        # Conv2 → rescale → clip → MaxPool  (no ReLU)
        x = _rescale_clip(self.model.conv2(x), self.nb.get('conv2', 0))
        x = self.model.pool2(x)

        # Conv3 → rescale → clip → MaxPool  (no ReLU)
        x = _rescale_clip(self.model.conv3(x), self.nb.get('conv3', 0))
        x = self.model.pool3(x)

        # Conv4 → ReLU → rescale → clip → MaxPool
        x = _rescale_clip(F.relu(self.model.conv4(x)), self.nb.get('conv4', 0))
        x = self.model.pool4(x)

        # GAP → FC (no rescaling; argmax on raw logits)
        x = self.model.gap(x).squeeze(-1)
        return self.model.fc(x)


# ── Quantize checkpoint ───────────────────────────────────────────

def quantize_checkpoint(checkpoint_path: str, output_dir: str,
                        data_dir: str, n_cal_batches: int = 20):
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

    print("[INFO] Computing per-layer weight shift bits from original float weights")
    original_w_shifts = {}
    for name in LAYER_ORDER:
        layer = getattr(float_model, name, None)
        if layer is None:
            continue
        w_np = layer.weight.data.cpu().numpy()
        abs_max = max(abs(float(w_np.min())), abs(float(w_np.max())))
        if abs_max == 0:
            shift_bits = 0
        else:
            qmax = 127
            shift_bits = int(np.floor(np.log2(qmax / abs_max)))
            shift_bits = max(0, min(shift_bits, 15))
        original_w_shifts[name] = shift_bits

    print("[INFO] Per-layer weight shift bits (scale = 2^N):")
    for name, nb in original_w_shifts.items():
        print(f"       {name:10s}  shift_bits={nb}  (scale=2^{nb})")

    print(f"\n[INFO] Calibrating nb and input_scale "
          f"over {n_cal_batches} training batches ...")
    print(f"       (using float32 weights for calibration)")

    # Create INT8 model with float weights first (for calibration)
    int8_model = ECG_1DCNN_INT8_fphys(float_model, pre_scales=original_w_shifts)
    int8_model = int8_model.to(device)

    _, train_loader, _ = get_dataloaders(data_dir, batch_size=128, num_workers=2)
    nb_dict = int8_model.calibrate(train_loader, device, n_batches=n_cal_batches)

    print(f"\n[INFO] Quantizing weights to INT8 after calibration")
    # Now quantize weights to INT8 after calibration
    with torch.no_grad():
        for name in LAYER_ORDER:
            layer = getattr(int8_model.model, name, None)
            if layer is None:
                continue
            w_np = layer.weight.data.cpu().numpy()
            shift_bits = original_w_shifts[name]
            scale = 2.0 ** shift_bits
            w_int = np.round(w_np * scale)
            w_int = np.clip(w_int, -127, 127).astype(np.float32)
            layer.weight.data = torch.tensor(w_int)

            if layer.bias is not None:
                b_np = layer.bias.data.cpu().numpy()
                b_int = np.round(b_np * scale)
                b_int = np.clip(b_int, -127, 127).astype(np.float32)
                layer.bias.data = torch.tensor(b_int)

    print(f"\n[INFO] input_shift_bits = {int8_model.input_shift_bits}  "
          f"(input_scale = 2^{int8_model.input_shift_bits})")
    print("[INFO] Per-layer activation shift bits nb (Eq.11):")
    for name, nb in nb_dict.items():
        print(f"       {name:10s}  nb={nb}  (output >> {nb} bits)")

    int8_ckpt = {
        'model_state_dict': int8_model.model.state_dict(),
        'quantization': 'INT8-fphys',
        'w_scales': original_w_shifts,  # shift_bits per layer (power-of-2 scale)
        'nb': nb_dict,
        'input_shift_bits': int8_model.input_shift_bits,
        'input_scale': float(int8_model.input_scale),
        'bit_width': 8,
        'original_epoch': ckpt.get('epoch', '?'),
        'original_val_acc': ckpt.get('val_acc', None),
    }
    for key in ('c1_out', 'c2_out', 'c3_out', 'c4_out'):
        if key in ckpt:
            int8_ckpt[key] = ckpt[key]

    output_path = os.path.join(output_dir, 'model_int8.pth')
    torch.save(int8_ckpt, output_path)
    print(f"\n[DONE] Saved: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='INT8 quantization following fphys (Liu et al., 2023)'
    )
    p.add_argument('--checkpoint',    type=str, required=True)
    p.add_argument('--output_dir',    type=str, default='./results/int8')
    p.add_argument('--data_dir',      type=str,
                   default='/home/duc/Thesis/data/Chapman')
    p.add_argument('--n_cal_batches', type=int, default=20,
                   help='Training batches for input_scale and nb calibration')
    args = p.parse_args()
    quantize_checkpoint(args.checkpoint, args.output_dir,
                        args.data_dir, args.n_cal_batches)


if __name__ == "__main__":
    main()
