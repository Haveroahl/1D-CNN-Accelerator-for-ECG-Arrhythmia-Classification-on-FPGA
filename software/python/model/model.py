"""
1-D CNN for ECG Classification (FPGA-target)
============================================
Lightweight 1D-CNN architecture optimized for FPGA deployment.
No ReLU after Conv1, Conv2, Conv3; ReLU only after Conv4.

Two architectures live in this codebase:
  - ECG_1DCNN (this file)   : base/dense  1→4→8→8→16  (FC 16→4)
  - ECG_1DCNN_Pruned        : hardware target, defined in prune_finetune.py,
                              channels 1→4→4→8→8 (FC 8→4) — the deployed model.

ECG_1DCNN (base, dense) tensor shapes:
  Input (2500×1)
  Conv1 (1→4,  K=5, P=2) → MaxPool(5) →  500×4
  Conv2 (4→8,  K=5, P=2) → MaxPool(5) →  100×8
  Conv3 (8→8,  K=5, P=2) → MaxPool(5) →   20×8
  Conv4 (8→16, K=5, P=2) + ReLU → MaxPool(5) →    4×16
  GAP → (16,)
  FC   (16→4)
  → logits (4 classes: AFIB / GSVT / SB / SR)

Supports float32 training + quantization (Q8.8, Q4.4, INT8).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  Float model (v3: no ReLU after Conv1, Conv2, Conv3)
# ============================================================

class ECG_1DCNN(nn.Module):
    """
    Lightweight 1-D CNN for 4-class ECG classification.

    v3 variant: All Conv/MaxPool use K=5. ReLU only after Conv4 (before FC).
    Weights and biases are Q8.8 (16-bit) in hardware — bias IS present.

    Uses standard nn.Conv1d (cross-correlation), matching hardware behavior:
      y = Σ_k w[k] × sr[k]  (direct index, no kernel flip needed)
    """

    def __init__(self, num_classes=4, input_length=2500):
        super().__init__()
        self.num_classes  = num_classes
        self.input_length = input_length

        # Block 1: 2500×1 → 2500×4 → 500×4 (NO ReLU)
        self.conv1 = nn.Conv1d(1,  4,  kernel_size=5, padding=2, bias=True)
        self.pool1 = nn.MaxPool1d(kernel_size=5)

        # Block 2: 500×4 → 500×8 → 100×8 (NO ReLU)
        self.conv2 = nn.Conv1d(4,  8,  kernel_size=5, padding=2, bias=True)
        self.pool2 = nn.MaxPool1d(kernel_size=5)

        # Block 3: 100×8 → 100×8 → 20×8 (NO ReLU)
        self.conv3 = nn.Conv1d(8,  8,  kernel_size=5, padding=2, bias=True)
        self.pool3 = nn.MaxPool1d(kernel_size=5)

        # Block 4: 20×8 → 20×16 → 4×16 (WITH ReLU before FC)
        self.conv4 = nn.Conv1d(8,  16, kernel_size=5, padding=2, bias=True)
        self.pool4 = nn.MaxPool1d(kernel_size=5)

        # Global Average Pooling: (B, 16, 4) → (B, 16)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Fully-connected classifier: 16 → num_classes
        self.fc = nn.Linear(16, num_classes, bias=True)

    def forward(self, x):
        """
        Args:
            x: (B, 1, 2500) or (B, 2500)
        Returns:
            logits: (B, num_classes)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)   # (B, L) → (B, 1, L)

        x = self.pool1(self.conv1(x))                        # → (B,  4, 500) [NO ReLU]
        x = self.pool2(self.conv2(x))                        # → (B,  8, 100) [NO ReLU]
        x = self.pool3(self.conv3(x))                        # → (B,  8,  20) [NO ReLU]
        x = self.pool4(F.relu(self.conv4(x)))                # → (B, 16,   4) [+ReLU]
        x = self.gap(x).squeeze(-1)                          # → (B, 16)
        return self.fc(x)                                     # → (B,  4)

    def predict(self, x):
        """Return argmax of logits (no softmax needed for prediction)."""
        return self.forward(x).argmax(dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_info(self):
        rows = [
            f"{'Layer':<28} {'Output Shape':<20} {'Params':<10}",
            "=" * 60,
            f"{'Input':<28} {'(B, 1, 2500)':<20} {'-':<10}",
        ]
        entries = [
            ('Conv1 (1→4, K=5,P=2)',         '(B,  4, 2500)', self.conv1),
            ('MaxPool1 (/5)',                '(B,  4,  500)', None),
            ('Conv2 (4→8, K=5,P=2)',         '(B,  8,  500)', self.conv2),
            ('MaxPool2 (/5)',                '(B,  8,  100)', None),
            ('Conv3 (8→8, K=5,P=2)',         '(B,  8,  100)', self.conv3),
            ('MaxPool3 (/5)',                '(B,  8,   20)', None),
            ('Conv4+ReLU (8→16, K=5,P=2)',   '(B, 16,   20)', self.conv4),
            ('MaxPool4 (/5)',                '(B, 16,    4)', None),
            ('GAP',                          '(B, 16)',       None),
            (f'FC (16→{self.num_classes})',  f'(B,  {self.num_classes})', self.fc),
        ]
        total = 0
        for name, shape, layer in entries:
            if layer is not None:
                p = sum(x.numel() for x in layer.parameters())
                total += p
                rows.append(f"{name:<28} {shape:<20} {p:<10}")
            else:
                rows.append(f"{name:<28} {shape:<20} {'-':<10}")
        rows += ["=" * 60, f"{'Total Parameters':<28} {'':<20} {total:<10}"]
        return "\n".join(rows)


# ============================================================
#  Q8.8 round-trip model (hardware-fidelity simulation)
# ============================================================

def _q88_snap(value: float) -> float:
    """
    Snap one float value through Q8.8 fixed-point round-trip.
    Equivalent to what the FPGA weight ROM stores:
        q = round(v * 256)  clamped to [-32768, 32767]
        v' = q / 256
    """
    q = int(round(value * 256))
    q = max(-32768, min(32767, q))
    return q / 256.0


class ECG_1DCNN_Q88(nn.Module):
    """
    Q8.8 round-tripped copy of ECG_1DCNN.
    Weights are snapped to the nearest Q8.8 value to simulate
    the quantization loss introduced by the FPGA weight ROM.
    Accuracy measured here equals what the hardware will produce.
    """

    def __init__(self, float_model: ECG_1DCNN):
        super().__init__()
        import copy
        self.model = copy.deepcopy(float_model)

        with torch.no_grad():
            for param in self.model.parameters():
                w_np    = param.data.cpu().numpy()
                w_snapped = np.vectorize(_q88_snap)(w_np)
                param.data = torch.tensor(w_snapped, dtype=torch.float32)

    def forward(self, x):
        return self.model(x)


# ============================================================
#  INT8 simulation model (hardware-fidelity, power-of-2 scale)
# ============================================================

_LAYER_ORDER = ['conv1', 'conv2', 'conv3', 'conv4', 'fc']
_CONV_LAYERS = ['conv1', 'conv2', 'conv3', 'conv4']


def _compute_shift_bits(abs_max, qmax=127):
    if abs_max == 0:
        return 0
    return max(0, min(int(math.floor(math.log2(qmax / abs_max))), 15))


def _round_shift(x, n):
    """Round-half-up arithmetic right shift (matches hardware pe_requantize_fused)."""
    if n > 0:
        return torch.floor((x + 2.0 ** (n - 1)) / (2.0 ** n))
    return x


class ECG_1DCNN_INT8(nn.Module):
    """
    INT8 simulation of any ECG_1DCNN-compatible model (unpruned or pruned).

    Simulates the hardware integer forward pass:
      input_int8 = round(x * 2^input_shift)  clamp[-127,127]
      w_int8     = round(w * 2^w_shift)       clamp[-127,127]
      acc_int32  = conv(input_int8, w_int8) + bias_scaled
      out_int8   = clamp(round_shift(acc_int32, nb), -127, 127)
      ReLU after conv4 only (integer comparison >0)

    Args:
        float_model: trained float model (ECG_1DCNN or ECG_1DCNN_Pruned)
        calib_loader: DataLoader used to calibrate input_shift (20 batches)
    """

    def __init__(self, float_model, calib_loader, device=None, n_cal_batches=20):
        super().__init__()
        if device is None:
            device = next(float_model.parameters()).device
        self.device = device

        # ---- Weight quantization ----
        self._w_int8  = {}
        self._b_float = {}
        self._w_shift = {}

        for name in _LAYER_ORDER:
            layer = getattr(float_model, name)
            w_np = layer.weight.data.cpu().numpy()
            abs_max = max(abs(w_np.min()), abs(w_np.max()))
            n = _compute_shift_bits(abs_max)
            self._w_shift[name] = n
            self._w_int8[name]  = np.clip(np.round(w_np * (2.0 ** n)), -127, 127).astype(np.float32)
            self._b_float[name] = layer.bias.data.cpu().numpy().astype(np.float32) if layer.bias is not None else None

        # ---- Input shift (calibrated from data) ----
        max_input = 0.0
        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                if i >= n_cal_batches:
                    break
                v = batch[0].abs().max().item()
                if v > max_input:
                    max_input = v
        self._input_shift = _compute_shift_bits(max_input)

        # ---- Activation shift (nb) ----
        self._nb = {}
        for name in _CONV_LAYERS:
            self._nb[name] = (self._input_shift + self._w_shift[name]
                              if name == 'conv1' else self._w_shift[name])

        # Keep pool/gap/padding info from the float model (no parameters)
        self._pool1   = float_model.pool1
        self._pool2   = float_model.pool2
        self._pool3   = float_model.pool3
        self._pool4   = float_model.pool4
        self._gap     = float_model.gap
        self._padding = 2  # all conv layers use padding=2

    def _conv_layer(self, x, name):
        w = torch.tensor(self._w_int8[name]).to(self.device)
        n = self._nb[name]
        b_scaled = torch.tensor(
            np.round(self._b_float[name] * (2.0 ** n)).astype(np.float32)
        ).to(self.device) if self._b_float[name] is not None else None
        out = F.conv1d(x, w, b_scaled, padding=self._padding)
        return torch.clamp(_round_shift(out, n), -127, 127)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = torch.clamp(torch.round(x * (2.0 ** self._input_shift)), -127, 127)

        x = self._pool1(self._conv_layer(x, 'conv1'))
        x = self._pool2(self._conv_layer(x, 'conv2'))
        x = self._pool3(self._conv_layer(x, 'conv3'))

        # conv4: match RTL order acc → >>nb → clamp[-127,127] → ReLU → MaxPool
        w4 = torch.tensor(self._w_int8['conv4']).to(self.device)
        n4 = self._nb['conv4']
        b4 = torch.tensor(
            np.round(self._b_float['conv4'] * (2.0 ** n4)).astype(np.float32)
        ).to(self.device)
        x = F.conv1d(x, w4, b4, padding=self._padding)
        x = torch.clamp(_round_shift(x, n4), -127, 127)
        x = F.relu(x)
        x = self._pool4(x)

        x = self._gap(x).squeeze(-1)

        # FC has no output rescale → bias scaled to logit domain by 2^w_shift[fc]
        # (commensurate with Σ gap[2^0]·w_fc[2^w_shift]), matching RTL fc_bias.hex.
        w_fc = torch.tensor(self._w_int8['fc']).to(self.device)
        fc_shift = self._w_shift['fc']
        b_fc = torch.tensor(
            np.round(self._b_float['fc'] * (2.0 ** fc_shift)).astype(np.float32)
        ).to(self.device)
        return F.linear(x, w_fc, b_fc)


# ============================================================
#  Factory
# ============================================================

def build_model(num_classes=4, input_length=2500) -> ECG_1DCNN:
    return ECG_1DCNN(num_classes=num_classes, input_length=input_length)


# ============================================================
#  Quick self-test
# ============================================================

if __name__ == "__main__":
    model = build_model()
    print(model.get_layer_info())
    print(f"\nTotal parameters: {model.count_parameters()}")

    dummy = torch.randn(4, 1, 2500)
    out   = model(dummy)
    print(f"\nInput:  {dummy.shape}")
    print(f"Output: {out.shape}")

    q88 = ECG_1DCNN_Q88(model)
    q88_out = q88(dummy)
    diff = (out - q88_out).abs().max().item()
    print(f"\nMax weight-quantization error (float vs Q8.8): {diff:.6f}")
