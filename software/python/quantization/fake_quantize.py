"""Fake quantization layers for Quantization-Aware Training (QAT)

Uses Straight-Through Estimator (STE) for backpropagation:
- Forward: apply quantization (round + clip)
- Backward: gradient passes through as-is (no quantization gradient)
"""

import torch
import torch.nn as nn
import numpy as np


class FakeQuantize(nn.Module):
    """Fake quantization layer using STE."""

    def __init__(self, num_bits=8, scale=None, symmetric=True):
        """
        Args:
            num_bits: Number of bits (default 8 for INT8)
            scale: Fixed quantization scale. If None, scale is learnable.
            symmetric: If True, range is [-S, S-1]. If False, range is [0, 2^num_bits - 1].
        """
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric

        if symmetric:
            self.qmax = 2 ** (num_bits - 1) - 1  # 127 for 8-bit
        else:
            self.qmax = 2 ** num_bits - 1  # 255 for 8-bit

        if scale is not None:
            self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))
        else:
            self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x):
        """Quantize via STE."""
        # Fake quantization: round and clip
        x_q = torch.round(x / self.scale)
        if self.symmetric:
            x_q = torch.clamp(x_q, -self.qmax - 1, self.qmax)
        else:
            x_q = torch.clamp(x_q, 0, self.qmax)

        # Dequantize
        x_q_dequant = x_q * self.scale

        # Straight-Through Estimator: output quantized value but gradient flows through unquantized
        return x + (x_q_dequant - x).detach()


class QuantizeConv1d(nn.Module):
    """Conv1d layer with weight quantization for QAT."""

    def __init__(self, conv_layer, num_bits=8, scale=None):
        """
        Args:
            conv_layer: nn.Conv1d instance to wrap
            num_bits: Quantization bit width
            scale: Fixed quantization scale (per-layer)
        """
        super().__init__()
        self.conv = conv_layer
        self.weight_quantizer = FakeQuantize(num_bits=num_bits, scale=scale, symmetric=True)

    def forward(self, x):
        # Quantize weights
        w_q = self.weight_quantizer(self.conv.weight)

        # Use quantized weights in convolution
        return torch.nn.functional.conv1d(
            x,
            w_q,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class QuantizeLinear(nn.Module):
    """Linear layer with weight quantization for QAT."""

    def __init__(self, linear_layer, num_bits=8, scale=None):
        """
        Args:
            linear_layer: nn.Linear instance to wrap
            num_bits: Quantization bit width
            scale: Fixed quantization scale (per-layer)
        """
        super().__init__()
        self.linear = linear_layer
        self.weight_quantizer = FakeQuantize(num_bits=num_bits, scale=scale, symmetric=True)

    def forward(self, x):
        # Quantize weights
        w_q = self.weight_quantizer(self.linear.weight)

        # Use quantized weights in linear transformation
        return torch.nn.functional.linear(x, w_q, self.linear.bias)


def wrap_model_for_qat(model, scales_dict=None, num_bits=8):
    """
    Wrap a model's Conv1d and Linear layers with quantization-aware layers.

    Args:
        model: ECG_1DCNN model
        scales_dict: Dictionary mapping layer names to fixed scales
                     If None, scales are learnable.
        num_bits: Quantization bit width (default 8 for INT8)

    Returns:
        Model with wrapped quantization-aware layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Conv1d):
            scale = scales_dict.get(name) if scales_dict else None
            setattr(model, name, QuantizeConv1d(module, num_bits=num_bits, scale=scale))
        elif isinstance(module, nn.Linear):
            scale = scales_dict.get(name) if scales_dict else None
            setattr(model, name, QuantizeLinear(module, num_bits=num_bits, scale=scale))

    return model
