"""Base quantizer interface for different fixed-point formats."""

import torch
import torch.nn as nn
import numpy as np
import copy


class BaseQuantizer(nn.Module):
    """Base class for quantized model wrappers."""

    def __init__(self, float_model, scale: int, q_min: int, q_max: int):
        super().__init__()
        self.float_model = copy.deepcopy(float_model)
        self.scale = scale
        self.q_min = q_min
        self.q_max = q_max

    def _snap_value(self, value: float) -> float:
        """Snap float value to quantized representation."""
        q = int(round(value * self.scale))
        q = max(self.q_min, min(self.q_max, q))
        return q / self.scale

    def quantize_weights(self):
        """Quantize all model weights."""
        with torch.no_grad():
            for param in self.float_model.parameters():
                w_np = param.data.cpu().numpy()
                w_snapped = np.vectorize(self._snap_value)(w_np)
                param.data = torch.tensor(w_snapped, dtype=torch.float32)

    def forward(self, x):
        return self.float_model(x)

    def get_bit_width(self) -> int:
        """Return effective bit width of this quantization."""
        raise NotImplementedError
