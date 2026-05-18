"""Export QAT-INT8 model to ONNX for hls4ml conversion.

The exported ONNX uses float32 arithmetic with QAT-trained weights
(fake-quantized). hls4ml will then apply fixed-point precision settings
to match the INT8 hardware pipeline.

Usage:
    python export_onnx.py \
        --checkpoint ./results/qat_int8/model_qat_float.pth \
        --output ./results/qat_int8/ecg_cnn.onnx

    # Verify the exported model:
    python export_onnx.py --verify --output ./results/qat_int8/ecg_cnn.onnx
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prune_finetune import ECG_1DCNN_Pruned


# ── Wrapper: plain nn.Module (no custom forward args) ──────────────────────────
# hls4ml requires a model whose forward() takes only the input tensor.
# ECG_1DCNN_QAT.forward() has a `quantize` kwarg which breaks ONNX tracing,
# so we wrap it here and bake the weights directly as nn parameters.

class ECGPrunedFloat(nn.Module):
    """Float32 forward pass using QAT-trained weights. ONNX-traceable."""

    def __init__(self, c1_out=3, c2_out=6, c3_out=6, c4_out=10, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1,      c1_out, kernel_size=5, padding=2, bias=True)
        self.conv2 = nn.Conv1d(c1_out, c2_out, kernel_size=5, padding=2, bias=True)
        self.conv3 = nn.Conv1d(c2_out, c3_out, kernel_size=5, padding=2, bias=True)
        self.conv4 = nn.Conv1d(c3_out, c4_out, kernel_size=5, padding=2, bias=True)
        self.pool  = nn.MaxPool1d(kernel_size=5, stride=5)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(c4_out, num_classes, bias=True)

    def forward(self, x):
        # x: (B, 1, 2500)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


def load_export_model(ckpt_path, device):
    """Load QAT float checkpoint into plain ONNX-traceable model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Detect pruned vs unpruned
    c1 = ckpt.get('c1_out', 4)
    c2 = ckpt.get('c2_out', 8)
    c3 = ckpt.get('c3_out', 8)
    c4 = ckpt.get('c4_out', 16)

    model = ECGPrunedFloat(c1_out=c1, c2_out=c2, c3_out=c3, c4_out=c4)

    # QAT checkpoint keys: 'conv1.weight', 'conv2.weight', etc.
    # (same as standard nn.Module state_dict)
    state = ckpt['model_state_dict']
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def export_onnx(model, output_path, input_length=2500):
    """Trace and export model to ONNX."""
    dummy = torch.randn(1, 1, input_length)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=['ecg_input'],
        output_names=['logits'],
        dynamic_axes={
            'ecg_input': {0: 'batch_size'},
            'logits':    {0: 'batch_size'},
        },
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"[OK] ONNX model saved: {output_path}")


def verify_onnx(onnx_path, input_length=2500):
    """Check ONNX graph is valid and run a forward pass."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[WARN] Install onnx + onnxruntime to verify: pip install onnx onnxruntime")
        return

    model_proto = onnx.load(onnx_path)
    onnx.checker.check_model(model_proto)
    print("[OK] ONNX graph validation passed")

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    dummy = np.random.randn(4, 1, input_length).astype(np.float32)
    out = sess.run(['logits'], {'ecg_input': dummy})[0]
    print(f"[OK] ONNX inference OK — output shape: {out.shape}, argmax: {out.argmax(axis=1)}")


def print_hls4ml_recipe(onnx_path, c4_out=10):
    """Print the hls4ml conversion recipe for this model."""
    print(f"""
─── hls4ml conversion recipe ────────────────────────────────────
import hls4ml

# 1. Load ONNX model
config = hls4ml.utils.config_from_onnx_model(
    '{onnx_path}',
    granularity='name',
    backend='Intel',          # use 'Vivado' for Xilinx
)

# 2. Set global precision to match QAT-INT8 pipeline
config['Model']['Precision'] = 'ap_fixed<8,1>'

# 3. Override per-layer if needed (accumulators need more bits)
# (hls4ml auto-widens accumulators, but you can tune here)

# 4. Convert
hls_model = hls4ml.converters.convert_from_onnx_model(
    '{onnx_path}',
    hls_config=config,
    output_dir='hls_ecg_cnn',
    backend='Intel',
    part='5CSEBA6U23I7',      # Cyclone V DE10-Nano part
)

# 5. C simulation
hls_model.compile()
dummy = np.random.randn(10, 1, 2500).astype(np.float32)
preds = hls_model.predict(dummy)
print(preds.argmax(axis=1))

# 6. Synthesis (requires Intel HLS or Quartus)
# hls_model.build()
─────────────────────────────────────────────────────────────────
""")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str,
                   default='./results/qat_int8/model_qat_float.pth',
                   help='QAT float checkpoint (model_qat_float.pth)')
    p.add_argument('--output', type=str,
                   default='./results/qat_int8/ecg_cnn.onnx')
    p.add_argument('--verify', action='store_true',
                   help='Run ONNX runtime verification after export')
    p.add_argument('--recipe', action='store_true',
                   help='Print hls4ml conversion recipe and exit')
    args = p.parse_args()

    if args.recipe:
        print_hls4ml_recipe(os.path.abspath(args.output))
        return

    device = torch.device('cpu')  # ONNX export always on CPU
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    model = load_export_model(args.checkpoint, device)

    # Quick sanity: count params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {n_params}")

    export_onnx(model, args.output)

    if args.verify:
        verify_onnx(args.output)

    print_hls4ml_recipe(os.path.abspath(args.output))


if __name__ == '__main__':
    main()
