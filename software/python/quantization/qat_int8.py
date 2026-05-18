"""Quantization-Aware Training (QAT) + INT8 Export

Pipeline:
  1. QAT training:   fake-quantize weights + activations during training
                     → weights learn to fit INT8 range naturally
  2. INT8 convert:   w_int8 = round(w * 2^N), clamp [-127, 127]
                     compute input_shift_bits from dataset max
  3. Evaluate:       simulate hardware integer forward pass
  4. Export:         .mem files for Verilog ROM

QAT fake quantization (Straight-Through Estimator):
  forward:   x_q = round(x / scale) * scale   (quantize + dequantize)
  backward:  dx_q = dx  (STE: pass gradient through)

Hardware pipeline (INT8):
  input_int8  = round(x * 2^input_shift)       clamp [-127, 127]
  w_int8      = round(w * 2^w_shift_l)         clamp [-127, 127]
  acc_int32   = conv(input_int8, w_int8) + bias_int
  out_int8    = clamp(acc_int32 >> nb_l, -127, 127)
  (ReLU after conv4 only: comparison >0 on integers)

Usage:
    # Full pipeline: QAT + convert + evaluate + export
    python quantization/qat_int8.py \\
        --checkpoint ./results/best_model_pruned.pth \\
        --output_dir ./results/qat_int8 \\
        --data_dir   /home/duc/Thesis/data/Chapman \\
        --epochs 30 --lr 1e-4

    # Evaluate only (model already trained)
    python quantization/qat_int8.py \\
        --checkpoint ./results/best_model_pruned.pth \\
        --output_dir ./results/qat_int8 \\
        --data_dir   /home/duc/Thesis/data/Chapman \\
        --eval_only
"""

import os
import sys
import argparse
import json
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from model.model import ECG_1DCNN
from prune_finetune import ECG_1DCNN_Pruned
from utils.dataset import get_dataloaders, CLASS_NAMES
from utils.evaluate import evaluate_model, compute_metrics, print_classification_report


LAYER_ORDER = ['conv1', 'conv2', 'conv3', 'conv4', 'fc']
CONV_LAYERS = ['conv1', 'conv2', 'conv3', 'conv4']


# ============================================================
#  Fake Quantize — EMA scale + STE (chuẩn QAT)
# ============================================================

class FakeQuantize(nn.Module):
    """Per-tensor fake quantization với EMA scale và STE gradient.

    Training: scale cập nhật bằng EMA của abs_max qua các batch.
              Sau warmup_steps, scale được freeze để ổn định.
    Eval:     dùng scale đã học, không cập nhật.

    STE: forward quantize+dequantize, backward gradient pass-through.
    """

    def __init__(self, qmax=127, momentum=0.01, warmup_steps=100):
        super().__init__()
        self.qmax = qmax
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('step', torch.tensor(0))

    def forward(self, x):
        if self.training:
            abs_max = x.detach().abs().max().clamp(min=1e-8)
            new_scale = abs_max / self.qmax
            if self.step == 0:
                self.scale.copy_(new_scale)
            else:
                self.scale.mul_(1.0 - self.momentum).add_(new_scale * self.momentum)
            self.step += 1

        scale = self.scale.clamp(min=1e-8)
        # STE: x_q trong forward, gradient thẳng trong backward
        x_q = (x / scale).round().clamp(-self.qmax, self.qmax) * scale
        return x + (x_q - x).detach()


# ============================================================
#  QAT Model
# ============================================================

class ECG_1DCNN_QAT(nn.Module):
    """ECG CNN với fake quantization chuẩn QAT.

    Thứ tự mô phỏng đúng RTL hardware pipeline:
      Conv1-3: fq_input(x) → conv(fq_w) → fq_act → pool
      Conv4:   fq_input(x) → conv(fq_w) → fq_act → ReLU → pool
               (RTL: acc → >>nb → clamp[-127,127] → ReLU → MaxPool)
      FC:      fq_input(x) → linear(fq_w)

    Mỗi layer có FakeQuantize riêng (scale độc lập).
    """

    def __init__(self, c1_out=4, c2_out=4, c3_out=8, c4_out=8, num_classes=4):
        super().__init__()
        self.c1_out = c1_out
        self.c2_out = c2_out
        self.c3_out = c3_out
        self.c4_out = c4_out

        self.conv1 = nn.Conv1d(1,      c1_out, kernel_size=5, padding=2, bias=True)
        self.conv2 = nn.Conv1d(c1_out, c2_out, kernel_size=5, padding=2, bias=True)
        self.conv3 = nn.Conv1d(c2_out, c3_out, kernel_size=5, padding=2, bias=True)
        self.conv4 = nn.Conv1d(c3_out, c4_out, kernel_size=5, padding=2, bias=True)

        self.pool1 = nn.MaxPool1d(kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=5)
        self.pool3 = nn.MaxPool1d(kernel_size=5)
        self.pool4 = nn.MaxPool1d(kernel_size=5)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(c4_out, num_classes, bias=True)

        # FakeQuantize per weight tensor
        self.fq_w1 = FakeQuantize()
        self.fq_w2 = FakeQuantize()
        self.fq_w3 = FakeQuantize()
        self.fq_w4 = FakeQuantize()
        self.fq_wfc = FakeQuantize()

        # FakeQuantize per activation (input quantization trước mỗi conv)
        self.fq_in  = FakeQuantize()   # input ECG
        self.fq_a1  = FakeQuantize()   # sau pool1 (input của conv2)
        self.fq_a2  = FakeQuantize()   # sau pool2 (input của conv3)
        self.fq_a3  = FakeQuantize()   # sau pool3 (input của conv4)
        self.fq_a4  = FakeQuantize()   # sau clamp conv4, trước ReLU
        self.fq_gap = FakeQuantize()   # sau gap (input của fc)

    def forward(self, x, quantize=True):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if not quantize:
            # Float eval — không fake-quantize
            x = self.pool1(F.conv1d(x, self.conv1.weight, self.conv1.bias, padding=2))
            x = self.pool2(F.conv1d(x, self.conv2.weight, self.conv2.bias, padding=2))
            x = self.pool3(F.conv1d(x, self.conv3.weight, self.conv3.bias, padding=2))
            x = self.pool4(F.relu(F.conv1d(x, self.conv4.weight, self.conv4.bias, padding=2)))
            x = self.gap(x).squeeze(-1)
            return F.linear(x, self.fc.weight, self.fc.bias)

        # QAT forward: fq_input → conv(fq_weight) → fq_activation
        # Đúng thứ tự RTL: input INT8 → MAC → >>nb → clamp → (ReLU conv4)

        # Conv1: quantize input ECG → conv(fq_w) → quantize activation
        x = self.fq_a1(self.pool1(F.conv1d(
            self.fq_in(x), self.fq_w1(self.conv1.weight), self.conv1.bias, padding=2
        )))

        # Conv2: input đã là fq_a1, quantize lại để đảm bảo scale ổn định
        x = self.fq_a2(self.pool2(F.conv1d(
            x, self.fq_w2(self.conv2.weight), self.conv2.bias, padding=2
        )))

        # Conv3
        x = self.fq_a3(self.pool3(F.conv1d(
            x, self.fq_w3(self.conv3.weight), self.conv3.bias, padding=2
        )))

        # Conv4: fq_act (mô phỏng clamp S7) → ReLU (S8) — đúng thứ tự RTL
        x = self.pool4(F.relu(self.fq_a4(F.conv1d(
            x, self.fq_w4(self.conv4.weight), self.conv4.bias, padding=2
        ))))

        # GAP + FC
        x = self.gap(x).squeeze(-1)
        x = self.fq_gap(x)
        return F.linear(x, self.fq_wfc(self.fc.weight), self.fc.bias)


def build_qat_model(base_model):
    """Tạo QAT model và copy weights từ float model."""
    if isinstance(base_model, ECG_1DCNN_Pruned):
        qat = ECG_1DCNN_QAT(
            c1_out=base_model.c1_out, c2_out=base_model.c2_out,
            c3_out=base_model.c3_out, c4_out=base_model.c4_out,
        )
    else:
        qat = ECG_1DCNN_QAT()

    with torch.no_grad():
        for name in LAYER_ORDER:
            src = getattr(base_model, name)
            dst = getattr(qat, name)
            dst.weight.copy_(src.weight)
            if src.bias is not None:
                dst.bias.copy_(src.bias)

    return qat


# ============================================================
#  INT8 conversion (power-of-2 scale)
# ============================================================

def compute_shift_bits(abs_max, qmax=127):
    """Largest N such that abs_max * 2^N <= qmax → dequantize via >>N."""
    if abs_max == 0:
        return 0
    n = int(math.floor(math.log2(qmax / abs_max)))
    return max(0, min(n, 15))


def convert_to_int8(qat_model, train_loader, device, n_cal_batches=20):
    """Convert QAT float weights → INT8, calibrate input + activation shifts.

    Returns:
        w_int8:  dict layer → int8 numpy array (weights)
        b_int8:  dict layer → int8 numpy array (bias)
        w_shift: dict layer → int  (weight shift bits)
        nb:      dict layer → int  (activation shift bits, conv only)
        input_shift: int  (input shift bits)
    """
    qat_model.eval()

    # ---- Weight shift bits ----
    w_shift = {}
    w_int8  = {}
    b_int8  = {}

    for name in LAYER_ORDER:
        layer = getattr(qat_model, name)
        w_np = layer.weight.data.cpu().numpy()
        abs_max = max(abs(w_np.min()), abs(w_np.max()))
        n = compute_shift_bits(abs_max)
        w_shift[name] = n
        scale = 2.0 ** n
        w_int8[name] = np.clip(np.round(w_np * scale), -127, 127).astype(np.int8)
        # Bias: no clamp to int8, store as int32-range (bias is added to acc_int32)
        # bias_int = round(b * 2^acc_shift) — computed after nb is known
        if layer.bias is not None:
            b_int8[name] = layer.bias.data.cpu().numpy()  # keep float, scale later

    # ---- Input shift bits (from dataset max) ----
    max_input = 0.0
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= n_cal_batches:
                break
            v = batch[0].abs().max().item()
            if v > max_input:
                max_input = v
    input_shift = compute_shift_bits(max_input)

    # ---- Activation shift bits (nb) ----
    # After INT8 conv: acc = (x_int * w_int) where x_int = x * 2^input_shift_prev
    # and w_int = w * 2^w_shift[layer]
    # So acc = real_value * 2^(input_shift_prev + w_shift[layer])
    # To get output back to INT8 range: nb = input_shift_prev + w_shift[layer]
    # input_shift evolves through layers: after conv_l, output is in range [-127,127]
    # so next layer sees input_shift = 0 (we clamp to INT8 after each conv)

    nb = {}
    for name in CONV_LAYERS:
        if name == 'conv1':
            acc_shift = input_shift + w_shift[name]
        else:
            acc_shift = w_shift[name]  # prev layer output already in INT8 range
        nb[name] = acc_shift

    # Warn if computed nb differs from RTL hardcoded values (PROJECT.md)
    RTL_NB = {'conv1': 8, 'conv2': 6, 'conv3': 6, 'conv4': 7}
    for name, rtl_val in RTL_NB.items():
        if nb[name] != rtl_val:
            print(f"  [WARNING] nb[{name}]={nb[name]} but RTL hardcodes {rtl_val} "
                  f"(w_shift={w_shift[name]}). "
                  f"RTL will shift wrong — retrain or update RTL nb.")

    return w_int8, b_int8, w_shift, nb, input_shift


# ============================================================
#  INT8 Simulation forward (for accuracy eval)
# ============================================================

def round_shift(x, n):
    """Round-half-up arithmetic right shift: matches hardware pe_requantize_fused.
    y = floor((x + 2^(n-1)) / 2^n)  for n > 0
    y = x                             for n == 0
    """
    if n > 0:
        return torch.floor((x + 2.0 ** (n - 1)) / (2.0 ** n))
    return x


def int8_forward(qat_model, x, w_int8, b_int8, w_shift, nb, input_shift):
    """Simulate hardware INT8 integer forward pass.

    All arithmetic done as float tensors representing integer values.
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)

    device = next(qat_model.parameters()).device

    # Quantize input: x_int = round(x * 2^input_shift), range [-127,127]
    x = torch.clamp(torch.round(x * (2.0 ** input_shift)), -127, 127)

    def conv_int8_layer(x, layer_name):
        """
        x: INT8 input (values in [-127,127])
        w: INT8 weights (values in [-127,127])
        acc = conv(x, w) + bias_scaled  →  acc >> nb  →  clamp[-127,127]

        bias_scaled = round(b_float * 2^nb) so that after >>nb we get b_float back
        nb = w_shift for conv2-4 (input already INT8)
           = input_shift + w_shift for conv1
        """
        w = torch.tensor(w_int8[layer_name].astype(np.float32)).to(device)
        b_float = b_int8[layer_name]  # stored as float
        layer = getattr(qat_model, layer_name)
        n = nb[layer_name]

        # Scale bias to same domain as accumulator (before >>n)
        b_scaled = torch.tensor(np.round(b_float * (2.0 ** n)).astype(np.float32)).to(device)

        out = F.conv1d(x, w, b_scaled, padding=layer.padding)

        return torch.clamp(round_shift(out, n), -127, 127)

    x = qat_model.pool1(conv_int8_layer(x, 'conv1'))
    x = qat_model.pool2(conv_int8_layer(x, 'conv2'))
    x = qat_model.pool3(conv_int8_layer(x, 'conv3'))

    # conv4: >>nb → clamp → ReLU  (matches RTL S6→S7→S8)
    w4 = torch.tensor(w_int8['conv4'].astype(np.float32)).to(device)
    b_float4 = b_int8['conv4']
    n4 = nb['conv4']
    b4_scaled = torch.tensor(np.round(b_float4 * (2.0 ** n4)).astype(np.float32)).to(device)
    x = torch.clamp(round_shift(F.conv1d(x, w4, b4_scaled, padding=qat_model.conv4.padding), n4), -127, 127)
    x = torch.clamp(x, min=0)
    x = qat_model.pool4(x)

    x = qat_model.gap(x).squeeze(-1)

    # FC: input is INT8, w_shift[fc] applied
    # Output = raw integer logits (no rescaling needed, argmax is scale-invariant)
    w_fc = torch.tensor(w_int8['fc'].astype(np.float32)).to(device)
    b_float_fc = b_int8['fc']
    # FC output is used for argmax only — no rescaling needed, so nb=0
    # bias is added unscaled (b_float * 2^0 = b_float, rounded to int)
    b_fc_scaled = torch.tensor(np.round(b_float_fc).astype(np.float32)).to(device)
    return F.linear(x, w_fc, b_fc_scaled)


def evaluate_int8(qat_model, test_loader, w_int8, b_int8, w_shift, nb, input_shift, device):
    """Evaluate INT8 model accuracy."""
    qat_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            y = batch[1]
            logits = int8_forward(qat_model, x, w_int8, b_int8, w_shift, nb, input_shift)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    preds_np  = np.array(all_preds)
    labels_np = np.array(all_labels)
    acc = (preds_np == labels_np).mean()
    return acc, preds_np, labels_np


# ============================================================
#  Export .mem files
# ============================================================

def export_mem_files(w_int8, b_int8, w_shift, nb, input_shift, output_dir):
    """Export INT8 weights to .mem files for Verilog ROM."""
    mem_dir = os.path.join(output_dir, 'mem_files')
    os.makedirs(mem_dir, exist_ok=True)

    def to_hex2(v):
        return f"{int(v) & 0xFF:02X}"

    all_weights = []

    for name in LAYER_ORDER:
        w = w_int8[name].flatten()
        mem_path = os.path.join(mem_dir, f"{name}_weight.mem")
        with open(mem_path, 'w') as f:
            f.write(f"// {name}_weight  shape={list(w_int8[name].shape)}\n")
            f.write(f"// INT8  shift_bits={w_shift[name]}  (scale=2^{w_shift[name]})\n")
            for v in w:
                f.write(to_hex2(v) + "\n")
        all_weights.extend(w.tolist())

        if name in b_int8:
            b_float = b_int8[name].flatten()
            # Bias must be pre-scaled by 2^nb so hardware can add directly to acc_int32
            # After acc_int32 >> nb, the bias contributes its original float value
            n_bias = nb.get(name, 0)  # conv: use nb; fc: nb=0 (argmax is scale-invariant)
            b_scaled = np.round(b_float * (2.0 ** n_bias)).astype(np.int32)
            mem_path = os.path.join(mem_dir, f"{name}_bias.mem")
            with open(mem_path, 'w') as f:
                f.write(f"// {name}_bias  scaled by 2^{n_bias}  (add to acc_int32 before >>nb)\n")
                f.write(f"// Hardware: b_eff = b_float * 2^{n_bias},  after >>nb = b_float\n")
                for v in b_scaled:
                    # Bias can exceed INT8 range — store as 32-bit hex (8 chars)
                    f.write(f"{int(v) & 0xFFFFFFFF:08X}\n")

    # flat_weights.mem
    flat_path = os.path.join(mem_dir, 'flat_weights.mem')
    with open(flat_path, 'w') as f:
        f.write(f"// All weights concatenated, order: {' -> '.join(LAYER_ORDER)}\n")
        f.write(f"// Total: {len(all_weights)} entries\n")
        for v in all_weights:
            f.write(to_hex2(v) + "\n")

    # nb_shifts.mem
    nb_path = os.path.join(mem_dir, 'nb_shifts.mem')
    with open(nb_path, 'w') as f:
        f.write("// Per-layer activation right-shift bits\n")
        f.write("// Hardware: acc_int32 >>> nb  (arithmetic right shift)\n")
        f.write(f"// Layer order: {' -> '.join(LAYER_ORDER)}\n")
        for name in LAYER_ORDER:
            n = nb.get(name, 0)
            f.write(f"{n:02X}  // {name}\n")

    # input_scale.txt
    with open(os.path.join(mem_dir, 'input_scale.txt'), 'w') as f:
        f.write(f"// Input quantization: x_int8 = round(x * 2^{input_shift})\n")
        f.write(f"// input_shift_bits = {input_shift}\n")
        f.write(f"// input_scale = {2**input_shift}\n")

    # weights_summary.json
    summary = {
        'w_shift': w_shift,
        'nb': nb,
        'input_shift_bits': input_shift,
        'input_scale': 2 ** input_shift,
        'total_weights': len(all_weights),
        'layers': {name: list(w_int8[name].shape) for name in LAYER_ORDER},
    }
    with open(os.path.join(mem_dir, 'weights_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[INFO] Exported .mem files to {mem_dir}/")
    print(f"  flat_weights.mem   ({len(all_weights)} entries)")
    print(f"  nb_shifts.mem      (per-layer activation shifts)")
    print(f"  input_scale.txt    (input_shift_bits={input_shift})")
    print(f"\n  Verilog parameters:")
    for name in CONV_LAYERS:
        print(f"    parameter NB_{name.upper()} = {nb.get(name, 0)};")
    print(f"    parameter INPUT_SHIFT = {input_shift};")

    return mem_dir


# ============================================================
#  Main pipeline
# ============================================================

def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # ---- Load base checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    is_pruned = 'c1_out' in ckpt

    if is_pruned:
        print(f"[INFO] Detected pruned model "
              f"(c1={ckpt['c1_out']}, c2={ckpt['c2_out']}, "
              f"c3={ckpt['c3_out']}, c4={ckpt['c4_out']})")
        base_model = ECG_1DCNN_Pruned(
            c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
            c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
        )
    else:
        base_model = ECG_1DCNN(num_classes=4)

    base_model.load_state_dict(ckpt['model_state_dict'])
    base_model = base_model.to(device)

    # ---- Data ----
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=2
    )

    # ---- Phase 1: QAT training ----
    qat_path = os.path.join(args.output_dir, 'model_qat_float.pth')

    if not args.eval_only:
        print(f"\n{'='*60}")
        print(f"  Phase 1: QAT Training ({args.epochs} epochs, lr={args.lr})")
        print(f"{'='*60}")

        qat_model = build_qat_model(base_model).to(device)
        optimizer = optim.Adam(qat_model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        history = []

        for epoch in range(args.epochs):
            qat_model.train()
            total_loss = total_correct = total_n = 0

            for batch in train_loader:
                x = batch[0].to(device)
                y = batch[1].to(device)
                optimizer.zero_grad()
                logits = qat_model(x, quantize=True)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    total_loss    += loss.item() * x.size(0)
                    total_correct += (logits.argmax(1) == y).sum().item()
                    total_n       += x.size(0)

            train_acc = total_correct / total_n

            qat_model.eval()
            with torch.no_grad():
                preds_v, labels_v = [], []
                for batch in val_loader:
                    xv = batch[0].to(device)
                    yv = batch[1]
                    preds_v.extend(qat_model(xv, quantize=False).argmax(1).cpu().numpy())
                    labels_v.extend(yv.numpy())
            val_acc = (np.array(preds_v) == np.array(labels_v)).mean()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({'model_state_dict': qat_model.state_dict(),
                            **({k: ckpt[k] for k in ('c1_out','c2_out','c3_out','c4_out')} if is_pruned else {})},
                           qat_path)

            history.append({'epoch': epoch, 'train_acc': float(train_acc), 'val_acc': float(val_acc)})
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
                      f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
                      + ("  ← best" if val_acc >= best_val_acc else ""))

        print(f"\n  Best val_acc: {best_val_acc:.4f}")
        with open(os.path.join(args.output_dir, 'qat_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    # ---- Load best QAT model ----
    print(f"\n{'='*60}")
    print(f"  Phase 2: INT8 Conversion + Calibration")
    print(f"{'='*60}")

    qat_ckpt = torch.load(qat_path, map_location=device, weights_only=False)
    if is_pruned:
        qat_model = ECG_1DCNN_QAT(
            c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
            c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
        ).to(device)
    else:
        qat_model = ECG_1DCNN_QAT().to(device)

    qat_model.load_state_dict(qat_ckpt['model_state_dict'])
    qat_model.eval()

    w_int8, b_int8, w_shift, nb, input_shift = convert_to_int8(
        qat_model, train_loader, device, n_cal_batches=20
    )

    print(f"\n  input_shift_bits = {input_shift}  (scale = 2^{input_shift})")
    print(f"  Per-layer weight shift bits:")
    for name, n in w_shift.items():
        print(f"    {name:10s}  shift={n}  (scale=2^{n})")
    print(f"  Per-layer activation nb (output right-shift):")
    for name, n in nb.items():
        print(f"    {name:10s}  nb={n}")

    # Save INT8 checkpoint
    int8_ckpt = {
        'model_state_dict': qat_model.state_dict(),
        'quantization': 'QAT-INT8',
        'w_int8': {k: v.tolist() for k, v in w_int8.items()},
        'b_int8': {k: v.tolist() for k, v in b_int8.items()},
        'w_shift': w_shift,
        'nb': nb,
        'input_shift_bits': input_shift,
    }
    if is_pruned:
        for key in ('c1_out', 'c2_out', 'c3_out', 'c4_out'):
            int8_ckpt[key] = ckpt[key]

    int8_path = os.path.join(args.output_dir, 'model_qat_int8.pth')
    torch.save(int8_ckpt, int8_path)
    print(f"\n  Saved INT8 checkpoint: {int8_path}")

    # ---- Phase 3: Evaluate ----
    print(f"\n{'='*60}")
    print(f"  Phase 3: Evaluation")
    print(f"{'='*60}")

    # Float32 (QAT weights, no fake quantize)
    qat_model.eval()
    with torch.no_grad():
        preds_f, labels_f = [], []
        for batch in test_loader:
            xf = batch[0].to(device)
            yf = batch[1]
            preds_f.extend(qat_model(xf, quantize=False).argmax(1).cpu().numpy())
            labels_f.extend(yf.numpy())
    preds_f   = np.array(preds_f)
    labels_f  = np.array(labels_f)
    float_acc = (preds_f == labels_f).mean()
    print(f"\n  QAT Float32 test accuracy : {float_acc:.4f} ({float_acc*100:.2f}%)")

    # INT8 simulation
    int8_acc, preds_int8, labels_int8 = evaluate_int8(
        qat_model, test_loader, w_int8, b_int8, w_shift, nb, input_shift, device
    )
    print(f"  QAT INT8 test accuracy    : {int8_acc:.4f} ({int8_acc*100:.2f}%)")
    print(f"  Accuracy drop (float→INT8): {(float_acc - int8_acc)*100:+.2f}%")

    metrics_int8 = compute_metrics(preds_int8, labels_int8, CLASS_NAMES)
    print(f"\n  INT8 Per-class metrics:")
    print_classification_report(metrics_int8)

    # ---- Phase 4: Export .mem ----
    print(f"\n{'='*60}")
    print(f"  Phase 4: Export .mem files")
    print(f"{'='*60}")
    export_mem_files(w_int8, b_int8, w_shift, nb, input_shift, args.output_dir)


def main():
    p = argparse.ArgumentParser(
        description='QAT training + INT8 conversion + evaluate + export'
    )
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Float32 checkpoint to start QAT from')
    p.add_argument('--output_dir', type=str, default='./results/qat_int8')
    p.add_argument('--data_dir',   type=str, default='/home/duc/Thesis/data/Chapman')
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--eval_only',  action='store_true',
                   help='Skip QAT training, only convert + evaluate + export')
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
