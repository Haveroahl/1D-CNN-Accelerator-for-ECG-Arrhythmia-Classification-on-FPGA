"""General-Scale INT8 QAT (A3) + Floor variant (A4) — Ablation for Table 4 (C1).

Two variants controlled by --rescale-mode:
  round  (A3): out = clamp(round(acc * s_out / (s_in * s_w)), -127, 127)
               hardware rescale needs 1 multiplier per layer → ~1 DSP18 per rescale
  floor  (A4-gen): floor variant of general-scale (drop round correction)

Compare against power-of-2 QAT (qat_int8.py, A2) for Table 4.

Key differences from qat_int8.py (A2):
  - FakeQuantize uses float scale s = abs_max / 127  (not power-of-2 aligned)
  - convert_to_int8_general: w_scale = abs_max / 127, x_scale = abs_max / 127
  - int8_forward_general: rescale = acc * (x_scale_out / (x_scale_in * w_scale))
    using float multiply → not implementable with barrel-shift alone
  - DSP estimate: each layer rescale needs 1 multiplier → ~5 DSP18 extra vs A2

Usage:
    python quantization/qat_int8_general.py \\
        --checkpoint ./results/best_model_pruned.pth \\
        --output_dir ./results/ablation_quant/a3_general \\
        --data_dir   D:/Thesis101/data/Chapman \\
        --rescale-mode round

    python quantization/qat_int8_general.py \\
        --checkpoint ./results/best_model_pruned.pth \\
        --output_dir ./results/ablation_quant/a4_gen_floor \\
        --data_dir   D:/Thesis101/data/Chapman \\
        --rescale-mode floor
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
from utils.evaluate import compute_metrics, print_classification_report


LAYER_ORDER = ['conv1', 'conv2', 'conv3', 'conv4', 'fc']
CONV_LAYERS = ['conv1', 'conv2', 'conv3', 'conv4']


# ============================================================
#  FakeQuantize — general float scale (EMA)
# ============================================================

class FakeQuantizeGeneral(nn.Module):
    """Per-tensor fake quantization with float scale s = abs_max / qmax.

    Identical EMA update to qat_int8.py FakeQuantize, but scale is a general
    float (not constrained to power-of-2).  Both A3 and A4 use this.
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
        x_q = (x / scale).round().clamp(-self.qmax, self.qmax) * scale
        return x + (x_q - x).detach()


# ============================================================
#  QAT Model (reuse identical architecture, different FQ class)
# ============================================================

class ECG_1DCNN_QAT_General(nn.Module):
    """Same topology as ECG_1DCNN_QAT but with general-scale FakeQuantize."""

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

        self.fq_w1  = FakeQuantizeGeneral()
        self.fq_w2  = FakeQuantizeGeneral()
        self.fq_w3  = FakeQuantizeGeneral()
        self.fq_w4  = FakeQuantizeGeneral()
        self.fq_wfc = FakeQuantizeGeneral()

        self.fq_in  = FakeQuantizeGeneral()
        self.fq_a1  = FakeQuantizeGeneral()
        self.fq_a2  = FakeQuantizeGeneral()
        self.fq_a3  = FakeQuantizeGeneral()
        self.fq_a4  = FakeQuantizeGeneral()
        self.fq_gap = FakeQuantizeGeneral()

    def forward(self, x, quantize=True):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if not quantize:
            x = self.pool1(F.conv1d(x, self.conv1.weight, self.conv1.bias, padding=2))
            x = self.pool2(F.conv1d(x, self.conv2.weight, self.conv2.bias, padding=2))
            x = self.pool3(F.conv1d(x, self.conv3.weight, self.conv3.bias, padding=2))
            x = self.pool4(F.relu(F.conv1d(x, self.conv4.weight, self.conv4.bias, padding=2)))
            x = self.gap(x).squeeze(-1)
            return F.linear(x, self.fc.weight, self.fc.bias)

        x = self.fq_a1(self.pool1(F.conv1d(
            self.fq_in(x), self.fq_w1(self.conv1.weight), self.conv1.bias, padding=2
        )))
        x = self.fq_a2(self.pool2(F.conv1d(
            x, self.fq_w2(self.conv2.weight), self.conv2.bias, padding=2
        )))
        x = self.fq_a3(self.pool3(F.conv1d(
            x, self.fq_w3(self.conv3.weight), self.conv3.bias, padding=2
        )))
        x = self.pool4(F.relu(self.fq_a4(F.conv1d(
            x, self.fq_w4(self.conv4.weight), self.conv4.bias, padding=2
        ))))
        x = self.gap(x).squeeze(-1)
        x = self.fq_gap(x)
        return F.linear(x, self.fq_wfc(self.fc.weight), self.fc.bias)


def build_qat_model(base_model):
    if isinstance(base_model, ECG_1DCNN_Pruned):
        qat = ECG_1DCNN_QAT_General(
            c1_out=base_model.c1_out, c2_out=base_model.c2_out,
            c3_out=base_model.c3_out, c4_out=base_model.c4_out,
        )
    else:
        qat = ECG_1DCNN_QAT_General()

    with torch.no_grad():
        for name in LAYER_ORDER:
            src = getattr(base_model, name)
            dst = getattr(qat, name)
            dst.weight.copy_(src.weight)
            if src.bias is not None:
                dst.bias.copy_(src.bias)

    return qat


# ============================================================
#  INT8 Conversion — general float scale
# ============================================================

def convert_to_int8_general(qat_model, train_loader, device, n_cal_batches=20):
    """Convert general-scale QAT model to INT8.

    Scale per layer: s = abs_max / 127  (general float, not power-of-2).
    Rescale multiplier per layer: s_out = s_in_prev * s_w  (accumulated scales).

    Returns:
        w_int8:    dict layer → int8 numpy array
        b_float:   dict layer → float32 numpy array (unscaled bias)
        w_scale:   dict layer → float  (weight scale: abs_max / 127)
        x_scale:   dict layer → float  (input activation scale at that layer)
        x_scale_out: dict conv_layer → float  (output activation scale)
        input_scale: float
    """
    qat_model.eval()

    # Weight scales
    w_scale = {}
    w_int8  = {}
    b_float = {}

    for name in LAYER_ORDER:
        layer = getattr(qat_model, name)
        w_np = layer.weight.data.cpu().numpy()
        abs_max = max(abs(w_np.min()), abs(w_np.max()))
        abs_max = max(abs_max, 1e-8)
        s = abs_max / 127.0
        w_scale[name] = s
        w_int8[name] = np.clip(np.round(w_np / s), -127, 127).astype(np.int8)
        if layer.bias is not None:
            b_float[name] = layer.bias.data.cpu().numpy()

    # Input activation scale from dataset
    max_input = 0.0
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= n_cal_batches:
                break
            v = batch[0].abs().max().item()
            if v > max_input:
                max_input = v
    max_input = max(max_input, 1e-8)
    input_scale = max_input / 127.0

    # Activation output scales — use the EMA scale learned by FakeQuantize modules
    # fq_a1 scale = output of conv1+pool1 = s_in * s_w1 * rescale_factor
    # We read directly from the trained FQ buffers (already in float range)
    fq_map = {
        'conv1': qat_model.fq_a1.scale.item(),
        'conv2': qat_model.fq_a2.scale.item(),
        'conv3': qat_model.fq_a3.scale.item(),
        'conv4': qat_model.fq_a4.scale.item(),
    }
    # x_scale: scale of the input tensor going into each conv
    # conv1 input = raw ECG → scale = input_scale
    # conv2 input = output of conv1+pool → scale = fq_a1 * 127 ≈ abs_max_a1/127 * 127 = abs_max_a1
    # But we want s (float per-value scale = abs_max/127) not abs_max itself
    x_scale_in = {
        'conv1': input_scale,
        'conv2': fq_map['conv1'],
        'conv3': fq_map['conv2'],
        'conv4': fq_map['conv3'],
    }
    x_scale_out = {name: fq_map[name] for name in CONV_LAYERS}

    return w_int8, b_float, w_scale, x_scale_in, x_scale_out, input_scale


# ============================================================
#  INT8 Simulation — general-scale rescale
# ============================================================

def int8_forward_general(qat_model, x, w_int8, b_float, w_scale, x_scale_in, x_scale_out,
                          input_scale, rescale_mode='round'):
    """Simulate hardware INT8 forward with general-scale rescale.

    Rescale formula:
      s_acc = s_x * s_w     (scale of accumulator in float domain)
      s_out = x_scale_out   (desired output scale)
      rescale_factor = s_acc / s_out  = (s_x_in * s_w) / s_out
      out_int = clamp(rescale(acc_int + bias_scaled), -127, 127)

    rescale (round mode, A3):  round(x * rescale_factor)
    rescale (floor mode, A4g): floor(x * rescale_factor)

    Hardware cost note:
      rescale_factor is a general float → requires 1 multiplier (DSP18) per layer.
      A2 power-of-2 uses barrel shift only → 0 DSP for rescale.
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)

    device = next(qat_model.parameters()).device

    x = torch.clamp(torch.round(x / input_scale), -127, 127)

    def rescale_fn(val, factor):
        # out_int = acc_int * (s_acc / s_out); factor = s_acc / s_out (< 1 here).
        # Hardware applies this float multiplier → 1 DSP18 per layer.
        if rescale_mode == 'floor':
            return torch.floor(val * factor)
        else:
            return torch.round(val * factor)

    def conv_layer(x, name, s_x_in, s_out):
        s_w = w_scale[name]
        s_acc = s_x_in * s_w
        rescale_factor = s_acc / s_out

        w = torch.tensor(w_int8[name].astype(np.float32)).to(device)
        layer = getattr(qat_model, name)

        # Bias scaled to accumulator domain: b_int = round(b_float / s_acc)
        # After rescale by s_acc/s_out: b_out_int = round(b_float / s_out) — correct
        b_scaled = torch.tensor(
            np.round(b_float[name] / s_acc).astype(np.float32)
        ).to(device)

        acc = F.conv1d(x, w, b_scaled, padding=layer.padding)
        return torch.clamp(rescale_fn(acc, rescale_factor), -127, 127)

    x = qat_model.pool1(conv_layer(x, 'conv1', x_scale_in['conv1'], x_scale_out['conv1']))
    x = qat_model.pool2(conv_layer(x, 'conv2', x_scale_in['conv2'], x_scale_out['conv2']))
    x = qat_model.pool3(conv_layer(x, 'conv3', x_scale_in['conv3'], x_scale_out['conv3']))

    # conv4: rescale → clamp → ReLU → pool
    s_x4 = x_scale_in['conv4']
    s_w4 = w_scale['conv4']
    s_acc4 = s_x4 * s_w4
    s_out4 = x_scale_out['conv4']
    rescale_factor4 = s_acc4 / s_out4

    w4 = torch.tensor(w_int8['conv4'].astype(np.float32)).to(device)
    b4_scaled = torch.tensor(
        np.round(b_float['conv4'] / s_acc4).astype(np.float32)
    ).to(device)
    acc4 = F.conv1d(x, w4, b4_scaled, padding=qat_model.conv4.padding)
    x = torch.clamp(rescale_fn(acc4, rescale_factor4), -127, 127)
    x = torch.clamp(x, min=0)
    x = qat_model.pool4(x)

    x = qat_model.gap(x).squeeze(-1)

    # FC: output is raw logits for argmax — no rescale needed (argmax is scale-invariant)
    w_fc = torch.tensor(w_int8['fc'].astype(np.float32)).to(device)
    s_fc = w_scale['fc']
    # x is INT8 after GAP clamp; FC output = x_int * w_int * (s_x_gap * s_fc)
    # argmax only → scale factor irrelevant; bias added at float scale (b_float)
    b_fc = torch.tensor(np.round(b_float['fc']).astype(np.float32)).to(device)
    return F.linear(x, w_fc, b_fc)


def evaluate_int8_general(qat_model, loader, w_int8, b_float, w_scale,
                           x_scale_in, x_scale_out, input_scale, rescale_mode, device):
    qat_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1]
            logits = int8_forward_general(
                qat_model, x, w_int8, b_float, w_scale,
                x_scale_in, x_scale_out, input_scale, rescale_mode
            )
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    return (preds == labels).mean(), preds, labels


# ============================================================
#  DSP estimate helper (for Table 4 comparison)
# ============================================================

def estimate_dsp(rescale_mode):
    """Return DSP18 count estimate for rescale operations.

    Power-of-2 (A2): 0 DSP per rescale (barrel shift + adder only).
    General-scale (A3/A4g): 1 DSP per conv rescale = 4 DSP for conv1-4.
    FC has no rescale (argmax scale-invariant).
    """
    if rescale_mode in ('round', 'floor'):
        return 4  # 1 per conv layer, general-scale multiply
    return 0


# ============================================================
#  Main pipeline
# ============================================================

def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Rescale mode: {args.rescale_mode}  "
          f"({'A3 general-scale round' if args.rescale_mode == 'round' else 'A4 general-scale floor'})")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    is_pruned = 'c1_out' in ckpt
    is_qat_ckpt = ckpt.get('quantization') == 'QAT-General-INT8'

    if args.eval_only and is_qat_ckpt:
        print(f"[INFO] Loading saved QAT-General checkpoint for eval-only")
        if is_pruned:
            qat_model = ECG_1DCNN_QAT_General(
                c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
                c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
            ).to(device)
        else:
            qat_model = ECG_1DCNN_QAT_General().to(device)
        qat_model.load_state_dict(ckpt['model_state_dict'])
        qat_model.eval()
    else:
        if is_pruned:
            print(f"[INFO] Pruned model (c1={ckpt['c1_out']}, c2={ckpt['c2_out']}, "
                  f"c3={ckpt['c3_out']}, c4={ckpt['c4_out']})")
            base_model = ECG_1DCNN_Pruned(
                c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
                c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
            )
        else:
            base_model = ECG_1DCNN(num_classes=4)
        base_model.load_state_dict(ckpt['model_state_dict'])
        base_model = base_model.to(device)
        qat_model = None

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=2
    )

    # ---- Phase 1: QAT Training ----
    qat_path = os.path.join(args.output_dir, 'model_qat_general_float.pth')

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
                pv, lv = [], []
                for batch in val_loader:
                    pv.extend(qat_model(batch[0].to(device), quantize=True).argmax(1).cpu().numpy())
                    lv.extend(batch[1].numpy())
            val_acc = (np.array(pv) == np.array(lv)).mean()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_dict = {'model_state_dict': qat_model.state_dict()}
                if is_pruned:
                    save_dict.update({k: ckpt[k] for k in ('c1_out','c2_out','c3_out','c4_out')})
                torch.save(save_dict, qat_path)

            history.append({'epoch': epoch, 'train_acc': float(train_acc), 'val_acc': float(val_acc)})
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
                      f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
                      + ("  <- best" if val_acc >= best_val_acc else ""))

        print(f"\n  Best val_acc: {best_val_acc:.4f}")
        with open(os.path.join(args.output_dir, 'qat_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    # ---- Load best QAT model ----
    if qat_model is None or (args.eval_only and not is_qat_ckpt):
        qat_ckpt = torch.load(qat_path, map_location=device, weights_only=False)
        if is_pruned:
            qat_model = ECG_1DCNN_QAT_General(
                c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
                c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
            ).to(device)
        else:
            qat_model = ECG_1DCNN_QAT_General().to(device)
        qat_model.load_state_dict(qat_ckpt['model_state_dict'])
        qat_model.eval()

    # ---- Phase 2: INT8 Conversion ----
    print(f"\n{'='*60}")
    print(f"  Phase 2: General-Scale INT8 Conversion")
    print(f"{'='*60}")

    w_int8, b_float_dict, w_scale, x_scale_in, x_scale_out, input_scale = \
        convert_to_int8_general(qat_model, train_loader, device, n_cal_batches=20)

    print(f"\n  input_scale = {input_scale:.6f}  (abs_max/127)")
    print(f"  Per-layer weight scales (float, not power-of-2):")
    for name, s in w_scale.items():
        print(f"    {name:10s}  s_w={s:.6f}")
    print(f"  Per-layer activation output scales:")
    for name, s in x_scale_out.items():
        print(f"    {name:10s}  s_out={s:.6f}")
    dsp_extra = estimate_dsp(args.rescale_mode)
    print(f"\n  DSP18 for rescale (estimate): +{dsp_extra} vs power-of-2 A2 (0 extra)")

    # Save checkpoint
    int8_ckpt = {
        'model_state_dict': qat_model.state_dict(),
        'quantization': 'QAT-General-INT8',
        'rescale_mode': args.rescale_mode,
        'w_scale': {k: float(v) for k, v in w_scale.items()},
        'x_scale_in': {k: float(v) for k, v in x_scale_in.items()},
        'x_scale_out': {k: float(v) for k, v in x_scale_out.items()},
        'input_scale': float(input_scale),
        'dsp_extra_rescale': dsp_extra,
    }
    if is_pruned:
        for key in ('c1_out', 'c2_out', 'c3_out', 'c4_out'):
            int8_ckpt[key] = ckpt[key]

    int8_path = os.path.join(args.output_dir, 'model_qat_general_int8.pth')
    torch.save(int8_ckpt, int8_path)
    print(f"\n  Saved: {int8_path}")

    # ---- Phase 3: Evaluate ----
    print(f"\n{'='*60}")
    print(f"  Phase 3: Evaluation  (rescale_mode={args.rescale_mode})")
    print(f"{'='*60}")

    # QAT fake-quantized accuracy
    qat_model.eval()
    pv, lv = [], []
    with torch.no_grad():
        for batch in test_loader:
            pv.extend(qat_model(batch[0].to(device), quantize=True).argmax(1).cpu().numpy())
            lv.extend(batch[1].numpy())
    fq_acc = (np.array(pv) == np.array(lv)).mean()
    print(f"\n  QAT fake-quantized accuracy : {fq_acc:.4f} ({fq_acc*100:.2f}%)")

    # INT8 simulation accuracy
    int8_acc, preds_int8, labels_int8 = evaluate_int8_general(
        qat_model, test_loader, w_int8, b_float_dict, w_scale,
        x_scale_in, x_scale_out, input_scale, args.rescale_mode, device
    )
    print(f"  INT8 simulated accuracy     : {int8_acc:.4f} ({int8_acc*100:.2f}%)")
    print(f"  Accuracy drop (fq→INT8)     : {(fq_acc - int8_acc)*100:+.2f}%")

    metrics = compute_metrics(preds_int8, labels_int8, CLASS_NAMES)
    print(f"\n  INT8 Per-class metrics:")
    print_classification_report(metrics)

    # Save results summary
    results = {
        'variant': f"A3_general_{args.rescale_mode}",
        'rescale_mode': args.rescale_mode,
        'fq_acc': float(fq_acc),
        'int8_acc': float(int8_acc),
        'acc_drop_pct': float((fq_acc - int8_acc) * 100),
        'dsp_extra_rescale': dsp_extra,
        'f1_macro': float(metrics['f1_macro']),
        'per_class_f1': {k: float(v['f1']) for k, v in metrics['per_class'].items()},
        'w_scale': {k: float(v) for k, v in w_scale.items()},
        'input_scale': float(input_scale),
        'note': ('General-scale rescale needs multiply per layer; '
                 'hardware cost = 1 DSP18 per conv rescale stage'),
    }
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    print(f"\n{'='*60}")
    print(f"  Table 4 row for this variant:")
    print(f"  Variant          : {'A3 General-scale (round)' if args.rescale_mode == 'round' else 'A4 General-scale (floor)'}")
    print(f"  Acc (INT8 sim)   : {int8_acc*100:.2f}%")
    print(f"  F1-macro         : {metrics['f1_macro']:.4f}")
    print(f"  DSP extra (rescale): +{dsp_extra} DSP18 vs power-of-2")
    print(f"{'='*60}")


def main():
    p = argparse.ArgumentParser(
        description='General-scale INT8 QAT (A3) + floor variant (A4-gen) for Table 4 ablation'
    )
    p.add_argument('--checkpoint',    type=str, required=True,
                   help='Float32 pruned checkpoint')
    p.add_argument('--output_dir',    type=str, default='./results/ablation_quant/a3_general')
    p.add_argument('--data_dir',      type=str, default='../../data/Chapman')
    p.add_argument('--epochs',        type=int, default=50)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--batch_size',    type=int, default=128)
    p.add_argument('--rescale-mode',  type=str, default='round', choices=['round', 'floor'],
                   dest='rescale_mode',
                   help='round=A3 general-scale, floor=A4-gen floor (both use general float scale)')
    p.add_argument('--eval_only',     action='store_true',
                   help='Skip training, load qat_general_float.pth and eval only')
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
