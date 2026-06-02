"""Power-of-2 QAT with Floor Truncation (A4) — Ablation for Table 4 (C1).

Identical to qat_int8.py (A2 power-of-2 round-half-up) except:
  - Rescale uses floor truncation: acc >> nb  (drop the +2^(nb-1) correction)
  - Purpose: quantify how much round-half-up matters vs naive shift (RQ2)

Hardware note: floor is cheaper than round-half-up (omit adder before shift).
Accuracy drop A2→A4 directly answers: "is round-half-up correction necessary?"

Usage:
    python quantization/qat_int8_floor.py \\
        --checkpoint ./results/best_model_pruned.pth \\
        --output_dir ./results/ablation_quant/a4_p2_floor \\
        --data_dir   D:/Thesis101/data/Chapman
"""

import os
import sys
import argparse
import json
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

# Reuse QAT model and conversion from A2 — only int8_forward differs
from qat_int8 import (
    FakeQuantize, ECG_1DCNN_QAT, build_qat_model,
    compute_shift_bits, convert_to_int8,
)

LAYER_ORDER = ['conv1', 'conv2', 'conv3', 'conv4', 'fc']
CONV_LAYERS = ['conv1', 'conv2', 'conv3', 'conv4']


def floor_shift(x, n):
    """Floor arithmetic right shift: matches simple >> n without round correction.
    y = floor(x / 2^n)   for n > 0
    y = x                 for n == 0
    """
    if n > 0:
        return torch.floor(x / (2.0 ** n))
    return x


def int8_forward_floor(qat_model, x, w_int8, b_int8, w_shift, nb, input_shift):
    """Same as int8_forward in A2 but uses floor_shift instead of round_shift."""
    if x.dim() == 2:
        x = x.unsqueeze(1)

    device = next(qat_model.parameters()).device
    x = torch.clamp(torch.round(x * (2.0 ** input_shift)), -127, 127)

    def conv_int8_layer(x, layer_name):
        w = torch.tensor(w_int8[layer_name].astype(np.float32)).to(device)
        b_float = b_int8[layer_name]
        layer = getattr(qat_model, layer_name)
        n = nb[layer_name]
        b_scaled = torch.tensor(np.round(b_float * (2.0 ** n)).astype(np.float32)).to(device)
        out = F.conv1d(x, w, b_scaled, padding=layer.padding)
        return torch.clamp(floor_shift(out, n), -127, 127)

    x = qat_model.pool1(conv_int8_layer(x, 'conv1'))
    x = qat_model.pool2(conv_int8_layer(x, 'conv2'))
    x = qat_model.pool3(conv_int8_layer(x, 'conv3'))

    w4 = torch.tensor(w_int8['conv4'].astype(np.float32)).to(device)
    b_float4 = b_int8['conv4']
    n4 = nb['conv4']
    b4_scaled = torch.tensor(np.round(b_float4 * (2.0 ** n4)).astype(np.float32)).to(device)
    x = torch.clamp(floor_shift(F.conv1d(x, w4, b4_scaled, padding=qat_model.conv4.padding), n4), -127, 127)
    x = torch.clamp(x, min=0)
    x = qat_model.pool4(x)

    x = qat_model.gap(x).squeeze(-1)

    w_fc = torch.tensor(w_int8['fc'].astype(np.float32)).to(device)
    b_float_fc = b_int8['fc']
    b_fc_scaled = torch.tensor(np.round(b_float_fc).astype(np.float32)).to(device)
    return F.linear(x, w_fc, b_fc_scaled)


def evaluate_int8_floor(qat_model, test_loader, w_int8, b_int8, w_shift, nb, input_shift, device):
    qat_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            y = batch[1]
            logits = int8_forward_floor(qat_model, x, w_int8, b_int8, w_shift, nb, input_shift)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    return (preds == labels).mean(), preds, labels


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Variant: A4 power-of-2 floor (no round-half-up correction)")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    is_pruned = 'c1_out' in ckpt
    is_qat_ckpt = ckpt.get('quantization') in ('QAT-INT8', 'QAT-Floor-INT8')

    if args.eval_only and is_qat_ckpt:
        if is_pruned:
            qat_model = ECG_1DCNN_QAT(
                c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
                c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
            ).to(device)
        else:
            qat_model = ECG_1DCNN_QAT().to(device)
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

    # ---- Phase 1: QAT Training (same as A2 — FakeQuantize is scale-agnostic) ----
    qat_path = os.path.join(args.output_dir, 'model_qat_float.pth')

    if not args.eval_only:
        print(f"\n{'='*60}")
        print(f"  Phase 1: QAT Training  ({args.epochs} epochs, lr={args.lr})")
        print(f"  Note: QAT training identical to A2 — floor only affects INT8 sim")
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
            qat_model = ECG_1DCNN_QAT(
                c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
                c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
            ).to(device)
        else:
            qat_model = ECG_1DCNN_QAT().to(device)
        qat_model.load_state_dict(qat_ckpt['model_state_dict'])
        qat_model.eval()

    # ---- Phase 2: INT8 Conversion (power-of-2 shifts, same as A2) ----
    print(f"\n{'='*60}")
    print(f"  Phase 2: Power-of-2 INT8 Conversion (floor rescale)")
    print(f"{'='*60}")

    w_int8, b_int8, w_shift, nb, input_shift = convert_to_int8(
        qat_model, train_loader, device, n_cal_batches=20
    )

    print(f"\n  input_shift_bits = {input_shift}")
    for name, n in w_shift.items():
        print(f"    {name:10s}  w_shift={n}  nb={nb.get(name, 'N/A')}")

    int8_ckpt = {
        'model_state_dict': qat_model.state_dict(),
        'quantization': 'QAT-Floor-INT8',
        'w_int8': {k: v.tolist() for k, v in w_int8.items()},
        'b_int8': {k: v.tolist() for k, v in b_int8.items()},
        'w_shift': w_shift,
        'nb': nb,
        'input_shift_bits': input_shift,
    }
    if is_pruned:
        for key in ('c1_out', 'c2_out', 'c3_out', 'c4_out'):
            int8_ckpt[key] = ckpt[key]

    int8_path = os.path.join(args.output_dir, 'model_qat_floor_int8.pth')
    torch.save(int8_ckpt, int8_path)
    print(f"  Saved: {int8_path}")

    # ---- Phase 3: Evaluate ----
    print(f"\n{'='*60}")
    print(f"  Phase 3: Evaluation")
    print(f"{'='*60}")

    # QAT fake-quantized (same as A2 — training was identical)
    qat_model.eval()
    pv, lv = [], []
    with torch.no_grad():
        for batch in test_loader:
            pv.extend(qat_model(batch[0].to(device), quantize=True).argmax(1).cpu().numpy())
            lv.extend(batch[1].numpy())
    fq_acc = (np.array(pv) == np.array(lv)).mean()
    print(f"\n  QAT fake-quantized accuracy   : {fq_acc:.4f} ({fq_acc*100:.2f}%)")

    # A4 floor INT8 simulation
    floor_acc, preds_floor, labels_floor = evaluate_int8_floor(
        qat_model, test_loader, w_int8, b_int8, w_shift, nb, input_shift, device
    )
    print(f"  A4 floor INT8 accuracy        : {floor_acc:.4f} ({floor_acc*100:.2f}%)")
    print(f"  Accuracy drop vs A2 (round)   : compare manually with A2 result")
    print(f"  Accuracy drop (fq→floor INT8) : {(fq_acc - floor_acc)*100:+.2f}%")

    metrics = compute_metrics(preds_floor, labels_floor, CLASS_NAMES)
    print(f"\n  Per-class metrics (floor):")
    print_classification_report(metrics)

    results = {
        'variant': 'A4_p2_floor',
        'rescale': 'power-of-2 shift, floor truncation (no round-half-up)',
        'fq_acc': float(fq_acc),
        'int8_acc': float(floor_acc),
        'acc_drop_pct': float((fq_acc - floor_acc) * 100),
        'dsp_extra_rescale': 0,
        'f1_macro': float(metrics['f1_macro']),
        'per_class_f1': {k: float(v['f1']) for k, v in metrics['per_class'].items()},
        'nb': nb,
        'w_shift': w_shift,
        'input_shift_bits': int(input_shift),
        'note': 'Power-of-2 scale same as A2, but floor truncation instead of round-half-up. '
                'Measures rounding correction contribution to accuracy.',
    }
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    print(f"\n{'='*60}")
    print(f"  Table 4 row for A4 (power-of-2 floor):")
    print(f"  Acc (INT8 sim): {floor_acc*100:.2f}%")
    print(f"  F1-macro      : {metrics['f1_macro']:.4f}")
    print(f"  DSP extra     : 0 (same as A2, shift only)")
    print(f"{'='*60}")


def main():
    p = argparse.ArgumentParser(
        description='A4: Power-of-2 QAT with floor truncation (ablation for Table 4)'
    )
    p.add_argument('--checkpoint',  type=str, required=True)
    p.add_argument('--output_dir',  type=str, default='./results/ablation_quant/a4_p2_floor')
    p.add_argument('--data_dir',    type=str, default='../../data/Chapman')
    p.add_argument('--epochs',      type=int, default=50)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--batch_size',  type=int, default=128)
    p.add_argument('--eval_only',   action='store_true')
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
