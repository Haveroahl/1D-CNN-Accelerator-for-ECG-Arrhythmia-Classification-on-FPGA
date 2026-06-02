"""General-scale PTQ (A0') — completes the QAT-vs-PTQ x scale-type 2x2 matrix.

           | Power-of-2          | General-scale
  ---------|---------------------|----------------------
  PTQ      | A0  (ptq_int8.py)   | A0' (this file)
  QAT      | A2  (qat_int8.py)   | A3  (qat_int8_general.py)

Purpose: isolate what QAT actually buys. With all 4 cells we can decompose:
  - QAT gain   = (A2 - A0)  and  (A3 - A0')      [holding scale-type fixed]
  - scale gain = (A0' - A0) and  (A3 - A2)       [holding train-method fixed]

This is PTQ with a GENERAL float scale (s = abs_max/127), NO fake-quant training.
Unlike A3, activation output scales cannot be read from trained FakeQuantize EMA
buffers — there is no training. Instead we CALIBRATE them with a float forward
pass over the train set: s_out[layer] = abs_max(conv+pool output) / 127.

Reuses int8_forward_general / convert weights from qat_int8_general.py.

Usage:
    cd software/python
    python quantization/ptq_int8_general.py \\
        --checkpoint ./results/best_model_pruned.pth \\
        --output_dir ./results/ablation_quant/a0g_ptq_general \\
        --data_dir   ../../data/Chapman \\
        --rescale-mode round
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from prune_finetune import ECG_1DCNN_Pruned
from model.model import ECG_1DCNN
from utils.dataset import get_dataloaders, CLASS_NAMES
from utils.evaluate import compute_metrics, print_classification_report

from quantization.qat_int8_general import (
    ECG_1DCNN_QAT_General, build_qat_model, int8_forward_general,
    estimate_dsp, LAYER_ORDER, CONV_LAYERS,
)


def calibrate_scales(qat_model, train_loader, device, n_cal_batches=20):
    """Calibrate general-scale params from a FLOAT forward pass (no training).

    - weight scale  : s_w = abs_max(W) / 127  per layer
    - input scale   : s_in = abs_max(ECG) / 127 over calibration batches
    - activation out: s_out[layer] = abs_max(conv+pool float output) / 127

    Returns the same tuple shape as convert_to_int8_general so int8_forward_general
    can consume it directly.
    """
    qat_model.eval()

    # ---- Weight scales + INT8 weights ----
    w_scale, w_int8, b_float = {}, {}, {}
    for name in LAYER_ORDER:
        layer = getattr(qat_model, name)
        w_np = layer.weight.data.cpu().numpy()
        abs_max = max(abs(w_np.min()), abs(w_np.max()), 1e-8)
        s = abs_max / 127.0
        w_scale[name] = s
        w_int8[name] = np.clip(np.round(w_np / s), -127, 127).astype(np.int8)
        if layer.bias is not None:
            b_float[name] = layer.bias.data.cpu().numpy()

    # ---- Input scale + per-layer activation-output abs_max (float forward) ----
    max_input = 0.0
    act_absmax = {name: 0.0 for name in CONV_LAYERS}

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= n_cal_batches:
                break
            x = batch[0].to(device)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            max_input = max(max_input, x.abs().max().item())

            # Float forward, capturing each conv+pool output (matches int8 staging:
            # conv1-3 no ReLU, conv4 ReLU before pool).
            a = qat_model.pool1(F.conv1d(x, qat_model.conv1.weight, qat_model.conv1.bias, padding=2))
            act_absmax['conv1'] = max(act_absmax['conv1'], a.abs().max().item())
            a = qat_model.pool2(F.conv1d(a, qat_model.conv2.weight, qat_model.conv2.bias, padding=2))
            act_absmax['conv2'] = max(act_absmax['conv2'], a.abs().max().item())
            a = qat_model.pool3(F.conv1d(a, qat_model.conv3.weight, qat_model.conv3.bias, padding=2))
            act_absmax['conv3'] = max(act_absmax['conv3'], a.abs().max().item())
            a = qat_model.pool4(F.relu(F.conv1d(a, qat_model.conv4.weight, qat_model.conv4.bias, padding=2)))
            act_absmax['conv4'] = max(act_absmax['conv4'], a.abs().max().item())

    input_scale = max(max_input, 1e-8) / 127.0
    x_scale_out = {name: max(act_absmax[name], 1e-8) / 127.0 for name in CONV_LAYERS}
    x_scale_in = {
        'conv1': input_scale,
        'conv2': x_scale_out['conv1'],
        'conv3': x_scale_out['conv2'],
        'conv4': x_scale_out['conv3'],
    }
    return w_int8, b_float, w_scale, x_scale_in, x_scale_out, input_scale


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] PTQ general-scale (A0'), rescale_mode={args.rescale_mode}, NO training")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    is_pruned = 'c1_out' in ckpt

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
    base_model.eval()

    train_loader, _, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=2
    )

    # Build general QAT shell, copy float weights, NO training
    qat_model = build_qat_model(base_model).to(device)
    qat_model.eval()

    print(f"\n{'='*60}")
    print(f"  Calibrate general scales (float forward, no fine-tune)")
    print(f"{'='*60}")
    w_int8, b_float, w_scale, x_scale_in, x_scale_out, input_scale = \
        calibrate_scales(qat_model, train_loader, device, n_cal_batches=20)
    print(f"  input_scale = {input_scale:.6f}")
    print(f"  x_scale_out = {{ " + ", ".join(f'{k}:{v:.4f}' for k, v in x_scale_out.items()) + " }}")
    dsp_extra = estimate_dsp(args.rescale_mode)

    # ---- Bit-exact INT8 eval (general-scale rescale) ----
    print(f"\n{'='*60}")
    print(f"  Evaluation (int8_forward_general, {args.rescale_mode})")
    print(f"{'='*60}")
    qat_model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            logits = int8_forward_general(
                qat_model, x, w_int8, b_float, w_scale,
                x_scale_in, x_scale_out, input_scale, args.rescale_mode
            )
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(batch[1].numpy())
    preds, labels = np.array(preds), np.array(labels)
    acc = (preds == labels).mean()
    metrics = compute_metrics(preds, labels, CLASS_NAMES)

    print(f"\n  PTQ-general INT8 accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-macro                  : {metrics['f1_macro']:.4f}")
    print_classification_report(metrics)

    results = {
        'variant': f"A0g_ptq_general_{args.rescale_mode}",
        'int8_acc': float(acc),
        'fq_acc': float(acc),
        'acc_drop_pct': 0.0,
        'dsp_extra_rescale': dsp_extra,
        'f1_macro': float(metrics['f1_macro']),
        'per_class_f1': {k: float(v['f1']) for k, v in metrics['per_class'].items()},
        'rescale_mode': args.rescale_mode,
        'note': ('General-scale PTQ (no fake-quant training; activation scales '
                 'calibrated from float forward). Completes the QAT-vs-PTQ x '
                 'scale-type 2x2 matrix with A0/A2/A3.'),
    }
    out_path = os.path.join(args.output_dir, 'results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")
    print(f"\n  Table 4 row — A0' PTQ general-scale:")
    print(f"  Acc {acc*100:.2f}%  F1 {metrics['f1_macro']:.4f}  +{dsp_extra} DSP18")


def main():
    p = argparse.ArgumentParser(description="General-scale PTQ (A0') for QAT-vs-PTQ 2x2 matrix")
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--output_dir', type=str, default="./results/ablation_quant/a0g_ptq_general")
    p.add_argument('--data_dir',   type=str, default='../../data/Chapman')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--rescale-mode', type=str, default='round', choices=['round', 'floor'],
                   dest='rescale_mode')
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
