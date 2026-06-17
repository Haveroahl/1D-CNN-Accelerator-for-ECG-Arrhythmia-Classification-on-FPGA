"""qat_ptbxl.py — power-of-2 QAT-INT8 for the PTB-XL fine-tuned model (Phase A C4).

The cross-dataset Phase A produced `ptbxl_finetuned.pth` (a FLOAT state_dict, eval
acc ~0.9329). To deploy it on the FPGA over the JTAG weight-reload path we need the
SAME power-of-2 INT8 pipeline as Chapman: QAT fine-tune -> convert to INT8
(power-of-2 scale, round-half-up) -> export. This script reuses the proven QAT
helpers from qat_int8.py but feeds the PTB-XL npz and inits from the C4 weights.

Output: results/qat_int8_ptbxl/model_qat_int8.pth  (same format as Chapman's, so
export_weights_int8.py works unchanged).

Usage:
  python quantization/qat_ptbxl.py \
    --ckpt   results/cross_eval/ptbxl_finetuned.pth \
    --npz    ../../data/ptbxl_processed/ptbxl_dataset.npz \
    --output results/qat_int8_ptbxl --epochs 10
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # software/python
sys.path.insert(0, HERE)                     # quantization/

from qat_int8 import (ECG_1DCNN_QAT, convert_to_int8, evaluate_int8,
                      compute_metrics)
try:
    from qat_int8 import CLASS_NAMES
except Exception:
    CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']


def npz_loaders(npz_path, batch_size=128):
    d = np.load(npz_path)
    def make(X, y, shuffle):
        # X is (N, 2500) float32 → (N, 1, 2500) for Conv1d
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                          shuffle=shuffle, num_workers=0)
    return (make(d['X_train'], d['y_train'], True),
            make(d['X_val'],   d['y_val'],   False),
            make(d['X_test'],  d['y_test'],  False))


def load_c4_into_qat(ckpt_path, device):
    """Load the raw float C4 state_dict into an ECG_1DCNN_QAT (4,4,8,8)."""
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in sd:        # tolerate either format
        sd = sd['model_state_dict']
    model = ECG_1DCNN_QAT(c1_out=4, c2_out=4, c3_out=8, c4_out=8).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing={missing} unexpected={unexpected}")
    return model


def eval_fq(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x.to(device), quantize=True).argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    return (np.array(preds) == np.array(labels)).mean()


def run(args):
    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device={device}")

    train_loader, val_loader, test_loader = npz_loaders(args.npz, args.batch_size)
    model = load_c4_into_qat(args.ckpt, device)
    print(f"[INFO] init fake-quant acc (pre-QAT) val={eval_fq(model, val_loader, device):.4f}")

    # ── QAT fine-tune (fake-quant in the loop) ──────────────────────────────
    if args.epochs > 0:
        opt = optim.Adam(model.parameters(), lr=args.lr)
        crit = nn.CrossEntropyLoss()
        best_val, best_sd = 0.0, None
        for ep in range(args.epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(model(x, quantize=True), y)
                loss.backward()
                opt.step()
            va = eval_fq(model, val_loader, device)
            if va >= best_val:
                best_val = va
                best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  epoch {ep+1:2d}/{args.epochs}  val_fq={va:.4f}"
                  + ("  <- best" if va >= best_val else ""))
        if best_sd is not None:
            model.load_state_dict(best_sd)
        print(f"[INFO] best QAT fake-quant val={best_val:.4f}")

    # ── Convert to INT8 (power-of-2, round-half-up) ─────────────────────────
    model.eval()
    w_int8, b_int8, w_shift, nb, input_shift = convert_to_int8(
        model, train_loader, device, n_cal_batches=20)
    print(f"[INFO] input_shift={input_shift}  w_shift={w_shift}  nb={nb}")

    int8_ckpt = {
        'model_state_dict': model.state_dict(),
        'quantization': 'QAT-INT8',
        'w_int8': {k: v.tolist() for k, v in w_int8.items()},
        'b_int8': {k: v.tolist() for k, v in b_int8.items()},
        'w_shift': w_shift, 'nb': nb, 'input_shift_bits': input_shift,
        'c1_out': 4, 'c2_out': 4, 'c3_out': 8, 'c4_out': 8,
    }
    out_path = os.path.join(args.output, 'model_qat_int8.pth')
    torch.save(int8_ckpt, out_path)
    print(f"[INFO] saved {out_path}")

    # ── Eval INT8 (hardware integer forward) ────────────────────────────────
    int8_acc, preds, labels = evaluate_int8(
        model, test_loader, w_int8, b_int8, w_shift, nb, input_shift, device)
    print(f"[INFO] PTB-XL INT8 test acc = {int8_acc:.4f} ({int8_acc*100:.2f}%)")
    metrics = compute_metrics(preds, labels, CLASS_NAMES)
    with open(os.path.join(args.output, 'ptbxl_int8_metrics.json'), 'w') as f:
        json.dump({'int8_acc': float(int8_acc), 'nb': nb, 'w_shift': w_shift,
                   'input_shift_bits': int(input_shift)}, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',   default='results/cross_eval/ptbxl_finetuned.pth')
    p.add_argument('--npz',    default='../../data/ptbxl_processed/ptbxl_dataset.npz')
    p.add_argument('--output', default='results/qat_int8_ptbxl')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr',     type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=128)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
