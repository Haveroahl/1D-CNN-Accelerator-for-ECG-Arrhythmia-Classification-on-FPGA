"""Export full test sets as INT8 ECG + labels for on-board demo (HPS driver).

For the DE10 demo: HPS streams each sample into the accelerator, reads result[1:0],
counts matches vs label, shows accuracy % on LEDs. This needs ONLY (quantized ECG,
label) — NOT golden intermediate stages (those are for bit-exact sim, a separate goal).

Quantization matches the RTL input path exactly:
    ecg_int8 = clamp(round(x * 2^input_shift), -127, 127)
so the .bin holds precisely the bytes HPS writes into input_sram. Both sets use the
SAME Chapman INT8 weights (PTB-XL = zero-shot, C3) — RTL bitstream unchanged.

Output (per set) — little flat binaries, HPS reads with fread:
    <set>_ecg_int8.bin   N * 2500 bytes, signed int8, row-major (sample-major)
    <set>_labels.bin     N bytes, uint8 label 0..3
    <set>_demo_meta.json N, expected accuracy, class distribution

Usage:
    python export_test_demo.py \
        --checkpoint ./results/qat_int8/model_qat_int8.pth \
        --out_dir ../../hardware/fpga/soc/demo_data
"""
import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantization.qat_int8 import ECG_1DCNN_QAT, int8_forward
from utils.dataset import get_dataloaders, CLASS_NAMES


def quantize_int8(x_float, input_shift):
    """x_float (N,2500) → int8 (N,2500) matching RTL input path."""
    xi = np.clip(np.round(x_float * (2.0 ** input_shift)), -127, 127)
    return xi.astype(np.int8)


def predict_all(model, x_float, w8, b8, ws, nb, ins, device, batch=256):
    preds = []
    with torch.no_grad():
        for i in range(0, x_float.shape[0], batch):
            xb = torch.from_numpy(x_float[i:i+batch].astype(np.float32)).to(device)
            preds.append(int8_forward(model, xb, w8, b8, ws, nb, ins).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def write_set(name, x_float, labels, model, qp, out_dir):
    w8, b8, ws, nb, ins, device = qp
    ecg_int8 = quantize_int8(x_float, ins)               # (N, 2500) int8
    labels = labels.astype(np.uint8)

    ecg_path = os.path.join(out_dir, f"{name}_ecg_int8.bin")
    lbl_path = os.path.join(out_dir, f"{name}_labels.bin")
    ecg_int8.tofile(ecg_path)                            # row-major, sample-major
    labels.tofile(lbl_path)

    preds = predict_all(model, x_float, w8, b8, ws, nb, ins, device)
    acc = float((preds == labels).mean())
    dist = {CLASS_NAMES[c]: int((labels == c).sum()) for c in range(4)}

    meta = {
        'set': name,
        'n_samples': int(x_float.shape[0]),
        'sample_len': 2500,
        'input_shift_bits': int(ins),
        'expected_accuracy': acc,
        'class_distribution': dist,
        'ecg_file': os.path.basename(ecg_path),
        'labels_file': os.path.basename(lbl_path),
        'format': 'ecg: N*2500 signed int8 row-major; labels: N uint8',
    }
    with open(os.path.join(out_dir, f"{name}_demo_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  [{name}] N={x_float.shape[0]}  expected_acc={acc*100:.2f}%  "
          f"dist={dist}")
    print(f"    {os.path.basename(ecg_path)} ({ecg_int8.nbytes} bytes), "
          f"{os.path.basename(lbl_path)} ({labels.nbytes} bytes)")


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cpu')

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = ECG_1DCNN_QAT(c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
                          c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    qp = (
        {k: np.array(v, dtype=np.int8)    for k, v in ckpt['w_int8'].items()},
        {k: np.array(v, dtype=np.float64) for k, v in ckpt['b_int8'].items()},
        ckpt['w_shift'], ckpt['nb'], ckpt['input_shift_bits'], device,
    )

    print(f"\n[INFO] Exporting demo test sets to {args.out_dir}/")

    if args.chapman:
        _, _, test_loader = get_dataloaders(args.chapman_dir, batch_size=256, num_workers=0)
        cx, cy = [], []
        for b in test_loader:
            cx.append(b[0].numpy()); cy.append(b[1].numpy())
        write_set('chapman_test', np.concatenate(cx), np.concatenate(cy),
                  model, qp, args.out_dir)

    if args.ptbxl:
        d = np.load(args.ptbxl_npz)
        write_set('ptbxl_test', d['X_test'], d['y_test'], model, qp, args.out_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--chapman_dir', default='../../data/Chapman')
    p.add_argument('--ptbxl_npz', default='../../data/ptbxl_processed/ptbxl_dataset.npz')
    p.add_argument('--out_dir', default='../../hardware/fpga/soc/demo_data')
    p.add_argument('--no-chapman', dest='chapman', action='store_false')
    p.add_argument('--no-ptbxl', dest='ptbxl', action='store_false')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
