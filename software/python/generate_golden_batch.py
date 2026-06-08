"""Generate golden reference files for a batch of samples (RTL verification).

Produces, for each selected sample:
  golden/<group><N>/{input_int8,after_pool1..4,after_gap,logits_fc}.mem + golden_meta.json
  ecg_<group><N>.hex   (= input_int8, the INT8 ECG the TB loads over Avalon)

Two groups, BOTH run through the SAME Chapman INT8 weights (model_qat_int8.pth):
  - chapman : 10 samples from the Chapman test split, class-balanced
  - ptbxl   : 10 samples from the PTB-XL test split (zero-shot, C2), class-balanced

PTB-XL inputs come pre-processed (2500 samples, lead II, 250 Hz, z-scored) from
ptbxl_dataset.npz — same domain as Chapman, so the same input_shift applies.

Read-only on weights; writes golden + hex only. Reuses int8_forward_golden so the
pipeline (incl. FC bias scaled by 2^w_shift[fc]) is bit-identical to single-sample.

Usage:
    python generate_golden_batch.py \
        --checkpoint ./results/qat_int8/model_qat_int8.pth \
        --chapman_dir ../../data/Chapman \
        --ptbxl_npz ../../data/ptbxl_processed/ptbxl_dataset.npz \
        --out_root ../../hardware/fpga/simulation/questa/golden \
        --hex_root ../../hardware/fpga/simulation/questa \
        --per_group 10
"""
import os
import sys
import json
import argparse
import shutil

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantization.qat_int8 import ECG_1DCNN_QAT
from utils.dataset import get_dataloaders, CLASS_NAMES
from generate_golden import (
    int8_forward_golden, write_int8_mem, write_int32_mem, to_hex2,
)

NUM_CLASSES = 4


def balanced_indices(labels, n_total, seed=42):
    """Pick n_total indices spread across the 4 classes (round-robin per class)."""
    rng = np.random.RandomState(seed)
    by_class = {c: rng.permutation(np.where(labels == c)[0]).tolist()
                for c in range(NUM_CLASSES)}
    picked = []
    c = 0
    while len(picked) < n_total:
        pool = by_class[c % NUM_CLASSES]
        if pool:
            picked.append(pool.pop(0))
        c += 1
        if c > n_total * NUM_CLASSES * 4:  # safety against empty classes
            break
    return picked[:n_total]


def dump_sample(stages, true_label, group, n, out_root, hex_root):
    sample_dir = os.path.join(out_root, f"{group}{n}")
    os.makedirs(sample_dir, exist_ok=True)

    stage_info = [
        ('input_int8',  'input_int8.mem',  'int8'),
        ('after_pool1', 'after_pool1.mem', 'int8'),
        ('after_pool2', 'after_pool2.mem', 'int8'),
        ('after_pool3', 'after_pool3.mem', 'int8'),
        ('after_pool4', 'after_pool4.mem', 'int8'),
        ('after_gap',   'after_gap.mem',   'int8'),
        ('logits_fc',   'logits_fc.mem',   'int32'),
    ]
    for key, fname, dtype in stage_info:
        path = os.path.join(sample_dir, fname)
        (write_int8_mem if dtype == 'int8' else write_int32_mem)(path, stages[key])

    # ecg hex (the TB loads this over Avalon; identical to input_int8)
    hex_path = os.path.join(hex_root, f"ecg_{group}{n}.hex")
    arr = stages['input_int8'].cpu().numpy().astype(np.int32).flatten()
    with open(hex_path, 'w') as f:
        for v in arr:
            f.write(to_hex2(v) + "\n")

    meta = {
        'group': group, 'index_in_group': n,
        'true_class': int(true_label),
        'predicted_class': int(stages['predicted_class']),
        'correct': int(stages['predicted_class']) == int(true_label),
    }
    with open(os.path.join(sample_dir, 'golden_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def run(args):
    device = torch.device('cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = ECG_1DCNN_QAT(c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
                          c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    w8 = {k: np.array(v, dtype=np.int8)    for k, v in ckpt['w_int8'].items()}
    b8 = {k: np.array(v, dtype=np.float64) for k, v in ckpt['b_int8'].items()}
    nb, w_shift, ins = ckpt['nb'], ckpt['w_shift'], ckpt['input_shift_bits']

    summary = {'chapman': [], 'ptbxl': []}

    # ── Chapman test split ──────────────────────────────────────────────
    _, _, test_loader = get_dataloaders(args.chapman_dir, batch_size=256, num_workers=0)
    cx, cy = [], []
    for b in test_loader:
        cx.append(b[0]); cy.append(b[1])
    cx = torch.cat(cx, 0); cy = torch.cat(cy, 0).numpy()
    c_idx = balanced_indices(cy, args.per_group, seed=args.seed)
    print(f"\n[INFO] Chapman picked idx: {c_idx}")
    for n, idx in enumerate(c_idx):
        with torch.no_grad():
            st = int8_forward_golden(model, cx[idx].unsqueeze(0), w8, b8, nb, w_shift, ins, device)
        m = dump_sample(st, cy[idx], 'chapman', n, args.out_root, args.hex_root)
        print(f"  chapman{n}: src_idx={idx} true={CLASS_NAMES[m['true_class']]} "
              f"pred={CLASS_NAMES[m['predicted_class']]} {'OK' if m['correct'] else 'WRONG'}")
        summary['chapman'].append({**m, 'src_idx': int(idx)})

    # ── PTB-XL test split (zero-shot, Chapman weights) ──────────────────
    d = np.load(args.ptbxl_npz)
    px, py = d['X_test'], d['y_test']
    p_idx = balanced_indices(py, args.per_group, seed=args.seed)
    print(f"\n[INFO] PTB-XL picked idx: {p_idx}")
    for n, idx in enumerate(p_idx):
        x = torch.from_numpy(px[idx].astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            st = int8_forward_golden(model, x, w8, b8, nb, w_shift, ins, device)
        m = dump_sample(st, py[idx], 'ptbxl', n, args.out_root, args.hex_root)
        print(f"  ptbxl{n}: src_idx={idx} true={CLASS_NAMES[m['true_class']]} "
              f"pred={CLASS_NAMES[m['predicted_class']]} {'OK' if m['correct'] else 'WRONG'}")
        summary['ptbxl'].append({**m, 'src_idx': int(idx)})

    with open(os.path.join(args.out_root, 'batch_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    c_acc = np.mean([s['correct'] for s in summary['chapman']])
    p_acc = np.mean([s['correct'] for s in summary['ptbxl']])
    print(f"\n[SUMMARY] Chapman {len(c_idx)} samples acc={c_acc*100:.0f}%  "
          f"PTB-XL {len(p_idx)} samples acc={p_acc*100:.0f}% (zero-shot)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--chapman_dir', default='../../data/Chapman')
    p.add_argument('--ptbxl_npz', default='../../data/ptbxl_processed/ptbxl_dataset.npz')
    p.add_argument('--out_root', default='../../hardware/fpga/simulation/questa/golden')
    p.add_argument('--hex_root', default='../../hardware/fpga/simulation/questa')
    p.add_argument('--per_group', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
