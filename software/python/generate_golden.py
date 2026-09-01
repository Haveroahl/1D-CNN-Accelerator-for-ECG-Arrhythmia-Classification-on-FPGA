"""Generate golden reference files for RTL verification.

Runs INT8 forward pass on one ECG sample and dumps intermediate
activations at each pipeline stage to .mem files.

Model: ECG_1DCNN Pruned — channels (4, 4, 8, 8)
  Conv1: Cin=1,  Cout=4,  K=5, pad=2, no ReLU → MaxPool(K=5,S=5) → 500×4
  Conv2: Cin=4,  Cout=4,  K=5, pad=2, no ReLU → MaxPool(K=5,S=5) → 100×4
  Conv3: Cin=4,  Cout=8,  K=5, pad=2, no ReLU → MaxPool(K=5,S=5) →  20×8
  Conv4: Cin=8,  Cout=8,  K=5, pad=2, ReLU    → MaxPool(K=5,S=5) →   4×8
  GAP → FC(8→4) → Argmax

RTL pipeline order (cp_block.v):
  acc(INT32) → +bias → >>nb + round_half_up → clamp[-127,127] → ReLU → MaxPool

nb:      Conv1=8, Conv2=6, Conv3=6, Conv4=7
Rounding: round_half_up: (acc + 2^(nb-1)) >> nb

Stages written:
  input_int8   → 2500,     INT8
  after_pool1  → 4×500,    INT8
  after_pool2  → 4×100,    INT8
  after_pool3  → 8×20,     INT8
  after_pool4  → 8×4,      INT8
  after_gap    → 8,        INT8
  logits_fc    → 4,        INT32

Usage:
    python generate_golden.py \\
        --checkpoint ./results/qat_int8/model_qat_int8.pth \\
        --data_dir   D:/Thesis101/data/Chapman \\
        --output_dir ./results/golden \\
        --sample_idx 0
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantization.qat_int8 import round_shift, ECG_1DCNN_QAT
from utils.dataset import get_dataloaders, CLASS_NAMES


def to_hex2(v: int) -> str:
    return f"{int(v) & 0xFF:02X}"

def to_hex8(v: int) -> str:
    return f"{int(v) & 0xFFFFFFFF:08X}"

def write_int8_mem(path, tensor):
    arr = tensor.cpu().numpy().astype(np.int32).flatten()
    with open(path, 'w') as f:
        for v in arr:
            f.write(to_hex2(v) + "\n")

def write_int32_mem(path, tensor):
    arr = tensor.cpu().numpy().astype(np.int64).flatten()
    with open(path, 'w') as f:
        for v in arr:
            f.write(to_hex8(v) + "\n")


def int8_forward_golden(qat_model, x_raw, w_int8, b_int8, nb, w_shift, input_shift, device):
    """INT8 forward matching RTL pipeline: acc → +bias → >>nb → clamp → ReLU → pool."""
    stages = {}

    if x_raw.dim() == 2:
        x_raw = x_raw.unsqueeze(1)
    x_raw = x_raw.to(device)

    x = torch.clamp(torch.round(x_raw * (2.0 ** input_shift)), -127, 127)
    stages['input_int8'] = x.squeeze(0).squeeze(0)   # (2500,)

    def conv_int8(x_in, name, use_relu=False):
        w = torch.tensor(w_int8[name].astype(np.float32)).to(device)
        n = nb[name]
        b_scaled = torch.tensor(
            np.round(b_int8[name] * (2.0 ** n)).astype(np.float32)
        ).to(device)
        layer = getattr(qat_model, name)
        acc = F.conv1d(x_in, w, b_scaled, padding=layer.padding)
        out = torch.clamp(round_shift(acc, n), -127, 127)
        if use_relu:
            out = torch.clamp(out, min=0)
        return out

    # Conv1 → Pool1
    after_conv1 = conv_int8(x, 'conv1')
    after_pool1 = qat_model.pool1(after_conv1)
    stages['after_pool1'] = after_pool1.squeeze(0)       # (4, 500)

    # Conv2 → Pool2
    after_conv2 = conv_int8(after_pool1, 'conv2')
    after_pool2 = qat_model.pool2(after_conv2)
    stages['after_pool2'] = after_pool2.squeeze(0)       # (4, 100)

    # Conv3 → Pool3
    after_conv3 = conv_int8(after_pool2, 'conv3')
    after_pool3 = qat_model.pool3(after_conv3)
    stages['after_pool3'] = after_pool3.squeeze(0)       # (8, 20)

    # Conv4 + ReLU → Pool4  (ReLU after >>nb + clamp, matching RTL S8)
    after_conv4 = conv_int8(after_pool3, 'conv4', use_relu=True)
    after_pool4 = qat_model.pool4(after_conv4)
    stages['after_pool4'] = after_pool4.squeeze(0)       # (8, 4)

    # GAP — RTL uses integer floor division by 4 (= arithmetic shift right by 2
    # on positive values, matching cp_block rescale convention). Python golden
    # must do the same; nn.AdaptiveAvgPool1d returns float which would introduce
    # 0..0.75 fractional LSB that RTL cannot represent.
    gap_sum  = after_pool4.sum(dim=-1)                   # (1, 8) INT8 sums, ∈ [0, 508]
    after_gap = torch.floor(gap_sum / 4.0)               # match RTL [9:2] slice
    stages['after_gap'] = after_gap.squeeze(0)           # (8,)

    # FC — no output rescale, so bias is scaled to the logit domain by
    # 2^w_shift[fc] (commensurate with Σ gap[2^0]·w_fc[2^w_shift]). RTL adds
    # fc_bias.hex (INT32) to fc_acc before argmax — this golden must match.
    w_fc = torch.tensor(w_int8['fc'].astype(np.float32)).to(device)
    fc_shift = w_shift['fc']
    b_fc = torch.tensor(
        np.round(b_int8['fc'] * (2.0 ** fc_shift)).astype(np.float32)).to(device)
    logits = F.linear(after_gap, w_fc, b_fc)
    stages['logits_fc'] = logits.squeeze(0)              # (4,)
    stages['predicted_class'] = int(logits.argmax(dim=1).item())

    return stages


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cpu')

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    is_pruned = 'c1_out' in ckpt
    qat_model = ECG_1DCNN_QAT(
        c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
        c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
    ) if is_pruned else ECG_1DCNN_QAT()
    qat_model.load_state_dict(ckpt['model_state_dict'])
    qat_model.eval()

    w_int8      = {k: np.array(v, dtype=np.int8)    for k, v in ckpt['w_int8'].items()}
    b_int8      = {k: np.array(v, dtype=np.float64) for k, v in ckpt['b_int8'].items()}
    nb          = ckpt['nb']
    w_shift     = ckpt['w_shift']
    input_shift = ckpt['input_shift_bits']

    if getattr(args, 'npz', None):
        from utils.npz_dataset import get_npz_dataloaders
        _, _, test_loader = get_npz_dataloaders(args.npz, batch_size=256, num_workers=0)
    else:
        _, _, test_loader = get_dataloaders(args.data_dir, batch_size=256, num_workers=0)
    all_x, all_y = [], []
    for batch in test_loader:
        all_x.append(batch[0]); all_y.append(batch[1])
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    idx = args.sample_idx
    x_sample   = all_x[idx].unsqueeze(0)
    true_label = int(all_y[idx].item())

    print(f"[INFO] Sample {idx}  true={true_label}({CLASS_NAMES[true_label]})"
          f"  input_shift={input_shift}  nb={nb}")

    with torch.no_grad():
        stages = int8_forward_golden(
            qat_model, x_sample, w_int8, b_int8, nb, w_shift, input_shift, device
        )

    pred = stages['predicted_class']
    print(f"[INFO] Predicted: {pred}({CLASS_NAMES[pred]})  {'OK' if pred == true_label else 'WRONG'}")

    stage_info = [
        ('input_int8',  'input_int8.mem',  'int8'),
        ('after_pool1', 'after_pool1.mem', 'int8'),
        ('after_pool2', 'after_pool2.mem', 'int8'),
        ('after_pool3', 'after_pool3.mem', 'int8'),
        ('after_pool4', 'after_pool4.mem', 'int8'),
        ('after_gap',   'after_gap.mem',   'int8'),
        ('logits_fc',   'logits_fc.mem',   'int32'),
    ]

    print(f"\n[INFO] Writing to {args.output_dir}/")
    for key, fname, dtype in stage_info:
        t = stages[key]
        path = os.path.join(args.output_dir, fname)
        if dtype == 'int8':
            write_int8_mem(path, t)
        else:
            write_int32_mem(path, t)
        print(f"  {fname:<25} shape={list(t.shape)}")

    meta = {
        'sample_idx':       idx,
        'true_class':       true_label,
        'predicted_class':  pred,
        'correct':          pred == true_label,
        'input_shift_bits': input_shift,
        'nb':               nb,
        'stage_files':      {key: fname for key, fname, _ in stage_info},
    }
    with open(os.path.join(args.output_dir, 'golden_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  {'golden_meta.json':<25} (metadata)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--data_dir',    default='../../data/Chapman')
    p.add_argument('--npz',         default=None,
                   help='Pre-split .npz path; overrides --data_dir when set')
    p.add_argument('--output_dir',  default='./results/golden')
    p.add_argument('--sample_idx',  type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
