"""gen_ptbxl_golden.py — bit-exact RTL golden for the PTB-XL INT8 model.

Proves (in simulation, no board) that the PTB-XL weights + per-layer nb shipped to
demo_data/ptbxl_weights/ run correctly through the SAME RTL as Chapman, just with
the runtime CONFIG (nb[Conv3]=7). Produces a topo_golden/ptbxl/ directory in the
exact layout tb_topo.v consumes (w_ram*/conv_bias/fc_weights/fc_bias/logits_fc.mem/
after_gap.mem) plus a ptbxl_sample.hex input — so tb_topo's `ptbxl` case can be
compared fc_acc[0..3] vs golden bit-exact.

Reference pipeline is byte-identical to qat_int8.int8_forward / gen_topo_golden:
  x_int8 = clamp(round(x_float * 2^input_shift), -127, 127)
  acc = conv(x_int8, w_int8) + round(b_float * 2^nb)
  out = clamp(round_half_up(acc, nb), -127, 127)   [ReLU after Conv4]
  GAP = floor(sum_4 / 4)   (integer, matches RTL — NOT AdaptiveAvgPool float avg)
  logit = b_fc*2^w_shift_fc + Σ gap * w_fc

Topology is the trained Chapman shape (1,4,4,8); only nb differs (Conv3 6->7).

Usage:
  python gen_ptbxl_golden.py \
    --ckpt results/qat_int8_ptbxl/model_qat_int8.pth \
    --npz  ../../data/ptbxl_processed/ptbxl_dataset.npz \
    --sample 0 \
    --out  ../../hardware/fpga/simulation/questa/topo_golden/ptbxl
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

N_TAPS = 5
N_WORDS = 32
PAD = 2
CH = (4, 4, 8, 8)   # Chapman/PTB-XL pruned shape


def round_shift(x, n):
    if n == 0:
        return x
    return torch.floor((x + (2.0 ** (n - 1))) / (2.0 ** n))


def load_ckpt(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    w = {k: np.array(v, dtype=np.int32) for k, v in ck['w_int8'].items()}
    b = {k: np.array(v, dtype=np.float64) for k, v in ck['b_int8'].items()}
    nb = ck['nb']
    w_shift = ck['w_shift']
    input_shift = ck['input_shift_bits']
    return w, b, nb, w_shift, input_shift


def make_input_int8(npz_path, sample_idx, input_shift):
    d = np.load(npz_path)
    x = d['X_test'][sample_idx].astype(np.float32)              # (2500,)
    x_int8 = np.clip(np.round(x * (2.0 ** input_shift)), -127, 127).astype(np.int64)
    truth = int(d['y_test'][sample_idx])
    return x_int8, truth


def int8_forward(x_int8, w, b, nb, w_shift):
    x = torch.tensor(x_int8, dtype=torch.float32).view(1, 1, -1)

    def conv(x_in, name, relu=False):
        wt = torch.tensor(w[name].astype(np.float32))
        n = nb[name]
        b_scaled = torch.tensor(np.round(b[name] * (2.0 ** n)).astype(np.float32))
        acc = F.conv1d(x_in, wt, b_scaled, padding=PAD)
        out = torch.clamp(round_shift(acc, n), -127, 127)
        if relu:
            out = torch.clamp(out, min=0)
        return out

    a1 = F.max_pool1d(conv(x, 'conv1'), 5, 5)
    a2 = F.max_pool1d(conv(a1, 'conv2'), 5, 5)
    a3 = F.max_pool1d(conv(a2, 'conv3'), 5, 5)
    a4 = F.max_pool1d(conv(a3, 'conv4', relu=True), 5, 5)        # (1, 8, 4)
    pools = (a1.squeeze(0).numpy().astype(np.int64),
             a2.squeeze(0).numpy().astype(np.int64),
             a3.squeeze(0).numpy().astype(np.int64),
             a4.squeeze(0).numpy().astype(np.int64))
    # GAP integer floor(/4) — matches RTL, not float average.
    gap = torch.floor(a4.sum(dim=-1) / 4.0).squeeze(0)           # (8,)
    gap8 = torch.zeros(8)
    gap8[:CH[3]] = gap
    w_fc = torch.tensor(w['fc'].astype(np.float32))
    b_fc = torch.tensor(np.round(b['fc'] * (2.0 ** w_shift['fc'])).astype(np.float32))
    logits = F.linear(gap8, w_fc, b_fc)
    return logits.numpy().astype(np.int64), gap8.numpy().astype(np.int64), pools


def pack_word(taps):
    word = 0
    for t in range(N_TAPS):
        word |= (int(taps[t]) & 0xFF) << (t * 8)
    return word


def write_golden(out, w, b, nb, w_shift, logits, gap8, pools):
    os.makedirs(out, exist_ok=True)
    # in_ch/out_ch/base for the (1,4,4,8) layout
    in_ch  = {'conv1': 1, 'conv2': 4, 'conv3': 4, 'conv4': 8}
    out_ch = {'conv1': 4, 'conv2': 4, 'conv3': 8, 'conv4': 8}
    order = ['conv1', 'conv2', 'conv3', 'conv4']
    base = {}
    acc = 0
    for name in order:
        base[name] = acc
        acc += in_ch[name]

    # 8 per-oc RAMs × 32 words
    ram = [[0] * N_WORDS for _ in range(8)]
    for name in order:
        for oc in range(out_ch[name]):
            for ic in range(in_ch[name]):
                ram[oc][base[name] + ic] = pack_word(w[name][oc, ic])
    for oc in range(8):
        with open(os.path.join(out, f'w_ram{oc}.hex'), 'w') as f:
            for word in ram[oc]:
                f.write(f"{word & 0xFFFFFFFFFF:010X}\n")

    # conv bias: 32 × INT32, addr = oc*4 + layer_idx, scaled 2^nb
    bias_arr = [0] * 32
    for li, name in enumerate(order):
        for oc in range(out_ch[name]):
            bias_arr[oc * 4 + li] = int(round(float(b[name][oc]) * (2 ** nb[name])))
    with open(os.path.join(out, 'conv_bias.hex'), 'w') as f:
        for v in bias_arr:
            f.write(f"{v & 0xFFFFFFFF:08X}\n")

    # FC weights: 32 × INT8, addr = k*8 + i
    with open(os.path.join(out, 'fc_weights.hex'), 'w') as f:
        for k in range(4):
            for i in range(8):
                v = int(w['fc'][k, i]) if i < w['fc'].shape[1] else 0
                f.write(f"{v & 0xFF:02X}\n")
    # FC bias: 4 × INT32, scaled 2^w_shift_fc
    with open(os.path.join(out, 'fc_bias.hex'), 'w') as f:
        for k in range(4):
            v = int(round(float(b['fc'][k]) * (2 ** w_shift['fc'])))
            f.write(f"{v & 0xFFFFFFFF:08X}\n")

    with open(os.path.join(out, 'logits_fc.mem'), 'w') as f:
        for v in logits:
            f.write(f"{int(v) & 0xFFFFFFFF:08X}\n")
    with open(os.path.join(out, 'after_gap.mem'), 'w') as f:
        for v in gap8.flatten():
            f.write(f"{int(v) & 0xFF:02X}\n")

    p1, p2, p3, p4 = pools
    for name, arr in [('after_pool1', p1), ('after_pool2', p2),
                      ('after_pool3', p3), ('after_pool4', p4)]:
        with open(os.path.join(out, name + '.mem'), 'w') as f:
            for v in arr.flatten():
                f.write(f"{int(v) & 0xFF:02X}\n")

    cp_en = {name: (1 << out_ch[name]) - 1 for name in order}
    cfg = {
        'tag': 'ptbxl', 'out_ch': list(CH),
        'in_ch': in_ch, 'cp_en': cp_en, 'nb': {n: nb[n] for n in order},
        'base': base, 'expected_argmax': int(np.argmax(logits)),
        'logits': [int(v) for v in logits],
    }
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    return cfg, base, cp_en


def run(args):
    w, b, nb, w_shift, input_shift = load_ckpt(args.ckpt)
    x_int8, truth = make_input_int8(args.npz, args.sample, input_shift)
    logits, gap8, pools = int8_forward(x_int8, w, b, nb, w_shift)
    cfg, base, cp_en = write_golden(args.out, w, b, nb, w_shift, logits, gap8, pools)

    # input hex for the testbench (one INT8 byte per line, 2500 lines)
    with open(os.path.join(args.out, 'ptbxl_sample.hex'), 'w') as f:
        for v in x_int8:
            f.write(f"{int(v) & 0xFF:02X}\n")

    print(f"[ptbxl] sample={args.sample} truth={truth} "
          f"argmax={cfg['expected_argmax']} logits={cfg['logits']}")
    print(f"  nb={cfg['nb']}  base={base}  cp_en={ {k: hex(v) for k, v in cp_en.items()} }")
    print(f"  -> {args.out}")
    print(f"  TB run_topology args: in_ch=(1,4,4,8) cp_en=(0F,0F,FF,FF) "
          f"nb=({nb['conv1']},{nb['conv2']},{nb['conv3']},{nb['conv4']}) "
          f"base=({base['conv1']},{base['conv2']},{base['conv3']},{base['conv4']})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='results/qat_int8_ptbxl/model_qat_int8.pth')
    p.add_argument('--npz',  default='../../data/ptbxl_processed/ptbxl_dataset.npz')
    p.add_argument('--sample', type=int, default=0)
    p.add_argument('--out', default='../../hardware/fpga/simulation/questa/topo_golden/ptbxl')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
