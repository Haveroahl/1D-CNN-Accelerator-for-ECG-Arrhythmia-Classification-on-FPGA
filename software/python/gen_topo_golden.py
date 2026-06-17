"""gen_topo_golden.py — synthetic end-to-end golden for arbitrary channel topology.

Purpose: prove the runtime-reconfigurable RTL (cp_engine/cnn_controller/gap_fc)
runs a FULL inference correctly for channel layouts other than the trained
Chapman (1,4,4,8) — specifically the MIN (2,2,2,2) and MAX (8,8,8,8) cases.

There is no trained model for those topologies, so we synthesize random INT8
weights and run the SAME bit-exact INT8 reference pipeline as generate_golden.py
(acc → +bias_scaled → round_half_up >>nb → clamp[-127,127] → ReLU(conv4) → pool →
 GAP floor(/4) → FC). Whatever logits the reference produces ARE the golden — the
only claim is "RTL == this reference", which is exactly what the bit-exact
framework verifies. nb/w_shift are fixed (not calibrated) since values are
synthetic; RTL is told the same nb via config, so the comparison stays bit-exact.

Topology convention (matches RTL):
  ch = (c1,c2,c3,c4) are the OUTPUT channels of Conv1..4.
  in_ch[Conv1]=1 (raw ECG); in_ch[ConvL]=out_ch[ConvL-1] for L>1.
  layer_base[L] = sum of in_ch of earlier layers (0-based cumulative).

Outputs (per topology, into --output_dir/<tag>/):
  w_ram0..7.hex   (8 per-oc RAMs × 32 words × 40b; unused slots zero)
  conv_bias.hex   (32 × INT32, addr=oc*4+layer_idx; scaled 2^nb)
  fc_weights.hex  (32 × INT8, addr=k*8+i; rows for inactive Conv4 ch = 0)
  fc_bias.hex     (4 × INT32, scaled 2^w_shift_fc)
  logits_fc.mem   (4 × INT32 golden logits)
  config.json     (in_ch/cp_en/nb/base per layer + expected argmax)

Usage:
  python gen_topo_golden.py --ecg ../../hardware/fpga/simulation/questa/ecg_sample0.hex \
      --output_dir ../../hardware/testbench/topo_golden
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

N_TAPS = 5
N_WORDS = 32          # RAM depth (matches cp_engine.v w_ram[0:31])
PAD = 2
# Fixed shifts for synthetic weights (RTL gets the same nb via config).
# conv1 nb = input_shift(2) + w_shift(6); conv2..4 nb = w_shift. The ECG hex is
# already INT8 (input_shift baked in at export) — do NOT re-scale the input.
W_SHIFT = {'conv1': 6, 'conv2': 6, 'conv3': 6, 'conv4': 7, 'fc': 8}
NB = {'conv1': 8, 'conv2': 6, 'conv3': 6, 'conv4': 7}   # fc nb = 0


def round_shift(x, n):
    """round-half-up arithmetic shift, signed: (x + 2^(n-1)) >> n via floor."""
    if n == 0:
        return x
    return torch.floor((x + (2.0 ** (n - 1))) / (2.0 ** n))


def read_ecg_hex(path):
    vals = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                v = int(ln, 16)
                if v >= 128:
                    v -= 256
                vals.append(v)
    return np.array(vals, dtype=np.int8)


def gen_topology(ch, seed):
    """Random INT8 weights for output-channel layout ch=(c1,c2,c3,c4)."""
    rng = np.random.default_rng(seed)
    c1, c2, c3, c4 = ch
    in_ch = {'conv1': 1, 'conv2': c1, 'conv3': c2, 'conv4': c3}
    out_ch = {'conv1': c1, 'conv2': c2, 'conv3': c3, 'conv4': c4}
    w = {}
    b = {}
    for name in ['conv1', 'conv2', 'conv3', 'conv4']:
        # small magnitude so accs stay well within INT8 after rescale
        w[name] = rng.integers(-20, 21, size=(out_ch[name], in_ch[name], N_TAPS)).astype(np.int8)
        b[name] = rng.integers(-5, 6, size=(out_ch[name],)).astype(np.int64)
    # FC: 4 outputs × 8 inputs (always 8 wide in HW). Inactive Conv4 channels
    # (>= c4) get zero weight rows so they never contribute (RTL also zeros them
    # via out_ch_mask, but zeroing here keeps the reference self-consistent).
    w['fc'] = np.zeros((4, 8), dtype=np.int8)
    w['fc'][:, :c4] = rng.integers(-30, 31, size=(4, c4)).astype(np.int8)
    b['fc'] = rng.integers(-3, 4, size=(4,)).astype(np.int64)
    return w, b, in_ch, out_ch


def int8_forward(x_int8, w, b, ch):
    # x_int8 is read from ecg_sample*.hex, which is ALREADY the INT8 input the RTL
    # consumes (input_shift was applied at export). Do NOT re-scale — the RTL's
    # input_sram holds these bytes verbatim. (Earlier bug: re-applying ×2^shift
    # double-scaled the input and made the golden diverge from the RTL.)
    c1, c2, c3, c4 = ch
    x = torch.tensor(x_int8, dtype=torch.float32).view(1, 1, -1)

    def conv(x_in, name, relu=False):
        wt = torch.tensor(w[name].astype(np.float32))
        n = NB[name]
        b_scaled = torch.tensor(np.round(b[name] * (2.0 ** n)).astype(np.float32))
        acc = F.conv1d(x_in, wt, b_scaled, padding=PAD)
        out = torch.clamp(round_shift(acc, n), -127, 127)
        if relu:
            out = torch.clamp(out, min=0)
        return out

    a1 = F.max_pool1d(conv(x, 'conv1'), 5, 5)
    a2 = F.max_pool1d(conv(a1, 'conv2'), 5, 5)
    a3 = F.max_pool1d(conv(a2, 'conv3'), 5, 5)
    a4 = F.max_pool1d(conv(a3, 'conv4', relu=True), 5, 5)   # (1, c4, 4)
    int8_forward.a4 = a4.squeeze(0).numpy().astype(np.int64)
    int8_forward.pools = (a1.squeeze(0).numpy().astype(np.int64),
                          a2.squeeze(0).numpy().astype(np.int64),
                          a3.squeeze(0).numpy().astype(np.int64),
                          a4.squeeze(0).numpy().astype(np.int64))
    # GAP over the 4 positions, floor(/4). Pad channels c4..7 with 0 → 8-wide.
    gap = torch.floor(a4.sum(dim=-1) / 4.0).squeeze(0)       # (c4,)
    gap8 = torch.zeros(8)
    gap8[:c4] = gap
    # FC: logit[k] = b_fc[k]*2^fcshift + Σ_i gap8[i]*w_fc[k][i]
    w_fc = torch.tensor(w['fc'].astype(np.float32))
    b_fc = torch.tensor(np.round(b['fc'] * (2.0 ** W_SHIFT['fc'])).astype(np.float32))
    logits = F.linear(gap8, w_fc, b_fc)                      # (4,)
    int8_forward.gap8 = gap8.numpy().astype(np.int64)
    return logits.numpy().astype(np.int64), gap8.numpy().astype(np.int64)


def pack_word(taps):
    word = 0
    for t in range(N_TAPS):
        word |= (int(taps[t]) & 0xFF) << (t * 8)
    return word


def write_topology(out_dir, tag, ch, w, b, in_ch, logits):
    d = os.path.join(out_dir, tag)
    os.makedirs(d, exist_ok=True)
    c1, c2, c3, c4 = ch
    out_ch = {'conv1': c1, 'conv2': c2, 'conv3': c3, 'conv4': c4}
    # layer_base = cumulative in_ch
    order = ['conv1', 'conv2', 'conv3', 'conv4']
    base = {}
    acc = 0
    for name in order:
        base[name] = acc
        acc += in_ch[name]
    if acc > N_WORDS:
        raise SystemExit(f"{tag}: total in_ch {acc} > RAM depth {N_WORDS}")

    # 8 per-oc RAMs × 32 words, zero default
    ram = [[0] * N_WORDS for _ in range(8)]
    for name in order:
        for oc in range(out_ch[name]):
            for ic in range(in_ch[name]):
                ram[oc][base[name] + ic] = pack_word(w[name][oc, ic])
    for oc in range(8):
        with open(os.path.join(d, f'w_ram{oc}.hex'), 'w') as f:
            for word in ram[oc]:
                f.write(f"{word & 0xFFFFFFFFFF:010X}\n")

    # conv bias: 32 × INT32, addr = oc*4 + layer_idx, scaled 2^nb
    bias_arr = [0] * 32
    for li, name in enumerate(order):
        for oc in range(out_ch[name]):
            bias_arr[oc * 4 + li] = int(round(int(b[name][oc]) * (2 ** NB[name])))
    with open(os.path.join(d, 'conv_bias.hex'), 'w') as f:
        for v in bias_arr:
            f.write(f"{v & 0xFFFFFFFF:08X}\n")

    # FC weights: 32 × INT8, addr = k*8 + i
    with open(os.path.join(d, 'fc_weights.hex'), 'w') as f:
        for k in range(4):
            for i in range(8):
                f.write(f"{int(w['fc'][k, i]) & 0xFF:02X}\n")
    # FC bias: 4 × INT32, scaled 2^fcshift
    with open(os.path.join(d, 'fc_bias.hex'), 'w') as f:
        for k in range(4):
            v = int(round(int(b['fc'][k]) * (2 ** W_SHIFT['fc'])))
            f.write(f"{v & 0xFFFFFFFF:08X}\n")

    # golden logits
    with open(os.path.join(d, 'logits_fc.mem'), 'w') as f:
        for v in logits:
            f.write(f"{int(v) & 0xFFFFFFFF:08X}\n")

    # GAP golden (8-wide, inactive channels 0) — direct FC input, well-defined.
    with open(os.path.join(d, 'after_gap.mem'), 'w') as f:
        for v in int8_forward.gap8.flatten():
            f.write(f"{int(v) & 0xFF:02X}\n")

    # intermediate pool stages (channel-major, matching RTL ping bank layout)
    p1, p2, p3, p4 = int8_forward.pools
    for name, arr in [('after_pool1', p1), ('after_pool2', p2),
                      ('after_pool3', p3), ('after_pool4', p4)]:
        with open(os.path.join(d, name + '.mem'), 'w') as f:
            for v in arr.flatten():
                f.write(f"{int(v) & 0xFF:02X}\n")

    cp_en = {name: (1 << out_ch[name]) - 1 for name in order}
    cfg = {
        'tag': tag,
        'out_ch': [c1, c2, c3, c4],
        'in_ch':  {name: in_ch[name] for name in order},
        'cp_en':  {name: cp_en[name] for name in order},
        'nb':     {name: NB[name] for name in order},
        'base':   {name: base[name] for name in order},
        'total_words': acc,
        'expected_argmax': int(np.argmax(logits)),
        'logits': [int(v) for v in logits],
    }
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    # Flat topology file for the Tcl/JTAG driver (no JSON parser in System
    # Console). One line per layer Conv1..4: in_ch cp_en nb base (all decimal).
    with open(os.path.join(d, 'topo.txt'), 'w') as f:
        f.write("# in_ch cp_en nb base (per layer Conv1..4) - driver CONFIG window\n")
        for name in order:
            f.write(f"{in_ch[name]} {cp_en[name]} {NB[name]} {base[name]}\n")
    print(f"[{tag}] out_ch={ch} in_ch={[in_ch[n] for n in order]} "
          f"base={[base[n] for n in order]} total={acc} "
          f"argmax={cfg['expected_argmax']} logits={cfg['logits']}")
    return cfg


def build_coverage_topos():
    """Coverage sweep over the 1..8-per-layer channel-scalable space.

    Exhaustive 8^4=4096 is wasteful (no extra proof vs a boundary+random sweep),
    so we pick a set that GUARANTEES every out_ch value 1..8 appears at every
    layer position, plus monotone / non-monotone / corner shapes, plus a handful
    of fixed-seed random points. Each entry = (tag, (c1,c2,c3,c4), seed).
    Constraint enforced by gen_topology/write_topology: out_ch in 1..8, total
    in_ch (1+c1+c2+c3) <= 32 (always true here).
    """
    topos = []
    seed = 1000
    def add(tag, ch):
        nonlocal seed
        topos.append((tag, ch, seed)); seed += 1

    # Anchors / named cases (keep the historically-meaningful ones).
    add('chapman', (1, 4, 4, 8)) if False else None  # chapman has its own golden
    add('min1111', (1, 1, 1, 1))
    add('min2222', (2, 2, 2, 2))
    add('max8888', (8, 8, 8, 8))

    # Per-layer value sweep: for each layer position L and each value v in 1..8,
    # set layer L = v, the other three layers = 4 (a mid baseline). This makes
    # every value 1..8 appear at every layer position at least once.
    for L in range(4):
        for v in range(1, 9):
            ch = [4, 4, 4, 4]
            ch[L] = v
            add(f'L{L}v{v}', tuple(ch))

    # Monotone increasing / decreasing.
    add('mono_up',   (1, 2, 4, 8))
    add('mono_down', (8, 4, 2, 1))
    add('asc4',      (3, 4, 5, 6))
    add('desc4',     (6, 5, 4, 3))

    # Non-monotone / stress (channel count goes up-down-up etc.).
    add('nm3175', (3, 1, 7, 5))
    add('nm8181', (8, 1, 8, 1))
    add('nm1818', (1, 8, 1, 8))
    add('odd3577', (3, 5, 7, 7))

    # A few fixed-seed random shapes for breadth (deterministic via seed).
    rng = np.random.default_rng(424242)
    for i in range(8):
        ch = tuple(int(x) for x in rng.integers(1, 9, size=4))
        add(f'rand{i}', ch)

    # De-duplicate by channel tuple, keep first tag.
    seen = {}
    uniq = []
    for tag, ch, sd in topos:
        if ch in seen:
            continue
        seen[ch] = tag
        uniq.append((tag, ch, sd))
    return uniq


def run(args):
    x_raw = read_ecg_hex(args.ecg)
    print(f"[INFO] ECG samples={len(x_raw)} from {args.ecg}")
    topos = build_coverage_topos()
    print(f"[INFO] coverage sweep: {len(topos)} unique topologies")

    manifest = []
    for tag, ch, seed in topos:
        w, b, in_ch, out_ch = gen_topology(ch, seed=seed)
        logits, gap = int8_forward(x_raw, w, b, ch)
        cfg = write_topology(args.output_dir, tag, ch, w, b, in_ch, logits)
        # one manifest row the testbench can $fscanf:
        #   tag c1 c2 c3 c4  ic0 ic1 ic2 ic3  ce0 ce1 ce2 ce3  nb0..3  bs0..3  argmax
        order = ['conv1', 'conv2', 'conv3', 'conv4']
        ic = [1, ch[0], ch[1], ch[2]]
        ce = [cfg['cp_en'][n] for n in order]
        nb = [cfg['nb'][n] for n in order]
        bs = [cfg['base'][n] for n in order]
        manifest.append((tag, list(ch), ic, ce, nb, bs, cfg['expected_argmax']))

    mpath = os.path.join(args.output_dir, 'topo_manifest.txt')
    with open(mpath, 'w') as f:
        f.write(f"# {len(manifest)} topologies: tag c1..c4 ic0..3 ce0..3 nb0..3 bs0..3 argmax\n")
        for tag, ch, ic, ce, nb, bs, am in manifest:
            row = [tag] + ch + ic + ce + nb + bs + [am]
            f.write(' '.join(str(x) for x in row) + '\n')
    print(f"[INFO] wrote manifest {mpath} ({len(manifest)} rows)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ecg', required=True)
    p.add_argument('--output_dir', default='../../hardware/testbench/topo_golden')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
