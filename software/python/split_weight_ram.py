"""
split_weight_ram.py — Phase B01 weight-RAM init generator
=========================================================
Re-pack the legacy per-layer conv weight hex files (conv1_w.hex .. conv4_w.hex)
into the 8 per-oc M10K RAM init files (w_ram0.hex .. w_ram7.hex) expected by the
refactored cp_engine.v.

RAM layout (cp_engine.v):  w_ram<oc>[word], word = layer_base + ic
    Conv1 base=0  (1 word,  ic=0)        Conv2 base=1  (4 words, ic=0..3)
    Conv3 base=5  (4 words, ic=0..3)     Conv4 base=9  (8 words, ic=0..7)  → 17 words

Source hex addressing (export_weights_int8.py):
    conv1_w.hex[oc]            (4  entries, oc=0..3,  ic=0)
    conv2_w.hex[oc*4 + ic]     (16 entries, oc=0..3,  ic=0..3)
    conv3_w.hex[oc*4 + ic]     (32 entries, oc=0..7,  ic=0..3)
    conv4_w.hex[oc*8 + ic]     (64 entries, oc=0..7,  ic=0..7)

Each line is a 40-bit packed 5-tap word (10 hex chars). We just MOVE words into
the new (oc, word) slots — values are byte-identical, so bit-exactness is
preserved. Unused (oc, word) slots are zero-filled.

Usage (from the dir holding conv*_w.hex, e.g. hardware/fpga/simulation/questa):
    python <path>/split_weight_ram.py            # in-place, reads/writes cwd
    python <path>/split_weight_ram.py --dir DIR
"""

import os
import argparse

N_WORDS = 17  # per-oc RAM depth
# (source file, n_oc, n_ic, ic-stride-in-source, layer_base-in-RAM)
LAYERS = [
    ("conv1_w.hex", 4, 1, 1, 0),
    ("conv2_w.hex", 4, 4, 4, 1),
    ("conv3_w.hex", 8, 4, 4, 5),
    ("conv4_w.hex", 8, 8, 8, 9),
]


def read_hex_lines(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="dir containing conv*_w.hex (output goes here too)")
    args = ap.parse_args()
    d = args.dir

    # 8 RAMs × 17 words, default zero
    ram = [["0000000000"] * N_WORDS for _ in range(8)]

    for fname, n_oc, n_ic, stride, base in LAYERS:
        lines = read_hex_lines(os.path.join(d, fname))
        expected = n_oc * n_ic
        if len(lines) != expected:
            raise SystemExit(f"{fname}: expected {expected} lines, got {len(lines)}")
        for oc in range(n_oc):
            for ic in range(n_ic):
                src_addr = oc * stride + ic
                word = base + ic
                ram[oc][word] = lines[src_addr]

    for oc in range(8):
        out = os.path.join(d, f"w_ram{oc}.hex")
        with open(out, "w") as f:
            for w in ram[oc]:
                f.write(w + "\n")
        print(f"wrote {out} ({N_WORDS} words)")


if __name__ == "__main__":
    main()
