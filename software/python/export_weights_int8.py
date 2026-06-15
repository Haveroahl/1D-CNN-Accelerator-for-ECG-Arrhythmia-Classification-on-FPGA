"""
INT8 Weight Export (fphys method) for Verilog ROM
==================================================
Converts INT8 quantized checkpoint to .mem format for FPGA.

INT8 Format (fphys)
-------------------
  Weights : Signed 8-bit integer [-128, 127]
             w_int8 = clamp(round(w_float / scale), -128, 127)
             scale  = |w|_max / 127  (per layer)
  Activation rescaling: round-half-up arithmetic right shift by nb bits
             nb_l = ceil(log2(max_dataset |O_l| / 127))
             Hardware: (acc + (1 << (nb-1))) >>> nb  (round-half-up, signed)

Output Files (Verilog RTL $readmemh format)
-----------
  Essential for RTL:
    - conv1_w.hex       (4  × 40b, addr = oc)
    - conv2_w.hex       (16 × 40b, addr = oc*4 + ic)
    - conv3_w.hex       (32 × 40b, addr = oc*4 + ic)
    - conv4_w.hex       (64 × 40b, addr = oc*8 + ic)
    - conv_bias.hex     (32 × INT32, addr = oc*4 + layer_idx)
    - fc_weights.hex    (32 × INT8, addr = k*8 + i)
    - nb_shifts.mem     (per-layer activation shift bits)

  Metadata & reference:
    - weights_summary.json
    - input_scale.txt
    - rom_address_map.txt

Usage
-----
  python export_weights_int8.py \\
      --checkpoint ./results/qat_int8/model_qat_int8.pth \\
      --output_dir ./results/weights_qat_int8

  Then in Verilog (cp_engine.v):
      $readmemh("conv1_w.hex",   w_rom_conv1);
      $readmemh("conv_bias.hex", b_store);
      // gap_fc_argmax.v: $readmemh("fc_weights.hex", fc_w);
"""

import os
import sys
import argparse
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.model import ECG_1DCNN
from prune_finetune import ECG_1DCNN_Pruned
from quantization.qat_int8 import ECG_1DCNN_QAT


# ============================================================
#  INT8 conversion helpers
# ============================================================

def float_to_int8(v: float) -> int:
    """Convert a single float to INT8 signed integer."""
    return max(-128, min(127, int(np.round(v))))


def array_to_int8(arr: np.ndarray) -> np.ndarray:
    """
    Vectorised float array → int8 array.
    Clips values to [-128, 127].
    """
    rounded = np.round(arr).astype(np.int32)
    clipped = np.clip(rounded, -128, 127)
    return clipped.astype(np.int8)


def int8_to_hex2(val: int) -> str:
    """
    Format a signed int8 as 2-character uppercase hex (two's complement).
      1   →  01
     -1   →  FF
    -128  →  80
    """
    return f"{val & 0xFF:02X}"


def int32_to_hex8(val: int) -> str:
    """Format a signed int32 as 8-character uppercase hex (two's complement)."""
    return f"{val & 0xFFFFFFFF:08X}"


def nb_to_hex2(val: int) -> str:
    """Format nb shift value as 2-character hex."""
    return f"{val & 0xFF:02X}"


# ============================================================
#  RTL hex file export (matches $readmemh layout in RTL)
# ============================================================

def export_rtl_hex(w_int8_ckpt: dict, b_float_ckpt: dict, nb_dict: dict,
                   w_shift_dict: dict, output_dir: str):
    """
    Export hex files matching RTL $readmemh layout (cp_engine.v packed ROM format):

      conv1_w.hex — w_rom_conv1[0:3]    addr = oc           (4 entries × 40b, 5 taps packed)
      conv2_w.hex — w_rom_conv2[0:15]   addr = oc*4 + ic    (16 entries × 40b)
      conv3_w.hex — w_rom_conv3[0:31]   addr = oc*4 + ic    (32 entries × 40b)
      conv4_w.hex — w_rom_conv4[0:63]   addr = oc*8 + ic    (64 entries × 40b)
      conv_bias.hex — b_store[oc*4 + layer_idx]              (32 entries × INT32)
      fc_weights.hex — fc_w[k*8 + i]                         (32 entries × INT8)
      fc_bias.hex   — fc_b[k], scaled by 2^w_shift[fc]       (4 entries × INT32)

    Packed word layout (40-bit):  {tap4, tap3, tap2, tap1, tap0}  (MSB to LSB)
    Each tap = 8-bit two's complement, line = 10 hex chars.

    Unused slots (oc >= out_ch, ic >= in_ch) filled with zero.
    """
    # Layer geometry (must match RTL cp_engine.v ROM sizes)
    LAYER_CFG = [
        ('conv1', 4, 1, 4),    # name, OC, IC, n_entries
        ('conv2', 4, 4, 16),
        ('conv3', 8, 4, 32),
        ('conv4', 8, 8, 64),
    ]
    N_TAPS = 5

    # ---- 4 packed weight ROMs per layer ----
    for lname, n_oc, n_ic, n_entries in LAYER_CFG:
        entries = [0] * n_entries  # default zero
        if lname in w_int8_ckpt:
            w = np.array(w_int8_ckpt[lname], dtype=np.int32)  # shape (out_ch, in_ch, K)
            for oc in range(n_oc):
                for ic in range(n_ic):
                    addr = oc * n_ic + ic
                    if oc < w.shape[0] and ic < w.shape[1]:
                        # Pack 5 taps into 40-bit: tap4 high, tap0 low
                        word = 0
                        for tap in range(N_TAPS):
                            byte = int(w[oc, ic, tap]) & 0xFF
                            word |= byte << (tap * 8)
                        entries[addr] = word

        path = os.path.join(output_dir, f"{lname}_w.hex")
        with open(path, 'w') as f:
            for v in entries:
                f.write(f"{v & 0xFFFFFFFFFF:010X}\n")
        print(f"\n  {lname}_w.hex       ({n_entries} entries × 40b, packed 5-tap, layout [oc*{n_ic}+ic])")

    # ---- conv_bias.hex: b_store[oc*4 + layer_idx], INT32 each, 32 entries ----
    # Layout in RTL: addr = oc*4 + layer_idx (oc=0..7, layer=0..3)
    CONV_LAYERS = ['conv1', 'conv2', 'conv3', 'conv4']
    N_OC_BIAS = 8
    b_entries = []
    for oc in range(N_OC_BIAS):
        for lname in CONV_LAYERS:
            nb = nb_dict.get(lname, 0)
            if lname in b_float_ckpt:
                b_float = np.array(b_float_ckpt[lname], dtype=np.float64)
                if oc < b_float.shape[0]:
                    val = int(np.clip(np.round(b_float[oc] * (2 ** nb)), -(2**31), 2**31 - 1))
                else:
                    val = 0
            else:
                val = 0
            b_entries.append(val)

    conv_b_path = os.path.join(output_dir, 'conv_bias.hex')
    with open(conv_b_path, 'w') as f:
        for v in b_entries:
            f.write(f"{v & 0xFFFFFFFF:08X}\n")
    print(f"  conv_bias.hex      ({len(b_entries)} INT32 entries, layout [oc*4+layer_idx])")

    # ---- fc_weights.hex: fc_w[k*8 + i], 4×8 INT8 = 32 entries ----
    fc_entries = []
    if 'fc' in w_int8_ckpt:
        w_fc = np.array(w_int8_ckpt['fc'], dtype=np.int32)
        for k in range(4):
            for i in range(8):
                val = int(w_fc[k, i]) if (k < w_fc.shape[0] and i < w_fc.shape[1]) else 0
                fc_entries.append(val)
    else:
        fc_entries = [0] * 32

    fc_path = os.path.join(output_dir, 'fc_weights.hex')
    with open(fc_path, 'w') as f:
        for v in fc_entries:
            f.write(f"{v & 0xFF:02X}\n")
    print(f"  fc_weights.hex     ({len(fc_entries)} INT8 entries, layout [k*8+i])")

    # ---- fc_bias.hex: fc_b[k], INT32, 4 entries ----
    # FC has no output rescale, so bias is scaled to the logit domain by
    # 2^w_shift[fc] (commensurate with Σ gap[2^0]·w_fc[2^w_shift]). Added to
    # fc_acc INT32 before argmax in gap_fc_argmax.v.
    fc_shift = w_shift_dict.get('fc', 0)
    fc_b_entries = [0, 0, 0, 0]
    if 'fc' in b_float_ckpt:
        b_fc = np.array(b_float_ckpt['fc'], dtype=np.float64)
        for k in range(4):
            if k < b_fc.shape[0]:
                fc_b_entries[k] = int(np.clip(
                    np.round(b_fc[k] * (2 ** fc_shift)), -(2**31), 2**31 - 1))

    fc_b_path = os.path.join(output_dir, 'fc_bias.hex')
    with open(fc_b_path, 'w') as f:
        for v in fc_b_entries:
            f.write(f"{v & 0xFFFFFFFF:08X}\n")
    print(f"  fc_bias.hex        (4 INT32 entries, scaled 2^{fc_shift}=2^w_shift[fc]: {fc_b_entries})")

    # ---- w_ram0..7.hex: per-oc M10K init (Phase B01 weight RAM) ----
    # Re-pack the 4 conv ROMs into 8 per-oc RAMs of 17 words: word = layer_base+ic.
    #   Conv1 base=0 (ic=0)  Conv2 base=1 (ic0..3)  Conv3 base=5  Conv4 base=9
    # Byte-identical 40-bit entries — just moved into (oc, word) slots.
    export_weight_ram(w_int8_ckpt, output_dir)


def export_weight_ram(w_int8_ckpt: dict, output_dir: str):
    """Write w_ram0..7.hex (8 per-oc RAMs × 17 words × 40-bit) for cp_engine.v."""
    N_WORDS, N_TAPS = 17, 5
    # (name, n_oc, n_ic, layer_base)
    LAYERS = [('conv1', 4, 1, 0), ('conv2', 4, 4, 1),
              ('conv3', 8, 4, 5), ('conv4', 8, 8, 9)]
    ram = [[0] * N_WORDS for _ in range(8)]
    for lname, n_oc, n_ic, base in LAYERS:
        if lname not in w_int8_ckpt:
            continue
        w = np.array(w_int8_ckpt[lname], dtype=np.int32)  # (out_ch, in_ch, K)
        for oc in range(n_oc):
            for ic in range(n_ic):
                if oc < w.shape[0] and ic < w.shape[1]:
                    word = 0
                    for tap in range(N_TAPS):
                        word |= (int(w[oc, ic, tap]) & 0xFF) << (tap * 8)
                    ram[oc][base + ic] = word
    for oc in range(8):
        with open(os.path.join(output_dir, f'w_ram{oc}.hex'), 'w') as f:
            for word in ram[oc]:
                f.write(f"{word & 0xFFFFFFFFFF:010X}\n")
    print(f"  w_ram0..7.hex      (8 per-oc RAMs × {N_WORDS} words × 40b, Phase B01)")


# ============================================================
#  Full export pipeline
# ============================================================

LAYER_ORDER = ['conv1', 'conv2', 'conv3', 'conv4', 'fc']


def export_all(checkpoint_path: str, output_dir: str):
    """
    Load INT8 checkpoint → export weights + per-layer scales to .mem files.

    Writes:
      <output_dir>/conv1_weight.mem
      <output_dir>/conv2_weight.mem
      ...
      <output_dir>/fc_weight.mem
      <output_dir>/flat_weights.mem       ← all INT8 weights concatenated
      <output_dir>/nb_shifts.mem          ← per-layer shift bits (fphys)
      <output_dir>/weights_summary.json
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load checkpoint ----
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    is_qat    = 'w_int8' in ckpt
    is_pruned = 'c1_out' in ckpt
    if is_qat:
        print(f"[INFO] Detected QAT checkpoint "
              f"(c1={ckpt.get('c1_out','?')}, c2={ckpt.get('c2_out','?')}, "
              f"c3={ckpt.get('c3_out','?')}, c4={ckpt.get('c4_out','?')})")
        model = ECG_1DCNN_QAT(
            c1_out=ckpt.get('c1_out', 4), c2_out=ckpt.get('c2_out', 4),
            c3_out=ckpt.get('c3_out', 8), c4_out=ckpt.get('c4_out', 8),
        )
    elif is_pruned:
        print(f"[INFO] Detected pruned float checkpoint "
              f"(c1={ckpt['c1_out']}, c2={ckpt['c2_out']}, "
              f"c3={ckpt['c3_out']}, c4={ckpt['c4_out']})")
        model = ECG_1DCNN_Pruned(
            c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
            c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
        )
    else:
        print("[INFO] Loading base ECG_1DCNN checkpoint")
        model = ECG_1DCNN(num_classes=4)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Extract quantization info
    w_shifts_dict = ckpt.get('w_shift', ckpt.get('w_scales', {}))
    nb_dict       = ckpt.get('nb', {})
    input_shift   = ckpt.get('input_shift_bits', None)
    quant_scheme  = ckpt.get('quantization', 'INT8-unknown')
    bit_width     = ckpt.get('bit_width', 8)
    checkpoint_epoch = ckpt.get('epoch', '?')
    checkpoint_acc   = ckpt.get('val_acc', None)

    # w_int8 / b_int8 stored directly in checkpoint (QAT pipeline)
    w_int8_ckpt = ckpt.get('w_int8', {})
    # NOTE: checkpoint key 'b_int8' is a misnomer — it holds FLOAT bias, not INT8.
    # Bias is scaled here (× 2^nb) at export time, NOT pre-quantized in the ckpt.
    b_float_ckpt = ckpt.get('b_int8', {})

    print(f"[INFO] Quantization scheme: {quant_scheme}  bit_width={bit_width}")
    print(f"[INFO] Checkpoint epoch={checkpoint_epoch}  "
          + (f"val_acc={checkpoint_acc:.4f}" if checkpoint_acc else ""))
    if input_shift is not None:
        print(f"[INFO] Input shift bits: {input_shift}  (input_int8 = clamp(x * 2^{input_shift}, -127, 127))")

    if not w_shifts_dict:
        print("[WARNING] No w_shift found. Using shift_bits=0 for all layers.")
        w_shifts_dict = {name: 0 for name in LAYER_ORDER}
    if not nb_dict:
        print("[WARNING] No nb found in checkpoint. Using nb=0 for all layers.")
        nb_dict = {name: 0 for name in LAYER_ORDER}

    # ---- Build summary (for JSON metadata only) ----
    print(f"\n[INFO] Exporting INT8 weights to {output_dir}/")
    print(f"       Reading w_int8 directly from checkpoint\n")

    summary = []

    for layer_name in LAYER_ORDER:
        shift_bits = w_shifts_dict.get(layer_name, 0)
        nb = nb_dict.get(layer_name, 0)

        # --- Weight: read from checkpoint w_int8 ---
        if layer_name in w_int8_ckpt:
            w_int8 = np.array(w_int8_ckpt[layer_name], dtype=np.int32)
        else:
            # fallback: read float from model_state_dict and quantize
            layer = getattr(model, layer_name)
            w_float = layer.weight.data.cpu().numpy()
            w_int8 = np.clip(np.round(w_float * (2 ** shift_bits)), -127, 127).astype(np.int32)
            print(f"  [WARNING] {layer_name}: w_int8 not in checkpoint, quantizing float weights")

        w_info = {
            'param': f"{layer_name}_weight",
            'shape': list(w_int8.shape),
            'n_values': int(w_int8.size),
            'shift_bits': shift_bits,
            'int8_range': [int(w_int8.min()), int(w_int8.max())],
        }
        summary.append(w_info)
        print(f"  {layer_name}_weight    shape={str(list(w_int8.shape)):<18} w_shift={shift_bits}  range=[{w_int8.min()},{w_int8.max()}]")

        # --- Bias: for metadata ---
        if layer_name in b_float_ckpt:
            b_float = np.array(b_float_ckpt[layer_name], dtype=np.float64)
            # FC has no >>nb rescale → bias scaled by 2^w_shift[fc] (logit domain);
            # conv bias scaled by 2^nb (returns to b_float after >>nb).
            b_shift = shift_bits if layer_name == 'fc' else nb
            b_int32 = np.clip(np.round(b_float * (2 ** b_shift)), -(2**31), 2**31 - 1).astype(np.int64)

            b_info = {
                'param': f"{layer_name}_bias",
                'shape': list(b_float.shape),
                'n_values': int(b_int32.size),
                'bias_shift': int(b_shift),
                'int32_range': [int(b_int32.min()), int(b_int32.max())],
            }
            summary.append(b_info)
            print(f"  {layer_name}_bias     shape={str(list(b_float.shape)):<18} bias_shift={b_shift}  int32_range=[{b_int32.min()},{b_int32.max()}]")

    # ---- Export nb as activation shift ROM ----
    nb_path = os.path.join(output_dir, 'nb_shifts.mem')
    with open(nb_path, 'w') as f:
        f.write("// Per-layer activation right-shift bits (fphys Eq.11)\n")
        f.write("// Hardware: acc >>> nb_lut[layer_idx]  (arithmetic right shift)\n")
        f.write(f"// Layer order: {' -> '.join(LAYER_ORDER)}\n")
        for name in LAYER_ORDER:
            nb = nb_dict.get(name, 0)
            f.write(nb_to_hex2(nb) + "\n")

    print(f"  nb_shifts.mem      ({len(LAYER_ORDER)} entries: per-layer shift bits)")
    for i, name in enumerate(LAYER_ORDER):
        nb = nb_dict.get(name, 0)
        print(f"    [{i}] {name:10s} nb={nb}  (>> {nb} bits)")


    # ---- RTL hex files: conv_weights.hex, conv_bias.hex, fc_weights.hex ----
    export_rtl_hex(w_int8_ckpt, b_float_ckpt, nb_dict, w_shifts_dict, output_dir)

    # ---- Summary JSON ----
    total_params = sum(s['n_values'] for s in summary)
    total_clipped = sum(s.get('n_clipped', 0) for s in summary)

    summary_doc = {
        'checkpoint': checkpoint_path,
        'checkpoint_epoch': checkpoint_epoch,
        'quantization_scheme': quant_scheme,
        'bit_width': bit_width,
        'input_shift_bits': int(input_shift) if input_shift is not None else None,
        'w_shifts': {k: int(v) for k, v in w_shifts_dict.items()},
        'nb': {k: int(v) for k, v in nb_dict.items()},
        'total_params': total_params,
        'layers': summary,
    }

    # Export input_shift as reference file for hardware
    if input_shift is not None:
        scale_path = os.path.join(output_dir, 'input_scale.txt')
        with open(scale_path, 'w') as f:
            f.write(f"// Input quantization (power-of-2)\n")
            f.write(f"// input_int8 = clamp(round(x * 2^input_shift), -127, 127)\n")
            f.write(f"// input_shift_bits = {input_shift}\n")
        print(f"\n  input_scale.txt    (input_shift_bits={input_shift})")

    json_path = os.path.join(output_dir, 'weights_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary_doc, f, indent=2)

    # ---- Verilog address map ----
    print_address_map(summary, output_dir)

    # ---- Runtime config values for reconfigurable hardware ----
    # Derive Cin/Cout from summary layer shapes
    cin_derived  = []
    cout_derived = []
    for layer_name in LAYER_ORDER[:4]:  # Conv1..Conv4 only
        for s in summary:
            if layer_name in s['param'] and 'weight' in s['param']:
                shape = s['shape']
                cout_derived.append(shape[0])  # out_channels
                cin_derived.append(shape[1])   # in_channels
                break

    w_base, b_base = compute_rom_bases(cin_derived, cout_derived)
    print_hw_config(cin_derived, cout_derived, w_base, b_base)

    # ---- Final report ----
    print(f"\n{'='*60}")
    print(f"  Total weights exported : {total_params}")
    if total_clipped:
        print(f"  ⚠  Values clipped     : {total_clipped}  "
              f"(weights outside INT8 range [-128, 127])")
    else:
        print(f"  ✓ No clipping (all weights within INT8 range)")
    print(f"  Output directory       : {output_dir}")
    print(f"  Quantization scheme    : {quant_scheme}")
    print(f"  Activation shift bits  : YES (in nb_shifts.mem)")
    print(f"{'='*60}")

    return summary_doc


# ============================================================
#  Runtime config helper: compute ROM base addresses for any
#  Cin/Cout variant (supports reconfigurable hardware)
# ============================================================

def compute_rom_bases(cin_list, cout_list, K=5):
    """
    Compute weight and bias ROM base addresses for a given Cin/Cout config.

    Args:
        cin_list:  [cin_l0, cin_l1, cin_l2, cin_l3]  (Conv1..Conv4)
        cout_list: [cout_l0, cout_l1, cout_l2, cout_l3]
        K: kernel size (default 5)

    Returns:
        w_base: list of 4 weight base addresses
        b_base: list of 4 bias base addresses

    Usage:
        w_base, b_base = compute_rom_bases([1,3,6,6], [3,6,6,10])
        # → w_base=[0,27,141,345], b_base=[15,117,321,645]
    """
    w_base, b_base = [], []
    offset = 0
    for cin, cout in zip(cin_list, cout_list):
        w_base.append(offset)
        b_base.append(offset + cin * K * cout)
        offset += (cin * K + 4) * cout
    return w_base, b_base


def print_hw_config(cin_list, cout_list, w_base, b_base):
    """Print Verilog instantiation parameters for hardware config ports."""
    print("\n// ---- Hardware config port values ----")
    for i, (cin, cout, wb, bb) in enumerate(zip(cin_list, cout_list, w_base, b_base)):
        print(f"// Layer {i}: Cin={cin}, Cout={cout}, w_base={wb}, b_base={bb}")
    print(f".cfg_cin_0({cin_list[0]}), .cfg_cin_1({cin_list[1]}), "
          f".cfg_cin_2({cin_list[2]}), .cfg_cin_3({cin_list[3]}),")
    print(f".cfg_cout_0({cout_list[0]}), .cfg_cout_1({cout_list[1]}), "
          f".cfg_cout_2({cout_list[2]}), .cfg_cout_3({cout_list[3]}),")
    print(f".cfg_w_base_0({w_base[0]}), .cfg_w_base_1({w_base[1]}), "
          f".cfg_w_base_2({w_base[2]}), .cfg_w_base_3({w_base[3]}),")
    print(f".cfg_b_base_0({b_base[0]}), .cfg_b_base_1({b_base[1]}), "
          f".cfg_b_base_2({b_base[2]}), .cfg_b_base_3({b_base[3]})")


# ============================================================
#  Address map helper
# ============================================================

def print_address_map(summary: list, output_dir: str):
    """Print and save Verilog ROM address map."""
    lines = [
        "// ============================================================",
        "// INT8 Weight ROM Address Map",
        "// Use as Verilog parameters: BASE_ADDR, N_WEIGHTS",
        "// ============================================================",
        f"// {'Parameter':<30} {'Base (dec)':>12} {'Count':>8} {'End (dec)':>12}",
        "// " + "-" * 66,
    ]
    addr = 0
    for s in summary:
        n = s['n_values']
        end = addr + n - 1
        param = s['param']
        lines.append(
            f"// {param:<30} {addr:>12}  {n:>8}  {end:>12}"
        )
        addr += n

    lines += [
        "// " + "-" * 66,
        f"// {'TOTAL':<30} {addr:>12}",
        "// ============================================================",
    ]

    map_txt = "\n".join(lines)
    print()
    for l in lines:
        print(l)

    map_path = os.path.join(output_dir, 'rom_address_map.txt')
    with open(map_path, 'w') as f:
        f.write(map_txt + "\n")


# ============================================================
#  CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Export INT8 weights with per-layer scales from checkpoint'
    )
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to INT8 checkpoint (with scales) from quantize_int8.py or QAT')
    p.add_argument('--output_dir', type=str,
                   default='./results/weights_int8',
                   help='Directory to write .mem files')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_all(args.checkpoint, args.output_dir)
