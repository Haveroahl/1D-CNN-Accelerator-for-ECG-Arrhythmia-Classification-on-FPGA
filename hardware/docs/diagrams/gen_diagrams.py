"""Generate block diagrams (PNG) for ECG CNN accelerator RTL.

Outputs:
  01_ecg_accelerator_top.png
  02_cp_block.png
  03_cp_engine.png
  04_gap_fc_argmax.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = r"d:\Thesis101\hardware\docs\diagrams"

# ───────────────────────── Helpers ─────────────────────────
def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=140)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_aspect('equal'); ax.axis('off')
    return fig, ax

def box(ax, x, y, w, h, text, fc="#E8F1FB", ec="#1F4E79", fs=8, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.5",
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, weight='bold' if bold else 'normal')

def lbl(ax, x, y, text, fs=7, color='black', ha='center', va='center', weight='normal'):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, weight=weight)

def arrow(ax, x1, y1, x2, y2, color="#333", lw=1.1, text=None, ts=7, tdy=0.6, style='->'):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=10, color=color, lw=lw)
    ax.add_patch(a)
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + tdy, text, ha='center', va='center',
                fontsize=ts, color=color)

def title(ax, t):
    ax.text(50, 97, t, ha='center', va='center', fontsize=13, weight='bold')

# ═════════════════════════ 1. ecg_accelerator_top ═════════════════════════
def diagram_top():
    fig, ax = new_fig(14, 10)
    title(ax, "ecg_accelerator_top — Top-Level Block Diagram")

    # Outer chip boundary
    ax.add_patch(Rectangle((4, 6), 92, 86, fc="#FAFCFF", ec="#1F4E79", lw=1.5, ls='--'))
    lbl(ax, 6, 90, "ecg_accelerator_top", fs=10, ha='left', weight='bold', color='#1F4E79')

    # External ports (left)
    ext_ports = [
        ("clk",            85),
        ("rst",            81),
        ("rst_n",          77),
        ("avs_address[4:0]", 70),
        ("avs_write",      66),
        ("avs_read",       62),
        ("avs_writedata[31:0]", 58),
        ("avs_readdata[31:0]", 54),
    ]
    for name, y in ext_ports:
        ax.plot([0, 4], [y, y], color='#666', lw=1)
        lbl(ax, 0.2, y+1.0, name, fs=7, ha='left')

    # avalon_slave
    box(ax, 8, 60, 18, 22, "avalon_slave\n\n• HPS bridge\n• Reg file\n• start/busy/done/result",
        fc="#FDE9D9", ec="#C55A11", bold=True, fs=8)

    # input_sram
    box(ax, 32, 70, 18, 12, "input_sram\n2500×8b  (M10K)", fc="#E2F0D9", ec="#548235", bold=True, fs=8)

    # ping_pong_sram
    box(ax, 56, 60, 22, 16,
        "ping_pong_sram\n2 sets × 8 ch × 500 × 8b\n16 M10K (+ bank_sel swap)",
        fc="#E2F0D9", ec="#548235", bold=True, fs=8)

    # cp_engine
    box(ax, 32, 38, 24, 24,
        "cp_engine\n\n• SRW (8×5-tap)\n• MUX → mux_s1\n• W-packed (4 ROMs)\n• 8 × cp_block\n• Bias (MLAB)",
        fc="#DEEBF7", ec="#1F4E79", bold=True, fs=8)

    # gap_fc_argmax
    box(ax, 64, 38, 22, 18, "gap_fc_argmax\n\nGAP → FC → Argmax\n22 cycles", fc="#DEEBF7", ec="#1F4E79", bold=True, fs=8)

    # cnn_controller
    box(ax, 14, 14, 72, 18,
        "cnn_controller (Unified FSM:  IDLE → LOAD → CONV1..4 → GAP_FC → DONE)\n"
        "ctrl outputs: a, t, shift_en, srw_rst, compute_en, in_ch, in_len, nb, relu_en,\n"
        "              cp_en, bank_sel, pong_addr, pool_rst, fc_sub_state, gap/fc/argmax_step,\n"
        "              layer_state, busy, done, result",
        fc="#FFF2CC", ec="#BF8F00", bold=True, fs=8)

    # Arrows
    arrow(ax, 17, 60, 17, 32, text="start, busy,\ndone, result", tdy=0, ts=6)
    arrow(ax, 26, 75, 32, 75, text="sram_wr_addr\nsram_din, we", tdy=1.2, ts=6)
    arrow(ax, 50, 73, 64, 67, text="input_sram_dout", ts=6)
    arrow(ax, 44, 70, 44, 62, text="(read addr)", ts=6)

    arrow(ax, 56, 50, 64, 47, text="ping_dout[63:0]", ts=6)
    arrow(ax, 56, 56, 67, 60, text="pong_din, pong_we,\npong_wr_addr", ts=6, tdy=0)
    arrow(ax, 67, 60, 56, 56, color='#888')  # back arrow for write
    # Re-do as bidirectional-style: cp_engine writes to PP
    arrow(ax, 50, 56, 56, 62, text="", ts=6)

    # Controller fans out
    arrow(ax, 30, 32, 30, 38, text="ctrl_*", ts=6)
    arrow(ax, 70, 32, 70, 38, text="fc_sub_state,\ngap/fc/argmax_step", tdy=1.4, ts=6)
    arrow(ax, 60, 32, 60, 56, color='#888', text="bank_sel,\npong_addr", tdy=0, ts=6)

    # gap result back to controller
    arrow(ax, 75, 38, 75, 32, text="argmax_result", ts=6)

    # gap_rd_addr to ping_pong
    arrow(ax, 72, 56, 72, 60, text="gap_rd_addr", ts=6)

    plt.savefig(f"{OUT}/01_ecg_accelerator_top.png", bbox_inches='tight', dpi=160)
    plt.close(fig)

# ═════════════════════════ 2. cp_block ═════════════════════════
def diagram_cp_block():
    fig, ax = new_fig(11, 14)
    title(ax, "cp_block — Conv-Pool Pipeline (1 output channel)")

    stages = [
        ("S1  MULT  (5×  DSP18)\nprod[k] = taps_in[k] × w[k]  (16b)", 86),
        ("S2  Adder-Tree-1\nsum01 = p0+p1, sum23 = p2+p3, p4_d1", 79),
        ("S3  Adder-Tree-2\nsum0123 = sum01+sum23, p4_d2", 72),
        ("S4  Adder-Tree-3\ntree_out = sum0123 + p4_d2  (20b)", 65),
        ("S5  Accumulator (32b)\nif (a==0) acc=tree else acc+=tree\nGated by compute_en_in (ce_d5)", 56),
        ("S5b  ACC_FINAL register\nacc_final = (a==0)?tree : acc+tree\n(out_valid = ce && a==in_ch-1)", 47),
        ("S_bias  +Bias  (32b)\nbiased = acc_final + bias_in", 39),
        ("S6  Rescale-1   (round-half-up)\nshifted = (biased + (1<<(nb-1))) >>> nb", 31),
        ("S7  Clamp [-127, 127]\nclamped[7:0]", 24),
        ("S8  ReLU   (Conv4 only)\nrelu_out = relu_en & sign ? 0 : clamped", 17),
        ("S9  MaxPool K=5  rolling\nmax_reg, pool_cnt 0..4, write @ cnt==4", 9),
    ]
    for txt, y in stages:
        box(ax, 22, y, 56, 5.5, txt, fc="#DEEBF7", ec="#1F4E79", fs=8)
        if y > 9:
            arrow(ax, 50, y, 50, y - 1.5, lw=1.3)

    # Left-side control inputs
    ctrls = [
        ("a_in (a_d5) [3:0]",        58, "S5/S5b"),
        ("in_ch [3:0]",              54, "S5/S5b"),
        ("compute_en_in (ce_d5)",    50, "S5"),
        ("pool_rst",                 46, "all regs"),
        ("bias_in [31:0]",           40, "S_bias"),
        ("nb [4:0]",                 32, "S6"),
        ("relu_en",                  18, "S8"),
    ]
    for name, y, _ in ctrls:
        ax.plot([0, 22], [y, y], color='#A33', lw=0.9, ls='--')
        lbl(ax, 0.5, y+0.9, name, fs=7, ha='left', color='#A33')

    # Top inputs
    lbl(ax, 30, 92.5, "taps_in[39:0]", fs=8, weight='bold')
    lbl(ax, 70, 92.5, "w[39:0]", fs=8, weight='bold')
    arrow(ax, 30, 91.5, 35, 88, lw=1.3)
    arrow(ax, 70, 91.5, 65, 88, lw=1.3)

    # Bottom outputs
    arrow(ax, 40, 9, 40, 4, lw=1.3)
    arrow(ax, 60, 9, 60, 4, lw=1.3)
    lbl(ax, 40, 2.5, "pool_write", fs=8, weight='bold')
    lbl(ax, 60, 2.5, "pool_out[7:0]", fs=8, weight='bold')

    # Notes
    lbl(ax, 50, 94, "Pipeline depth = 5 cy (mux_s1→prod→sum01/23→sum0123→tree_out)",
        fs=7, color='#444')

    plt.savefig(f"{OUT}/02_cp_block.png", bbox_inches='tight', dpi=160)
    plt.close(fig)

# ═════════════════════════ 3. cp_engine ═════════════════════════
def diagram_cp_engine():
    fig, ax = new_fig(15, 12)
    title(ax, "cp_engine — 8 CP blocks parallel + SRW + MUX + W-packed")

    # Outer
    ax.add_patch(Rectangle((2, 4), 96, 87, fc="#FAFCFF", ec="#1F4E79", lw=1.5, ls='--'))

    # Data inputs (top)
    lbl(ax, 12, 89, "input_sram_dout[7:0]", fs=8, weight='bold')
    lbl(ax, 30, 89, "ping_dout[63:0]\n(8ch packed)", fs=8, weight='bold')
    arrow(ax, 12, 87.5, 14, 83)
    arrow(ax, 30, 87.5, 26, 83)

    # Top MUX (input source select + pad)
    box(ax, 8, 76, 30, 7,
        "Top-MUX  +  pad_zero_r mask\nConv1: input_sram_dout    Conv2..4: ping_dout\nsrw_din[0..7] (8 × 8b)",
        fc="#FFF2CC", ec="#BF8F00", fs=7)
    arrow(ax, 23, 76, 23, 71, text="srw_din", ts=6)

    # SRW BLOCK
    box(ax, 4, 56, 38, 14,
        "★  SRW ARRAY  (8 ch × 5-tap shift registers)\n\n"
        "ch0: [s0][s1][s2][s3][s4]   ← srw_din[0]\n"
        "ch1: [s0][s1][s2][s3][s4]   ← srw_din[1]\n"
        "   ...  (ch2..ch7)\n"
        "s0=newest   s4=oldest    (shift_en, srw_rst)",
        fc="#E2F0D9", ec="#548235", bold=True, fs=7)

    # MUX BLOCK
    box(ax, 4, 42, 38, 11,
        "★  MUX  (combinational 5 × 8:1)\n"
        "mux_comb[0] = srw[a*5+4]  (oldest = x[t−2])\n"
        "mux_comb[2] = srw[a*5+2]  (center = x[t])\n"
        "mux_comb[4] = srw[a*5+0]  (newest = x[t+2])\n"
        "→  mux_s1[39:0]  (FF, 1 cy)",
        fc="#E2F0D9", ec="#548235", bold=True, fs=7)
    arrow(ax, 23, 56, 23, 53)

    # W-PACKED BLOCK
    box(ax, 50, 56, 44, 22,
        "★  WEIGHT PATH\n\n"
        "w_rom_conv1[0:3]   (4×40b)   ← $readmemh conv1_w.hex\n"
        "w_rom_conv2[0:15]  (16×40b)  ← $readmemh conv2_w.hex\n"
        "w_rom_conv3[0:31]  (32×40b)  ← $readmemh conv3_w.hex\n"
        "w_rom_conv4[0:63]  (64×40b)  ← $readmemh conv4_w.hex\n"
        "      │  (async read, FF array)\n"
        "      ▼\n"
        "w_comb[oc]  (combinational 4:1 layer + 8:1 ic MUX)\n"
        "      │  (layer_state, a[2:0])\n"
        "      ▼\n"
        "w_packed[0..7]  (8 × 40b, FF — aligns w/ mux_s1)",
        fc="#E2F0D9", ec="#548235", bold=True, fs=7)

    # Bias path
    box(ax, 50, 44, 44, 10,
        "Bias path\n"
        "b_store[0:31] (MLAB) ← $readmemh conv_bias.hex\n"
        "b_cur[oc] = b_store[oc*4 + layer_idx]  (registered)",
        fc="#FDE9D9", ec="#C55A11", fs=7)

    # Delay chain
    box(ax, 4, 30, 90, 7,
        "Delay chain:   a_d1..a_d5,  inch_d1..d5,  ce_d1..ce_d5   (5 stages → align controller signals with pipeline depth at acc edge)",
        fc="#FFF2CC", ec="#BF8F00", fs=7)
    arrow(ax, 23, 42, 23, 37, text="mux_s1", ts=6)
    arrow(ax, 72, 56, 72, 37, text="w_packed[0..7]", ts=6)
    arrow(ax, 72, 44, 72, 37, color='#888', text="b_cur[0..7]", ts=6)

    # 8 cp_blocks
    box(ax, 4, 14, 90, 13,
        "8 × cp_block  (parallel output channels oc=0..7)\n\n"
        "┌── oc0 ──┐  ┌── oc1 ──┐  ┌── oc2 ──┐   ...   ┌── oc7 ──┐\n"
        "│ taps_in │  │ taps_in │  │ taps_in │         │ taps_in │     (mux_s1 broadcast)\n"
        "│ w=w_pk0 │  │ w=w_pk1 │  │ w=w_pk2 │         │ w=w_pk7 │\n"
        "│ bias=b0 │  │ bias=b1 │  │ bias=b2 │         │ bias=b7 │     a_d5/inch_d5/ce_d5 shared",
        fc="#DEEBF7", ec="#1F4E79", bold=True, fs=7)
    arrow(ax, 50, 30, 50, 27, text="", ts=6)

    # Pong write gate
    box(ax, 4, 5.5, 90, 7,
        "Pong write gate:    pong_we[oc] = pool_write[oc] & cp_en[oc]      pong_din[oc*8+:8] = pool_out[oc]\n"
        "Address path:       pong_wr_addr = pong_addr_in        sram_rd_addr = sram_rd_addr_in − 2  (front-pad offset)",
        fc="#FDE9D9", ec="#C55A11", fs=7)
    arrow(ax, 50, 14, 50, 12.5)

    # Controller inputs (right side, dashed)
    ctrl_lines = ["a[3:0]", "in_ch[3:0]", "in_len[11:0]", "shift_en", "srw_rst",
                  "compute_en", "nb[4:0]", "relu_en", "cp_en[7:0]",
                  "layer_state[2:0]", "pool_rst", "pong_addr_in[8:0]", "sram_rd_addr_in[11:0]"]
    for i, c in enumerate(ctrl_lines):
        y = 88 - i*1.6
        lbl(ax, 99.5, y, c, fs=6, ha='right', color='#A33')

    # Outputs (right edge)
    lbl(ax, 99.5, 9, "pong_wr_addr / pong_din / pong_we / sram_rd_addr  →",
        fs=7, ha='right', weight='bold', color='#1F4E79')

    plt.savefig(f"{OUT}/03_cp_engine.png", bbox_inches='tight', dpi=160)
    plt.close(fig)

# ═════════════════════════ 4. gap_fc_argmax ═════════════════════════
def diagram_gap_fc():
    fig, ax = new_fig(12, 12)
    title(ax, "gap_fc_argmax — GAP → FC → Argmax  (22 cycles)")

    ax.add_patch(Rectangle((3, 4), 94, 88, fc="#FAFCFF", ec="#1F4E79", lw=1.5, ls='--'))

    # Top input
    lbl(ax, 50, 89, "ping_dout[63:0]  (8 ch × 8b, from ping_pong_sram, 1cy read latency)",
        fs=8, weight='bold')
    arrow(ax, 50, 87.5, 50, 84)

    # GAP STAGE
    box(ax, 12, 64, 76, 19,
        "GAP STAGE  (6 cycles)        ← fc_sub_state == GAP_S    ← gap_step[3:0] (0..5)\n\n"
        "gap_rd_addr ← comb. from gap_step\n"
        "  step 0 → addr 0   (issue read pos 0)\n"
        "  step 1 → addr 1   + accumulate pos 0\n"
        "  step 2 → addr 2   + accumulate pos 1\n"
        "  step 3 → addr 3   + accumulate pos 2\n"
        "  step 4           + accumulate pos 3\n"
        "  step 5: gap_reg[ch] = gap_acc[ch][9:2]  (floor(sum/4))",
        fc="#DEEBF7", ec="#1F4E79", bold=True, fs=7)
    arrow(ax, 50, 64, 50, 60, text="gap_reg[0..7]  (INT8 × 8)", ts=7)

    # FC STAGE
    box(ax, 12, 36, 76, 22,
        "FC STAGE  (10 + 1 flush cycles)    ← fc_sub_state == FC_S / FC_FLUSH    ← fc_step\n\n"
        "FC weight ROM:  fc_w[0:31]  (4 oc × 8 in, INT8)   ← $readmemh fc_weights.hex   (no bias)\n\n"
        "2-cycle pipeline per input:\n"
        "  stage 1: fc_gap_pipe ← gap_reg[fc_step−1] ;  fc_w_idx ← fc_step−1\n"
        "  stage 2: fc_prod[k] ← fc_gap_pipe × fc_w[k*8 + fc_w_idx]   (4 multipliers, k=0..3)\n"
        "  stage 3: fc_acc[k] += sext32(fc_prod[k])   (gated by prod_valid, step≥3)\n\n"
        "FC_FLUSH: drain last product gap[7]×w[k][7]   →  fc_acc[0..3] = 4 INT32 logits",
        fc="#DEEBF7", ec="#1F4E79", bold=True, fs=7)
    arrow(ax, 50, 36, 50, 32, text="fc_acc[0..3]  (4 × INT32)", ts=7)

    # ARGMAX
    box(ax, 12, 13, 76, 18,
        "ARGMAX STAGE  (4 cycles)    ← fc_sub_state == ARGMAX_S    ← argmax_step[1:0]\n\n"
        "  step 0: argmax_max = fc_acc[0]; argmax_idx = 0\n"
        "  step 1: if fc_acc[1] > max → update\n"
        "  step 2: if fc_acc[2] > max → update\n"
        "  step 3: if fc_acc[3] > max → update\n\n"
        "argmax_max[31:0], argmax_idx[1:0]",
        fc="#DEEBF7", ec="#1F4E79", bold=True, fs=7)
    arrow(ax, 50, 13, 50, 9)
    lbl(ax, 50, 7, "result[1:0]", fs=9, weight='bold', color='#1F4E79')

    # Side outputs / control
    lbl(ax, 95, 81, "gap_rd_addr[8:0]  →\n(to ping_pong_sram\n  rd_addr via top MUX)",
        fs=7, ha='right', color='#A33')
    arrow(ax, 88, 75, 95, 78, color='#A33')

    lbl(ax, 95, 17, "done  →\n(fc_sub_state == DONE_S,\n 1-cycle pulse)",
        fs=7, ha='right', color='#A33')

    # Total budget
    lbl(ax, 50, 92, "Total: GAP 6 + FC 10 + FC_FLUSH 1 + Argmax 4 + DONE 1 = 22 cycles",
        fs=8, color='#555')

    plt.savefig(f"{OUT}/04_gap_fc_argmax.png", bbox_inches='tight', dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    diagram_top()
    diagram_cp_block()
    diagram_cp_engine()
    diagram_gap_fc()
    print("Done: 4 PNGs written to", OUT)
