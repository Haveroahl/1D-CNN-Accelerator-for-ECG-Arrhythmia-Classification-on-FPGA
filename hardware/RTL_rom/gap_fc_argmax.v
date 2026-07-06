// gap_fc_argmax.v
// Sequential engine: GAP (6cy) → FC (10cy + 1 flush) → Argmax (4cy) → Done (1cy)
// Total: 22 cycles after entry from Conv4 layer_done
//
// Thin wrapper (bit-exact structural split, no logic change) over 3 submodules:
//   gap_unit    — Global Average Pooling  (ping_dout → gap_reg)
//   fc_unit     — Fully-connected + FC weight/bias store  (gap_reg → fc_acc)
//   argmax_unit — Argmax over 4 logits  (fc_acc → result)
//
// Public ports are UNCHANGED from the pre-split module, so ecg_core needs no edit.
//
// FC weights: stored in fc_unit's fc_w[k][i], k=0..3 output neurons, i=0..7 inputs.
// NB=0 → no output rescale (argmax is scale-invariant). FC logits live at scale
// 2^w_shift[fc], so the FC bias is pre-scaled by 2^w_shift[fc] (fc_bias.hex,
// INT32) to be commensurate with fc_acc, and seeded into fc_acc before the MACs.
//
// ROM single-load build: FC weights baked in via $readmemh (in fc_unit); no bus.

module gap_fc_argmax (
    input  wire        clk,
    input  wire        rst,

    // Sub-FSM control (driven by cnn_controller)
    input  wire [2:0]  fc_sub_state,   // GAP_S/FC_S/FC_FLUSH/ARGMAX_S/DONE_S
    input  wire [3:0]  gap_step,       // 0..5
    input  wire [3:0]  fc_step,        // 0..9
    input  wire [1:0]  argmax_step,    // 0..3

    // Ping SRAM data (Conv4 output, 8 channels × 4 entries)
    input  wire [63:0] ping_dout,               // packed: ping_dout[ch*8+:8], 1-cy latency

    // Conv4 active-output mask (= cp_en of Conv4). Channels with mask bit 0 were
    // never written this inference, so their ping bank holds stale data; GAP
    // forces their pooled value to 0 so a reduced Conv4 out_ch (e.g. 2/4/6) is
    // bit-exact without the driver having to zero-pad FC weights. Default 8'hFF
    // (all active) reproduces the original full-8-channel behavior.
    input  wire [7:0]  out_ch_mask,

    // Ping SRAM read address (to top-level → ping_pong_sram rd_addr)
    output wire [8:0]  gap_rd_addr,    // broadcast to all 8 channels (0..3)

    // Outputs
    output wire [1:0]  result          // argmax class index
);

    // ── Inter-stage buses (flattened, Verilog-2001) ─────────────────────────
    wire [63:0]  gap_reg_flat;   // 8 × INT8 GAP output
    wire [127:0] fc_acc_flat;    // 4 × INT32 FC logits

    // ── GAP ──────────────────────────────────────────────────────────────────
    gap_unit u_gap (
        .clk          (clk),
        .rst          (rst),
        .fc_sub_state (fc_sub_state),
        .gap_step     (gap_step),
        .ping_dout    (ping_dout),
        .out_ch_mask  (out_ch_mask),
        .gap_rd_addr  (gap_rd_addr),
        .gap_reg_flat (gap_reg_flat)
    );

    // ── FC (+ weight/bias store) ──────────────────────────────────────────────
    fc_unit u_fc (
        .clk          (clk),
        .rst          (rst),
        .fc_sub_state (fc_sub_state),
        .fc_step      (fc_step),
        .gap_reg_flat (gap_reg_flat),
        .fc_acc_flat  (fc_acc_flat)
    );

    // ── Argmax ─────────────────────────────────────────────────────────────────
    argmax_unit u_argmax (
        .clk          (clk),
        .rst          (rst),
        .fc_sub_state (fc_sub_state),
        .argmax_step  (argmax_step),
        .fc_acc_flat  (fc_acc_flat),
        .result       (result)
    );

endmodule
