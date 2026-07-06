// ecg_core.v
// Bus-agnostic accelerator core — knows nothing about Avalon.
//
// Split out from ecg_accelerator_top so the same core can be reused under
// different wrappers (Phase C virtual-pin top, Phase D Qsys/HPS top) without
// touching verified datapath/control logic.
//
// Modules instantiated (copied verbatim from old top, logic unchanged):
//   ping_pong_sram    — 2 sets × 8 banks × 500 entries, inter-layer feature maps
//   cp_engine         — 8 CP blocks (Conv1..4)
//   gap_fc_argmax     — GAP/FC/Argmax post-processing
//   cnn_controller    — Unified FSM
//
// input_sram lives in the WRAPPER, not here. The split is along the I/O write
// boundary: input_sram is the input buffer the host WRITES (an I/O concern),
// so it sits outside the compute core. The core only READS it as an external
// memory subsystem via input_rd_addr → input_dout (1-cycle synchronous, same
// as before). ping_pong stays inside the core: it is compute scratch the host
// never touches. This keeps "compute latency" (start→done) cleanly scoped to
// the core and excludes the host's input-load phase.
//
// Interface with the bus adapter / wrapper:
//   out: input_rd_addr  → wrapper input_sram read address
//   in : input_dout     ← wrapper input_sram read data (1-cycle latency)
//   in : start                               (kick off inference)
//   out: busy / done / result                (status + class 0..3)

module ecg_core (
    input  wire        clk,
    input  wire        rst,         // synchronous reset (active high)

    // ── Input SRAM read port (input_sram lives in the wrapper) ─────────
    output wire [11:0] input_rd_addr,
    input  wire [7:0]  input_dout,

    // ── Control / status (with bus adapter) ────────────────────────────
    input  wire        start,
    output wire        busy,
    output wire        done,
    output wire [1:0]  result
);

    // ── Controller outputs ─────────────────────────────────────────────
    wire [3:0]  ctrl_a;
    wire [11:0] ctrl_t;
    wire        ctrl_shift_en;
    wire        ctrl_srw_rst;
    wire        ctrl_compute_en;
    wire [3:0]  ctrl_in_ch;
    wire [11:0] ctrl_in_len;
    wire [3:0]  ctrl_nb;
    wire        ctrl_relu_en;
    wire [7:0]  ctrl_cp_en;
    wire        ctrl_bank_sel;
    wire [11:0] ctrl_pong_addr;
    wire        ctrl_pool_rst;
    wire [2:0]  ctrl_fc_sub_state;
    wire [3:0]  ctrl_gap_step;
    wire [3:0]  ctrl_fc_step;
    wire [1:0]  ctrl_argmax_step;
    wire [2:0]  ctrl_layer_state;

    // ── Ping-Pong SRAM ─────────────────────────────────────────────────
    wire [63:0] pp_dout;            // Ping output packed: pp_dout[ch*8+:8]

    // ── CP engine ──────────────────────────────────────────────────────
    wire        cp_pool_write;      // from cp_engine (representative ch0)
    wire [63:0] cp_pong_din;        // packed: cp_pong_din[ch*8+:8]
    wire [7:0]  cp_pong_we;
    wire [11:0] cp_sram_rd_addr;

    // ── GAP/FC/Argmax ──────────────────────────────────────────────────
    wire [8:0]  gap_rd_addr;
    wire [1:0]  gap_result;

    // ── Pool write: cp_engine ch0 pool_write feeds controller ──────────
    // All active cp_blocks write simultaneously — ch0 is representative.
    // cp_engine exposes pong_we[0] as the representative pool_write signal.
    assign cp_pool_write = cp_pong_we[0];

    // ── input_sram read port → wrapper (input_sram is instantiated there) ──
    assign input_rd_addr = cp_sram_rd_addr[11:0];

    // ── ping_pong_sram ─────────────────────────────────────────────────
    // Read address: cp_engine during CONV1..4, gap_fc_argmax during GAP_FC_S
    // Use gap_rd_addr when in GAP_FC_S state (layer_state == 3'd6), else cp_sram_rd_addr
    wire [8:0] pp_rd_addr = (ctrl_layer_state == 3'd6) ? gap_rd_addr
                                                        : cp_sram_rd_addr[8:0];

    ping_pong_sram u_pp (
        .clk     (clk),
        .bank_sel(ctrl_bank_sel),
        .wr_addr (ctrl_pong_addr[8:0]),
        .din     (cp_pong_din),
        .we      (cp_pong_we),
        .rd_addr (pp_rd_addr),
        .dout    (pp_dout)
    );

    // ── cp_engine ──────────────────────────────────────────────────────
    cp_engine u_cpe (
        .clk              (clk),
        .rst              (rst),
        .a                (ctrl_a),
        .in_ch            (ctrl_in_ch),
        .in_len           (ctrl_in_len),
        .shift_en         (ctrl_shift_en),
        .srw_rst          (ctrl_srw_rst),
        .compute_en       (ctrl_compute_en),
        .nb               (ctrl_nb),
        .relu_en          (ctrl_relu_en),
        .cp_en            (ctrl_cp_en),
        .layer_state      (ctrl_layer_state),
        .pool_rst         (ctrl_pool_rst),
        .input_sram_dout  (input_dout),
        .ping_dout        (pp_dout),
        .pong_din         (cp_pong_din),
        .pong_we          (cp_pong_we),
        .sram_rd_addr     (cp_sram_rd_addr),
        .sram_rd_addr_in  (ctrl_t)
    );

    // ── gap_fc_argmax ──────────────────────────────────────────────────
    gap_fc_argmax u_gfa (
        .clk          (clk),
        .rst          (rst),
        .fc_sub_state (ctrl_fc_sub_state),
        .gap_step     (ctrl_gap_step),
        .fc_step      (ctrl_fc_step),
        .argmax_step  (ctrl_argmax_step),
        .ping_dout    (pp_dout),
        .out_ch_mask  (8'hFF),   // fixed Chapman Conv4: all 8 output channels active
        .gap_rd_addr  (gap_rd_addr),
        .result       (gap_result)
    );

    // ── cnn_controller ─────────────────────────────────────────────────
    cnn_controller u_ctrl (
        .clk          (clk),
        .rst          (rst),
        .start        (start),
        .pool_write   (cp_pool_write),
        .a            (ctrl_a),
        .t            (ctrl_t),
        .shift_en     (ctrl_shift_en),
        .srw_rst      (ctrl_srw_rst),
        .compute_en   (ctrl_compute_en),
        .in_ch        (ctrl_in_ch),
        .in_len       (ctrl_in_len),
        .nb           (ctrl_nb),
        .relu_en      (ctrl_relu_en),
        .cp_en        (ctrl_cp_en),
        .bank_sel     (ctrl_bank_sel),
        .pong_addr    (ctrl_pong_addr),
        .pool_rst     (ctrl_pool_rst),
        .fc_sub_state (ctrl_fc_sub_state),
        .gap_step     (ctrl_gap_step),
        .fc_step      (ctrl_fc_step),
        .argmax_step  (ctrl_argmax_step),
        .argmax_result(gap_result),
        .layer_state  (ctrl_layer_state),
        .busy         (busy),
        .done         (done),
        .result       (result)
    );

endmodule
