// ecg_core_asic.v  —  ASIC (Sky130/OpenLane) top-level for the ECG accelerator.
//
// Same datapath/FSM as ecg_core.v (verified bit-exact), with two differences:
//   1. Memory uses the macro-friendly variant:
//        ping_pong_sram_asic (2 macros 512x64, per-byte wmask)
//      input_sram_asic (1 macro 4096x8) lives in the wrapper, read via
//      input_rd_addr → input_dout (same I/O-write-boundary split as ecg_core).
//   2. No Avalon/JTAG/PLL wrapper — the core interface is exposed directly as
//      chip pins (control/status + the input_sram read port). The wrapper holds
//      input_sram_asic; a test/host writes it, pulses start, polls busy/done.
//
// cp_engine / cp_block / cnn_controller / gap_fc_argmax are reused VERBATIM from
// hardware/RTL (no ASIC-specific copy needed — they are technology-agnostic logic).
// The run script compiles them from hardware/RTL alongside these asic files.

module ecg_core_asic (
    input  wire        clk,
    input  wire        rst,         // synchronous reset (active high)

    // ── Input SRAM read port (input_sram_asic lives in the wrapper) ─────────
    output wire [11:0] input_rd_addr,
    input  wire [7:0]  input_dout,

    // ── Control / status ───────────────────────────────────────────────────
    input  wire        start,
    output wire        busy,
    output wire        done,
    output wire [1:0]  result
);

    // ── Controller outputs ─────────────────────────────────────────────────
    wire [3:0]  ctrl_a;
    wire [11:0] ctrl_t;
    wire        ctrl_shift_en;
    wire        ctrl_srw_rst;
    wire        ctrl_compute_en;
    wire [3:0]  ctrl_in_ch;
    wire [11:0] ctrl_in_len;
    wire [4:0]  ctrl_nb;
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

    wire [63:0] pp_dout;

    wire        cp_pool_write;
    wire [63:0] cp_pong_din;
    wire [7:0]  cp_pong_we;
    wire [11:0] cp_sram_rd_addr;

    wire [8:0]  gap_rd_addr;
    wire [1:0]  gap_result;

    assign cp_pool_write = cp_pong_we[0];

    // ── Topology config: fixed Chapman layout (no bus on ASIC variant) ───────
    // Same default values avalon_slave loads on reset. Tied to constants here so
    // the runtime-configurable cp_engine/cnn_controller cfg_* inputs are driven.
    localparam [15:0] CFG_IN_CH = {4'd8, 4'd4, 4'd4, 4'd1};
    localparam [31:0] CFG_CP_EN = {8'hFF, 8'hFF, 8'h0F, 8'h0F};
    localparam [19:0] CFG_NB    = {5'd7, 5'd6, 5'd6, 5'd8};
    localparam [19:0] CFG_BASE  = {5'd9, 5'd5, 5'd1, 5'd0};

    // ── input_sram read port → wrapper (input_sram_asic instantiated there) ──
    assign input_rd_addr = cp_sram_rd_addr[11:0];

    // ── ping_pong_sram (macro variant) ───────────────────────────────────────
    wire [8:0] pp_rd_addr = (ctrl_layer_state == 3'd6) ? gap_rd_addr
                                                        : cp_sram_rd_addr[8:0];

    ping_pong_sram_asic u_pp (
        .clk     (clk),
        .bank_sel(ctrl_bank_sel),
        .wr_addr (ctrl_pong_addr[8:0]),
        .din     (cp_pong_din),
        .we      (cp_pong_we),
        .rd_addr (pp_rd_addr),
        .dout    (pp_dout)
    );

    // ── cp_engine (reused verbatim from hardware/RTL) ────────────────────────
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
        .sram_rd_addr_in  (ctrl_t),
        .cfg_base         (CFG_BASE)
    );

    // ── gap_fc_argmax (reused verbatim) ──────────────────────────────────────
    gap_fc_argmax u_gfa (
        .clk          (clk),
        .rst          (rst),
        .fc_sub_state (ctrl_fc_sub_state),
        .gap_step     (ctrl_gap_step),
        .fc_step      (ctrl_fc_step),
        .argmax_step  (ctrl_argmax_step),
        .ping_dout    (pp_dout),
        .out_ch_mask  (CFG_CP_EN[3*8 +: 8]),   // Conv4 active-output mask
        .gap_rd_addr  (gap_rd_addr),
        .result       (gap_result)
    );

    // ── cnn_controller (reused verbatim) ─────────────────────────────────────
    cnn_controller u_ctrl (
        .clk          (clk),
        .rst          (rst),
        .start        (start),
        .pool_write   (cp_pool_write),
        .cfg_in_ch    (CFG_IN_CH),
        .cfg_cp_en    (CFG_CP_EN),
        .cfg_nb       (CFG_NB),
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
