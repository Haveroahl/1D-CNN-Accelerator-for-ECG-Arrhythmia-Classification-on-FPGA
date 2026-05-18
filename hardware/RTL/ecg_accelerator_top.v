// ecg_accelerator_top.v
// Top-level: wires all sub-modules together.
//
// Modules instantiated:
//   avalon_slave      — Avalon-MM bridge (HPS ↔ accelerator)
//   input_sram        — 2500×8b, stores raw ECG input from HPS
//   ping_pong_sram    — 2 sets × 8 banks × 500 entries, inter-layer feature maps
//   cp_engine         — 8 CP blocks (Conv1..4)
//   gap_fc_argmax     — GAP/FC/Argmax post-processing
//   cnn_controller    — Unified FSM

module ecg_accelerator_top (
    input  wire        clk,
    input  wire        rst,         // synchronous reset (active high, for RTL)
    input  wire        rst_n,       // async active-low reset (for avalon_slave)

    // Avalon-MM Slave port (from HPS Lightweight bridge)
    input  wire [4:0]  avs_address,
    input  wire        avs_write,
    input  wire        avs_read,
    input  wire [31:0] avs_writedata,
    output wire [31:0] avs_readdata
);

    // ── Controller outputs ─────────────────────────────────────────────
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
    wire        ctrl_busy;
    wire        ctrl_done;
    wire [1:0]  ctrl_result;

    // ── Avalon slave outputs ───────────────────────────────────────────
    wire [11:0] avs_sram_wr_addr;
    wire [7:0]  avs_sram_din;
    wire        avs_sram_we;
    wire        avs_start;

    // ── Input SRAM ─────────────────────────────────────────────────────
    wire [7:0]  input_sram_dout;

    // ── Ping-Pong SRAM ─────────────────────────────────────────────────
    wire [63:0] pp_dout;            // Ping output packed: pp_dout[ch*8+:8]

    // ── CP engine ──────────────────────────────────────────────────────
    wire        cp_pool_write;      // from cp_engine (representative ch0)
    wire [8:0]  cp_pong_wr_addr;
    wire [63:0] cp_pong_din;        // packed: cp_pong_din[ch*8+:8]
    wire [7:0]  cp_pong_we;
    wire [11:0] cp_sram_rd_addr;

    // ── GAP/FC/Argmax ──────────────────────────────────────────────────
    wire [8:0]  gap_rd_addr;
    wire [1:0]  gap_result;
    wire        gap_done;           // unused at top — controller tracks via fc_sub_state

    // ── Pool write: cp_engine ch0 pool_write feeds controller ──────────
    // All active cp_blocks write simultaneously — ch0 is representative.
    // cp_engine exposes pong_we[0] as the representative pool_write signal.
    assign cp_pool_write = cp_pong_we[0];

    // ── avalon_slave ───────────────────────────────────────────────────
    avalon_slave u_avs (
        .clk          (clk),
        .rst_n        (rst_n),
        .avs_address  (avs_address),
        .avs_write    (avs_write),
        .avs_read     (avs_read),
        .avs_writedata(avs_writedata),
        .avs_readdata (avs_readdata),
        .sram_wr_addr (avs_sram_wr_addr),
        .sram_din     (avs_sram_din),
        .sram_we      (avs_sram_we),
        .start        (avs_start),
        .busy         (ctrl_busy),
        .done         (ctrl_done),
        .result       (ctrl_result)
    );

    // ── input_sram ─────────────────────────────────────────────────────
    input_sram u_isram (
        .clk    (clk),
        .wr_addr(avs_sram_wr_addr),
        .din    (avs_sram_din),
        .we     (avs_sram_we),
        .rd_addr(cp_sram_rd_addr[11:0]),
        .dout   (input_sram_dout)
    );

    // ── ping_pong_sram ─────────────────────────────────────────────────
    // Read address: cp_engine during CONV1..4, gap_fc_argmax during GAP_FC_S
    // Use gap_rd_addr when in GAP_FC_S state (layer_state == 3'd6), else cp_sram_rd_addr
    wire [8:0] pp_rd_addr = (ctrl_layer_state == 3'd6) ? gap_rd_addr
                                                        : cp_sram_rd_addr[8:0];

    ping_pong_sram u_pp (
        .clk     (clk),
        .bank_sel(ctrl_bank_sel),
        .wr_addr (cp_pong_wr_addr),
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
        .input_sram_dout  (input_sram_dout),
        .ping_dout        (pp_dout),
        .pong_wr_addr     (cp_pong_wr_addr),
        .pong_addr_in     (ctrl_pong_addr[8:0]),
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
        .gap_rd_addr  (gap_rd_addr),
        .result       (gap_result),
        .done         (gap_done)
    );

    // ── cnn_controller ─────────────────────────────────────────────────
    cnn_controller u_ctrl (
        .clk          (clk),
        .rst          (rst),
        .start        (avs_start),
        .pool_write   (cp_pool_write),
        .a            (ctrl_a),
        .t            (ctrl_t),
        .shift_en     (ctrl_shift_en),
        .sram_addr_en (),
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
        .busy         (ctrl_busy),
        .done         (ctrl_done),
        .result       (ctrl_result)
    );

endmodule
