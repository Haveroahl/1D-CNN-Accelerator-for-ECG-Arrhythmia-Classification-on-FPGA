// cp_block.v
// Conv-Pool Block — 1 output channel (thin wrapper over 3 submodules)
//
// Pipeline: MULT(1) → TREE(3) → ACC(IN_CH) → ACC_FINAL(1) → BIAS(1) → RESCALE(2) → RELU(1) → POOL
// Structurally split (no logic change, bit-exact) into:
//   cp_mac               — S1→S4 : 5 MULT + 3-stage adder tree → tree_out
//   cp_accumulate_rescale— S5→S8 : acc (fold bias+round) → acc_final → >>>nb → clamp → ReLU
//   cp_pool              — S9    : MaxPool rolling comparator
//
// Rescale: bias + round_add (1<<(nb-1)) are folded into the ACC init term (a_in==0);
//          S_bias is a passthrough and S6 is a pure >>> nb. Numerically identical to
//          (acc + bias + round) >>> nb (round-half-up, signed) → clamp[-127, 127].
//
// Interface boundary:
//   - x_in[0:4] : 5 tap values from cp_engine mux_s2 (2-cy delayed)
//   - a_in         : channel counter delayed 5 cycles from cp_engine (a_d5)
//   - compute_en_in: pipeline enable delayed 5 cycles from cp_engine (ce_d5)
//   - w[0:4]       : weights from cp_engine w_cur_flat (2-stage pipeline, arrive cycle N+2)
//   - bias_in      : INT32 bias from b_cur
//   - pool_write   : output — write strobe to Pong SRAM (AND with cp_en externally)
//   - pool_out     : output — max-pooled INT8 value

module cp_block #(
    parameter IN_CH_W = 4   // width of a_d5 counter (fixed 4-bit, matches cp_engine)
) (
    input  wire                  clk,
    input  wire                  rst,

    // Tap inputs from cp_engine (= mux_s1, already registered), packed 5×8b
    input  wire [39:0]           x_in,    // x_in[tap*8+:8]

    // Weight inputs (from w_packed, registered 1 cycle), packed 5×8b
    input  wire [39:0]           w,          // w[tap*8+:8]

    // Bias (INT32, from bias_rom)
    input  wire signed [31:0]    bias_in,

    // Controller signals
    input  wire [3:0]            a_in,           // channel counter delayed 5 cycles (a_d5)
    input  wire [3:0]            in_ch,          // IN_CH for current layer (1/4/4/8)
    input  wire                  compute_en_in,   // pipeline enable delayed 5 cycles (ce_d5)
    input  wire [3:0]            nb,              // rescale shift amount (0..15; max used = 8)
    input  wire                  relu_en,         // 1 = Conv4 only

    // Pool reset (layer transition)
    input  wire                  pool_rst,

    // Outputs
    output wire                  pool_write,      // write strobe (AND cp_en externally)
    output wire signed [7:0]     pool_out         // max-pooled value to Pong SRAM
);

    // ── S1→S4: MAC datapath ──────────────────────────────────────────────
    wire signed [19:0] tree_out;
    cp_mac u_mac (
        .clk      (clk),
        .x_in     (x_in),
        .w        (w),
        .tree_out (tree_out)
    );

    // ── S5→S8: Accumulate + bias + rescale + ReLU ──────────────────────────
    wire signed [7:0] relu_out;
    wire              relu_v;
    cp_accumulate_rescale u_accres (
        .clk           (clk),
        .rst           (rst),
        .pool_rst      (pool_rst),
        .tree_out      (tree_out),
        .bias_in       (bias_in),
        .a_in          (a_in),
        .in_ch         (in_ch),
        .compute_en_in (compute_en_in),
        .nb            (nb),
        .relu_en       (relu_en),
        .relu_out      (relu_out),
        .relu_v        (relu_v)
    );

    // ── S9: MaxPool ─────────────────────────────────────────────────────────
    cp_pool u_pool (
        .clk           (clk),
        .rst           (rst),
        .pool_rst      (pool_rst),
        .relu_out      (relu_out),
        .relu_v        (relu_v),
        .compute_en_in (compute_en_in),
        .pool_write    (pool_write),
        .pool_out      (pool_out)
    );

endmodule
