// cp_engine.v
// 8 CP blocks running in parallel for 8 output channels
//
// Responsibilities:
//   1. SRW array: 8 shift-register windows (5-tap × 8-bit), one per input channel
//   2. MUX: select SRW[a] each cycle → mux_s1 (1-stage, broadcast to all 8 cp_blocks)
//   3. Delay chain: a → a_d5, compute_en → ce_d5  (5 cycles: mux_s1 + MULT + TREE×3)
//   4. SRAM read address generation for Ping SRAM and Input SRAM
//   5. 8 cp_block instances, weight from 8 per-oc M10K RAMs (40b/word = 5 taps packed)
//   6. Pool write gating: pool_write && cp_en[oc]
//
// Weight storage (Phase B01 — runtime-loadable): 8 per-oc M10K RAMs, 40b × 32.
//   w_ram0..7[0:31]  — one RAM per output channel; word = layer_base + ic.
//     Default Chapman: Conv1 base=0 (1w) Conv2 base=1 (4w) Conv3 base=5 (4w)
//     Conv4 base=9 (8w) = 17 words. Depth 32 covers the MAX topology
//     in_ch=(8,8,8,8) (bases {0,8,16,24}, top word 31) for runtime reconfig.
//   Read: 8 RAMs read the SAME word {layer_base+a} → 8 oc weights in parallel.
//     M10K is SYNC-read: addr at cy N → q at cy N+1. That q register replaces
//     the old async-ROM + w_packed FF stage, so pipeline alignment is identical.
//   Write: w_wr_oc selects the RAM, full 40b/word (host sends lo+hi 32b halves).
//   Init: $readmemh w_ram0..7.hex (unless +define+NO_WEIGHT_INIT); bus overrides.
//
// Bias: b_store[oc*4 + layer_idx], INT32 little-endian, from conv_bias.hex (32 entries)
//
// Note: ping_pong_sram and input_sram are instantiated in ecg_accelerator_top.
//       cp_engine receives dout from those SRAMs as inputs.
//
// Pipeline latency from mux_comb to acc register edge: 5 cycles
//   cy N   : a → mux_comb (SRW async) + w_rd_word = layer_base+a (M10K rd addr)
//   cy N+1 : mux_s1 ← mux_comb;  w_packed ← w_ram[w_rd_word] (M10K sync-read q)
//   cy N+2 : prod ← mux_s1 * w_packed (S1 MULT)
//   cy N+3 : sum01/sum23 (S2)
//   cy N+4 : sum0123 (S3)
//   cy N+5 : tree_out (S4) — acc edge reads a_d5/ce_d5/inch_d5 to match a from cy N

module cp_engine (
    input  wire        clk,
    input  wire        rst,

    // Controller signals
    input  wire [3:0]  a,            // channel counter 0..IN_CH-1
    input  wire [3:0]  in_ch,        // IN_CH for current layer: 1/4/4/8
    input  wire [11:0] in_len,       // IN_LEN for current layer (2500/500/100/20)
    input  wire        shift_en,     // = (a == IN_CH-1)
    input  wire        srw_rst,      // SRW clear pulse (layer transition)
    input  wire        compute_en,   // pipeline enable (0 during pre-fetch)
    input  wire [3:0]  nb,           // rescale shift per layer (0..15; max used = 8)
    input  wire        relu_en,      // 1 = Conv4 only
    input  wire [7:0]  cp_en,        // bitmask: which output channels are active
    input  wire [2:0]  layer_state,  // CONV1=2, CONV2=3, CONV3=4, CONV4=5 (match controller)
    input  wire        pool_rst,     // reset pool_cnt on layer transition

    // Data from SRAMs (driven by top-level)
    input  wire [7:0]  input_sram_dout,     // from input_sram (Conv1 only)
    input  wire [63:0] ping_dout,           // from ping_pong_sram: ping_dout[ch*8+:8]

    // Pong SRAM write interface (to ping_pong_sram write port)
    // Note: write address is driven directly from cnn_controller.pong_addr at top-level
    // (no logic needed inside cp_engine — saved a passthrough port).
    output wire [63:0] pong_din,            // per-channel write data: pong_din[ch*8+:8]
    output wire [7:0]  pong_we,             // per-channel write enable

    // SRAM read address output (to top-level, feeds both input_sram and ping_pong_sram)
    output wire [11:0] sram_rd_addr,        // driven by t and rp logic in controller
    input  wire [11:0] sram_rd_addr_in,     // sram_rd_addr from controller

    // ── Weight write port (from bus adapter, Phase B01 runtime reload) ──────
    // Conv weights live in 8 per-oc M10K RAMs (one per output channel), each
    // 40-bit × 32 words (word = layer_base + ic). To load one 40-bit entry the
    // host issues TWO 32-bit writes (lo = bits[31:0], hi = bits[39:32]); the bus
    // adapter assembles them and pulses w_wr_en for the whole 40-bit word.
    //   w_wr_oc   : which of the 8 per-oc RAMs to write (0..7)
    //   w_wr_word : RAM word index (0..31) = layer_base + ic
    //   w_wr_data : full 40-bit packed 5-tap entry
    // Bias write (b_store FF, 32 × INT32): b_wr_en + b_wr_addr + b_wr_data.
    input  wire        w_wr_en,
    input  wire [2:0]  w_wr_oc,
    input  wire [4:0]  w_wr_word,
    input  wire [39:0] w_wr_data,
    input  wire        b_wr_en,
    input  wire [4:0]  b_wr_addr,
    input  wire [31:0] b_wr_data,

    // ── Topology config: weight-RAM word base per layer (4 × 5-bit) ─────────
    // Replaces the hard-coded {0,1,5,9} bases so a runtime-reconfigured channel
    // layout (driver-loaded) lands its weights at the correct RAM words.
    input  wire [19:0] cfg_base
);

    // ── Layer state encoding (must match cnn_controller) ──────────────────
    localparam CONV1 = 3'd2;
    localparam CONV2 = 3'd3;
    localparam CONV3 = 3'd4;
    localparam CONV4 = 3'd5;

    // ── SRAM read address: t - 2 (padding offset) ─────────────────────────
    assign sram_rd_addr = (sram_rd_addr_in >= 12'd2) ? (sram_rd_addr_in - 12'd2) : 12'd0;

    // ── SRW[0..7]: 5-tap shift register per input channel ─────────────────
    // 1D flat (Verilog-2001): srw_flat[ch*5 + slot]
    //   Physical slot order (hardware shift register):
    //     slot 0 = newest sample (just arrived from SRAM)
    //     slot 4 = oldest sample (4 shifts ago)
    //   Logical tap index (PyTorch cross-correlation pairing) is re-mapped at the MUX:
    //     mux_comb[k=0] reads slot 4 (oldest) → pairs with w[k=0] · x[t-2]
    //     mux_comb[k=4] reads slot 0 (newest) → pairs with w[k=4] · x[t+2]
    //   This matches PyTorch F.conv1d: out[t] = Σ_k w[k] · x[t-2+k]
    reg signed [7:0] srw_flat [0:39];  // 8ch × 5slot

    // Zero-padding logic:
    //   Front pad: sram_rd_addr_in < 2  → rd_addr would be negative (clamped to 0, garbage)
    //   Back pad : sram_rd_addr_in >= in_len + 2  → rd_addr >= in_len (out of valid data range)
    //
    // The pad signal must align with SRAM dout (which has 1-cycle latency from
    // rd_addr drive). We compute pad_zero_pre from ctrl_t (= drive cycle) then
    // register it once so pad_zero_r arrives at SRW input the same cycle as dout.
    wire pad_zero_pre = (sram_rd_addr_in < 12'd2) ||
                        (sram_rd_addr_in >= (in_len + 12'd2));
    reg  pad_zero_r;
    always @(posedge clk) begin
        if (srw_rst) pad_zero_r <= 1'b1;
        else         pad_zero_r <= pad_zero_pre;
    end

    // Top-level MUX: Conv1 reads input_sram, Conv2..4 read ping_pong_sram
    wire [7:0] srw_din [0:7];
    assign srw_din[0] = pad_zero_r ? 8'h00
                      : (layer_state == CONV1) ? input_sram_dout : ping_dout[0*8 +: 8];
    assign srw_din[1] = pad_zero_r ? 8'h00 : ping_dout[1*8 +: 8];
    assign srw_din[2] = pad_zero_r ? 8'h00 : ping_dout[2*8 +: 8];
    assign srw_din[3] = pad_zero_r ? 8'h00 : ping_dout[3*8 +: 8];
    assign srw_din[4] = pad_zero_r ? 8'h00 : ping_dout[4*8 +: 8];
    assign srw_din[5] = pad_zero_r ? 8'h00 : ping_dout[5*8 +: 8];
    assign srw_din[6] = pad_zero_r ? 8'h00 : ping_dout[6*8 +: 8];
    assign srw_din[7] = pad_zero_r ? 8'h00 : ping_dout[7*8 +: 8];

    integer ch;
    always @(posedge clk) begin
        if (srw_rst) begin
            for (ch = 0; ch < 40; ch = ch + 1)
                srw_flat[ch] <= 8'sd0;
        end else if (shift_en) begin
            for (ch = 0; ch < 8; ch = ch + 1) begin
                srw_flat[ch*5 + 4] <= srw_flat[ch*5 + 3];
                srw_flat[ch*5 + 3] <= srw_flat[ch*5 + 2];
                srw_flat[ch*5 + 2] <= srw_flat[ch*5 + 1];
                srw_flat[ch*5 + 1] <= srw_flat[ch*5 + 0];
                srw_flat[ch*5 + 0] <= srw_din[ch];
            end
        end
    end

    // ── MUX: select srw_flat[a] → mux_comb → mux_s1 (1-stage) ───────────
    // Re-index physical slots to logical kernel taps (PyTorch cross-correlation order):
    //   mux_comb[k] pairs with w[k]; PyTorch w[0]·x[t-2], w[4]·x[t+2]
    //   → mux_comb[0] = oldest (slot 4), mux_comb[4] = newest (slot 0)
    wire signed [7:0] mux_comb [0:4];
    assign mux_comb[0] = srw_flat[a[2:0]*5 + 4];  // oldest = x[t-2]
    assign mux_comb[1] = srw_flat[a[2:0]*5 + 3];
    assign mux_comb[2] = srw_flat[a[2:0]*5 + 2];  // center = x[t]
    assign mux_comb[3] = srw_flat[a[2:0]*5 + 1];
    assign mux_comb[4] = srw_flat[a[2:0]*5 + 0];  // newest = x[t+2]

    reg [39:0] mux_s1;   // packed 5×8b, cycle N+1, matches w_packed arrival
    integer mi;
    always @(posedge clk) begin
        for (mi = 0; mi < 5; mi = mi + 1)
            mux_s1[mi*8 +: 8] <= mux_comb[mi];
    end

    // ── Delay chain: a, in_ch, compute_en delayed 5 cycles ──────────────────
    // mux_s1(1) + MULT(1) + TREE(3) = 5 cycles. d5 outputs feed cp_block ports
    // a_in / in_ch / compute_en_in so that at the acc-register edge cy N+5 the
    // conditional matches the a value that drove mux_comb at cy N.
    reg [3:0] a_d1, a_d2, a_d3, a_d4, a_d5;
    reg [3:0] inch_d1, inch_d2, inch_d3, inch_d4, inch_d5;
    reg       ce_d1, ce_d2, ce_d3, ce_d4, ce_d5;
    always @(posedge clk) begin
        if (srw_rst) begin
            a_d1 <= 4'd0; a_d2 <= 4'd0; a_d3 <= 4'd0; a_d4 <= 4'd0; a_d5 <= 4'd0;
            inch_d1 <= 4'd0; inch_d2 <= 4'd0; inch_d3 <= 4'd0;
            inch_d4 <= 4'd0; inch_d5 <= 4'd0;
            ce_d1 <= 1'b0; ce_d2 <= 1'b0; ce_d3 <= 1'b0;
            ce_d4 <= 1'b0; ce_d5 <= 1'b0;
        end else begin
            a_d1 <= a;      a_d2 <= a_d1;  a_d3 <= a_d2;  a_d4 <= a_d3;  a_d5 <= a_d4;
            inch_d1 <= in_ch;  inch_d2 <= inch_d1; inch_d3 <= inch_d2;
            inch_d4 <= inch_d3; inch_d5 <= inch_d4;
            ce_d1 <= compute_en; ce_d2 <= ce_d1; ce_d3 <= ce_d2;
            ce_d4 <= ce_d3;      ce_d5 <= ce_d4;
        end
    end

    // ── Conv weight + bias storage (factored into cp_weight_store) ──────────
    // Two build variants — IDENTICAL ports, IDENTICAL N+1 timing → bit-exact:
    //   default            V2 (Phase B01): 8× M10K weight RAM + bus reload (cfg_base)
    //   +define+WEIGHT_ROM V1: hard FF-array ROM, no bus, fixed Chapman topology
    // Outputs are flattened (Verilog-2001 array-port restriction) → unpacked at
    // the cp_block instances via [oc*40 +: 40] / [oc*32 +: 32] slices.
    wire [319:0] w_packed_flat;   // 8 × 40b
    wire [255:0] b_cur_flat;      // 8 × 32b
    cp_weight_store u_wstore (
        .clk           (clk),
        .rst           (rst),
        .layer_state   (layer_state),
        .a             (a),
        .cfg_base      (cfg_base),
        .w_wr_en       (w_wr_en),
        .w_wr_oc       (w_wr_oc),
        .w_wr_word     (w_wr_word),
        .w_wr_data     (w_wr_data),
        .b_wr_en       (b_wr_en),
        .b_wr_addr     (b_wr_addr),
        .b_wr_data     (b_wr_data),
        .w_packed_flat (w_packed_flat),
        .b_cur_flat    (b_cur_flat)
    );

    // ── 8 CP block instances ───────────────────────────────────────────────
    wire        cp_pool_write [0:7];
    wire signed [7:0] cp_pool_out [0:7];

    genvar oc;
    generate
        for (oc = 0; oc < 8; oc = oc + 1) begin : cp_blocks
            cp_block u_cp (
                .clk           (clk),
                .rst           (rst),
                .x_in          (mux_s1),
                .w             (w_packed_flat[oc*40 +: 40]),
                .bias_in       (b_cur_flat[oc*32 +: 32]),
                // Pipeline depth from mux_comb to acc-register-update edge = 5 cycles
                // (mux_s1, prod, sum01/23, sum0123, tree_out). At edge cy N+5, the
                // acc conditional reads signals that must match mux_comb cy N → use d5.
                .a_in          (a_d5),
                .in_ch         (inch_d5),
                .compute_en_in (ce_d5),
                .nb            (nb),
                .relu_en       (relu_en),
                .pool_rst      (pool_rst),
                .pool_write    (cp_pool_write[oc]),
                .pool_out      (cp_pool_out[oc])
            );
        end
    endgenerate

    // ── Pong write port — gate pool_write with cp_en ───────────────────────
    generate
        for (oc = 0; oc < 8; oc = oc + 1) begin : pong_wr
            assign pong_we[oc]           = cp_pool_write[oc] && cp_en[oc];
            assign pong_din[oc*8 +: 8]   = cp_pool_out[oc];
        end
    endgenerate

endmodule
