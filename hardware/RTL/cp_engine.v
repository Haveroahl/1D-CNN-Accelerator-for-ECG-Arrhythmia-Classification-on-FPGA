// cp_engine.v
// 8 CP blocks running in parallel for 8 output channels
//
// Responsibilities:
//   1. SRW array: 8 shift-register windows (5-tap × 8-bit), one per input channel
//   2. MUX: select SRW[a] each cycle → mux_s1 (1-stage, broadcast to all 8 cp_blocks)
//   3. Delay chain: a → a_d6, compute_en → ce_d6  (6 cycles: MUX_reg + WROM_reg + MULT + TREE×3)
//   4. SRAM read address generation for Ping SRAM and Input SRAM
//   5. 8 cp_block instances, weight from per-layer FF arrays (40b/word = 5 taps packed)
//   6. Pool write gating: pool_write && cp_en[oc]
//
// Weight storage: 4 FF arrays per layer, each entry = 40-bit packed 5 taps.
//   w_rom_conv1[0:3]   — 4 oc × 1 ic         (4 entries)
//   w_rom_conv2[0:15]  — 4 oc × 4 ic         (16 entries)  addr = oc*4 + ic
//   w_rom_conv3[0:31]  — 8 oc × 4 ic         (32 entries)  addr = oc*4 + ic
//   w_rom_conv4[0:63]  — 8 oc × 8 ic         (64 entries)  addr = oc*8 + ic
//   Loaded from conv1_w.hex .. conv4_w.hex (40b/line hex, MSB tap4 ... LSB tap0)
//   FF array (no ramstyle): async read, no port limit, ~185 FF total — deterministic timing.
//
// Bias: b_store[oc*4 + layer_idx], INT32 little-endian, from conv_bias.hex (32 entries)
//
// Note: ping_pong_sram and input_sram are instantiated in ecg_accelerator_top.
//       cp_engine receives dout from those SRAMs as inputs.
//
// Pipeline latency from a to acc input: 6 cycles
//   cy N   : a → mux_comb (SRW async) + w_comb (ROM async, 8:1 MUX per oc)
//   cy N+1 : mux_s1 ← mux_comb;  w_packed ← w_comb  (both FF registered)
//   cy N+2 : prod ← mux_s1 * w_packed (S1 MULT)
//   cy N+3 : sum01/sum23 (S2)
//   cy N+4 : sum0123 (S3)
//   cy N+5 : tree_out (S4)
//   cy N+6 : acc updates with a_d6 (S5)

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
    input  wire [4:0]  nb,           // rescale shift per layer
    input  wire        relu_en,      // 1 = Conv4 only
    input  wire [7:0]  cp_en,        // bitmask: which output channels are active
    input  wire [2:0]  layer_state,  // CONV1=2, CONV2=3, CONV3=4, CONV4=5 (match controller)
    input  wire        pool_rst,     // reset pool_cnt on layer transition

    // Data from SRAMs (driven by top-level)
    input  wire [7:0]  input_sram_dout,     // from input_sram (Conv1 only)
    input  wire [63:0] ping_dout,           // from ping_pong_sram: ping_dout[ch*8+:8]

    // Pong SRAM write interface (to ping_pong_sram write port)
    output wire [8:0]  pong_wr_addr,        // shared write address (from controller)
    input  wire [8:0]  pong_addr_in,        // pong_addr from controller
    output wire [63:0] pong_din,            // per-channel write data: pong_din[ch*8+:8]
    output wire [7:0]  pong_we,             // per-channel write enable

    // SRAM read address output (to top-level, feeds both input_sram and ping_pong_sram)
    output wire [11:0] sram_rd_addr,        // driven by t and rp logic in controller
    input  wire [11:0] sram_rd_addr_in      // sram_rd_addr from controller
);

    // ── Layer state encoding (must match cnn_controller) ──────────────────
    localparam CONV1 = 3'd2;
    localparam CONV2 = 3'd3;
    localparam CONV3 = 3'd4;
    localparam CONV4 = 3'd5;

    // ── SRAM read address: t - 2 (padding offset) ─────────────────────────
    assign sram_rd_addr = (sram_rd_addr_in >= 12'd2) ? (sram_rd_addr_in - 12'd2) : 12'd0;
    assign pong_wr_addr = pong_addr_in;

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

    // ── Delay chain: a, in_ch and compute_en delayed 6 cycles ────────────────────
    // MUX_reg(1) + WROM_reg(1) + MULT(1) + TREE(3) = 6 cycles before ACC
    //   cy N   : a → mux_comb + w_rom read (combinational)
    //   cy N+1 : mux_s1, w_packed registered
    //   cy N+2 : prod (S1 MULT)
    //   cy N+3 : sum01/sum23 (S2)
    //   cy N+4 : sum0123 (S3)
    //   cy N+5 : tree_out (S4)
    //   cy N+6 : acc updates → needs a_d6 (a value from cycle N)
    reg [3:0] a_d1, a_d2, a_d3, a_d4, a_d5, a_d6;
    reg [3:0] inch_d1, inch_d2, inch_d3, inch_d4, inch_d5, inch_d6;
    reg       ce_d1, ce_d2, ce_d3, ce_d4, ce_d5, ce_d6;
    always @(posedge clk) begin
        if (srw_rst) begin
            a_d1 <= 4'd0; a_d2 <= 4'd0; a_d3 <= 4'd0; a_d4 <= 4'd0; a_d5 <= 4'd0; a_d6 <= 4'd0;
            inch_d1 <= 4'd0; inch_d2 <= 4'd0; inch_d3 <= 4'd0;
            inch_d4 <= 4'd0; inch_d5 <= 4'd0; inch_d6 <= 4'd0;
            ce_d1 <= 1'b0; ce_d2 <= 1'b0; ce_d3 <= 1'b0;
            ce_d4 <= 1'b0; ce_d5 <= 1'b0; ce_d6 <= 1'b0;
        end else begin
            a_d1 <= a;      a_d2 <= a_d1;  a_d3 <= a_d2;  a_d4 <= a_d3;  a_d5 <= a_d4;  a_d6 <= a_d5;
            inch_d1 <= in_ch;  inch_d2 <= inch_d1; inch_d3 <= inch_d2;
            inch_d4 <= inch_d3; inch_d5 <= inch_d4; inch_d6 <= inch_d5;
            ce_d1 <= compute_en; ce_d2 <= ce_d1; ce_d3 <= ce_d2;
            ce_d4 <= ce_d3;      ce_d5 <= ce_d4; ce_d6 <= ce_d5;
        end
    end

    // ── Layer index ────────────────────────────────────────────────────────
    // CONV1=2→0, CONV2=3→1, CONV3=4→2, CONV4=5→3
    wire [1:0] layer_idx = layer_state[1:0] - 2'd2;

    // ── Weight arrays: 4 per-layer, packed 5-tap (40-bit) ───────────────────
    // FF array (no ramstyle): Quartus infers async read register file.
    // Total: (4+16+32+64)×40b = 4640b ≈ 185 FF — 0.1% of Cyclone V FF budget.
    // Async read: w_comb[oc] combinational from addr → 1 FF stage (w_packed) →
    // arrives at S1 MULT same cycle as mux_s1. No port replication needed.
    reg [39:0] w_rom_conv1 [0:3];    // 4 oc × 1 ic
    reg [39:0] w_rom_conv2 [0:15];   // 4 oc × 4 ic
    reg [39:0] w_rom_conv3 [0:31];   // 8 oc × 4 ic
    reg [39:0] w_rom_conv4 [0:63];   // 8 oc × 8 ic

    // Bias: INT32, 8 oc × 4 layer = 32 entries, addr = oc*4 + layer_idx
    (* ramstyle = "MLAB" *) reg signed [31:0] b_store [0:31];

    initial begin
        $readmemh("conv1_w.hex", w_rom_conv1);
        $readmemh("conv2_w.hex", w_rom_conv2);
        $readmemh("conv3_w.hex", w_rom_conv3);
        $readmemh("conv4_w.hex", w_rom_conv4);
        $readmemh("conv_bias.hex", b_store);
    end

    // ── Weight combinational MUX: w_rom[oc, a] → w_comb (async) ─────────
    // Mirror of SRW mux_comb pattern: async select → 1 FF stage (w_packed).
    // Per-oc: 4:1 layer MUX + 8:1 ic MUX (a[2:0]) — all combinational, ~2 LUT levels.
    // oc=0..1: valid in Conv1..4; oc=2..3: Conv2..4; oc=4..7: Conv3..4.
    wire [39:0] w_comb [0:7];

    // Index widths per ROM depth (Quartus needs explicit width on array index):
    //   conv1[0:3]   → 2-bit  (oc only)
    //   conv2[0:15]  → 4-bit  (oc[1:0] || a[1:0])
    //   conv3[0:31]  → 5-bit  (oc[2:0] || a[1:0])
    //   conv4[0:63]  → 6-bit  (oc[2:0] || a[2:0])

    assign w_comb[0] = (layer_state == CONV1) ? w_rom_conv1[2'd0]                :
                       (layer_state == CONV2) ? w_rom_conv2[{2'd0, a[1:0]}]      :
                       (layer_state == CONV3) ? w_rom_conv3[{3'd0, a[1:0]}]      :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd0, a[2:0]}]      : 40'd0;

    assign w_comb[1] = (layer_state == CONV1) ? w_rom_conv1[2'd1]                :
                       (layer_state == CONV2) ? w_rom_conv2[{2'd1, a[1:0]}]      :
                       (layer_state == CONV3) ? w_rom_conv3[{3'd1, a[1:0]}]      :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd1, a[2:0]}]      : 40'd0;

    assign w_comb[2] = (layer_state == CONV1) ? w_rom_conv1[2'd2]                :
                       (layer_state == CONV2) ? w_rom_conv2[{2'd2, a[1:0]}]      :
                       (layer_state == CONV3) ? w_rom_conv3[{3'd2, a[1:0]}]      :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd2, a[2:0]}]      : 40'd0;

    assign w_comb[3] = (layer_state == CONV1) ? w_rom_conv1[2'd3]                :
                       (layer_state == CONV2) ? w_rom_conv2[{2'd3, a[1:0]}]      :
                       (layer_state == CONV3) ? w_rom_conv3[{3'd3, a[1:0]}]      :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd3, a[2:0]}]      : 40'd0;

    assign w_comb[4] = (layer_state == CONV3) ? w_rom_conv3[{3'd4, a[1:0]}]             :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd4, a[2:0]}]             : 40'd0;

    assign w_comb[5] = (layer_state == CONV3) ? w_rom_conv3[{3'd5, a[1:0]}]             :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd5, a[2:0]}]             : 40'd0;

    assign w_comb[6] = (layer_state == CONV3) ? w_rom_conv3[{3'd6, a[1:0]}]             :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd6, a[2:0]}]             : 40'd0;

    assign w_comb[7] = (layer_state == CONV3) ? w_rom_conv3[{3'd7, a[1:0]}]             :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd7, a[2:0]}]             : 40'd0;

    // ── w_packed: register w_comb (1 FF stage, aligns with mux_s1) ───────
    reg [39:0] w_packed [0:7];
    always @(posedge clk) begin
        w_packed[0] <= w_comb[0];
        w_packed[1] <= w_comb[1];
        w_packed[2] <= w_comb[2];
        w_packed[3] <= w_comb[3];
        w_packed[4] <= w_comb[4];
        w_packed[5] <= w_comb[5];
        w_packed[6] <= w_comb[6];
        w_packed[7] <= w_comb[7];
    end

    // ── Bias: registered per layer (only changes on layer transition) ─────
    reg signed [31:0] b_cur [0:7];
    integer bi;
    always @(posedge clk) begin
        for (bi = 0; bi < 8; bi = bi + 1)
            b_cur[bi] <= b_store[{bi[2:0], layer_idx}];  // 5-bit index: oc[2:0]*4 + layer_idx[1:0]
    end

    // ── 8 CP block instances ───────────────────────────────────────────────
    wire        cp_pool_write [0:7];
    wire signed [7:0] cp_pool_out [0:7];

    genvar oc;
    generate
        for (oc = 0; oc < 8; oc = oc + 1) begin : cp_blocks
            cp_block u_cp (
                .clk           (clk),
                .rst           (rst),
                .taps_in       (mux_s1),
                .w             (w_packed[oc]),
                .bias_in       (b_cur[oc]),
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
