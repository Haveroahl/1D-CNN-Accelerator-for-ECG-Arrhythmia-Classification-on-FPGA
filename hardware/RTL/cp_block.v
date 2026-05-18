// cp_block.v
// Conv-Pool Block — 1 output channel
//
// Pipeline: MULT(1) → TREE(3) → ACC(IN_CH) → ACC_FINAL(1) → BIAS(1) → RESCALE(2) → RELU(1) → POOL
// Rescale: (biased + (1 << (nb-1))) >>> nb  (round-half-up, signed) → clamp[-127, 127]
//
// Interface boundary:
//   - taps_in[0:4] : 5 tap values from cp_engine mux_s2 (2-cy delayed)
//   - a_in         : channel counter delayed 6 cycles from cp_engine (a_d6)
//   - compute_en_in: pipeline enable delayed 6 cycles from cp_engine (ce_d6)
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
    input  wire [39:0]           taps_in,    // taps_in[tap*8+:8]

    // Weight inputs (from w_packed, registered 1 cycle), packed 5×8b
    input  wire [39:0]           w,          // w[tap*8+:8]

    // Bias (INT32, from bias_rom)
    input  wire signed [31:0]    bias_in,

    // Controller signals
    input  wire [3:0]            a_in,           // channel counter delayed 6 cycles (a_d6)
    input  wire [3:0]            in_ch,          // IN_CH for current layer (1/4/4/8)
    input  wire                  compute_en_in,   // pipeline enable delayed 6 cycles
    input  wire [4:0]            nb,              // rescale shift amount
    input  wire                  relu_en,         // 1 = Conv4 only

    // Pool reset (layer transition)
    input  wire                  pool_rst,

    // Outputs
    output wire                  pool_write,      // write strobe (AND cp_en externally)
    output wire signed [7:0]     pool_out         // max-pooled value to Pong SRAM
);

    // ── out_valid ──────────────────────────────────────────────────────────
    wire out_valid = compute_en_in && (a_in == in_ch - 4'd1);

    // ── S1: 5 Multipliers (registered) ────────────────────────────────────
    reg signed [15:0] prod [0:4];
    always @(posedge clk) begin
        prod[0] <= $signed(taps_in[0*8 +: 8]) * $signed(w[0*8 +: 8]);
        prod[1] <= $signed(taps_in[1*8 +: 8]) * $signed(w[1*8 +: 8]);
        prod[2] <= $signed(taps_in[2*8 +: 8]) * $signed(w[2*8 +: 8]);
        prod[3] <= $signed(taps_in[3*8 +: 8]) * $signed(w[3*8 +: 8]);
        prod[4] <= $signed(taps_in[4*8 +: 8]) * $signed(w[4*8 +: 8]);
    end

    // ── S2: Adder stage-1 ──────────────────────────────────────────────────
    reg signed [16:0] sum01, sum23;
    reg signed [15:0] p4_d1;
    always @(posedge clk) begin
        sum01 <= prod[0] + prod[1];
        sum23 <= prod[2] + prod[3];
        p4_d1 <= prod[4];
    end

    // ── S3: Adder stage-2 ──────────────────────────────────────────────────
    reg signed [17:0] sum0123;
    reg signed [15:0] p4_d2;
    always @(posedge clk) begin
        sum0123 <= sum01 + sum23;
        p4_d2   <= p4_d1;
    end

    // ── S4: Adder stage-3 ──────────────────────────────────────────────────
    reg signed [19:0] tree_out;
    always @(posedge clk)
        tree_out <= sum0123 + p4_d2;

    // ── S5: Accumulator ────────────────────────────────────────────────────
    reg signed [31:0] acc;
    wire signed [31:0] tree_sext = {{12{tree_out[19]}}, tree_out};

    always @(posedge clk) begin
        if (rst || pool_rst) acc <= 32'sd0;
        else if (compute_en_in) begin
            if (a_in == 0) acc <= tree_sext;
            else           acc <= acc + tree_sext;
        end
    end

    // ── S5b: acc_final register (breaks 2-cascaded-adder critical path) ──────
    // acc_final_r = acc + tree_sext for last ic; registered 1 cycle after out_valid.
    reg signed [31:0] acc_final_r;
    reg               acc_final_v;
    always @(posedge clk) begin
        if (rst || pool_rst) begin
            acc_final_v <= 1'b0;
            acc_final_r <= 32'sd0;
        end else begin
            acc_final_v <= out_valid;
            if (out_valid)
                acc_final_r <= (a_in == 0) ? tree_sext : (acc + tree_sext);
        end
    end

    // ── S_bias: +Bias (registered) ─────────────────────────────────────────
    reg signed [31:0] biased;
    reg               bias_valid;
    always @(posedge clk) begin
        if (rst || pool_rst) begin
            bias_valid <= 1'b0;
            biased     <= 32'sd0;
        end else begin
            bias_valid <= acc_final_v;
            if (acc_final_v)
                biased <= acc_final_r + bias_in;
        end
    end

    // ── S6: Rescale stage-1 (shift + round-half-up) ────────────────────────
    // round_add precomputed as wire: separates subtract+shift from add+shift
    wire signed [31:0] round_add;
    assign round_add = (nb > 5'd0) ? (32'sd1 << (nb - 5'd1)) : 32'sd0;

    reg signed [31:0] shifted;
    reg               rescale_v1;
    always @(posedge clk) begin
        if (rst || pool_rst) begin
            rescale_v1 <= 1'b0;
            shifted    <= 32'sd0;
        end else begin
            rescale_v1 <= bias_valid;
            if (bias_valid)
                shifted <= (biased + round_add) >>> nb;
        end
    end

    // ── S7: Rescale stage-2 (clamp to [-127, 127]) ─────────────────────────
    reg signed [7:0] clamped;
    reg              rescale_v2;
    always @(posedge clk) begin
        if (rst || pool_rst) begin
            rescale_v2 <= 1'b0;
            clamped    <= 8'sd0;
        end else begin
            rescale_v2 <= rescale_v1;
            if (rescale_v1)
                clamped <= (shifted > 32'sd127)  ?  8'sd127 :
                           (shifted < -32'sd127) ? -8'sd127 : shifted[7:0];
        end
    end

    // ── S8: ReLU (Conv4 only) ──────────────────────────────────────────────
    reg signed [7:0] relu_out;
    reg              relu_v;
    always @(posedge clk) begin
        if (rst || pool_rst) begin
            relu_v   <= 1'b0;
            relu_out <= 8'sd0;
        end else begin
            relu_v <= rescale_v2;
            if (rescale_v2)
                relu_out <= (relu_en && clamped[7]) ? 8'sd0 : clamped;
        end
    end

    // ── S9: MaxPool rolling comparator ────────────────────────────────────
    reg signed [7:0] max_reg;
    reg [2:0]        pool_cnt;
    reg              pool_write_r;


    always @(posedge clk) begin
        pool_write_r <= 1'b0;
        if (rst || pool_rst) begin
            pool_cnt     <= 3'd0;
            pool_write_r <= 1'b0;
            max_reg      <= 8'sd0;   // clear max so X-state never propagates post-reset
        end else if (relu_v && compute_en_in) begin
            // Gate pool_cnt with compute_en_in (= ce_d6): only count when pipeline
            // carries valid data, not junk from SRW priming phase.
            if (pool_cnt == 3'd0)
                max_reg <= relu_out;
            else if (relu_out > max_reg)
                max_reg <= relu_out;

            if (pool_cnt == 3'd4) begin
                pool_cnt     <= 3'd0;
                pool_write_r <= 1'b1;
            end else begin
                pool_cnt <= pool_cnt + 3'd1;
            end
        end
    end

    assign pool_write = pool_write_r;
    assign pool_out   = max_reg;

endmodule
