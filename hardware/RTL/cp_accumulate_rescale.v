// cp_accumulate_rescale.v
// Accumulate + bias + rescale + ReLU (S5→S8)
//
// Takes tree_out from cp_mac and produces an INT8 activation:
//   S5    : accumulate tree_out over in_ch cycles (per a_in). bias and round_add
//           (round-half-up) are FOLDED into the acc-init term (a_in==0) so the
//           downstream stages carry no extra add on the critical path.
//   S5b   : acc_final register — breaks the 2-cascaded-adder critical path.
//   S_bias: pure register passthrough (bias already folded). Kept to preserve
//           pipeline depth + valid-bit timing.
//   S6    : pure arithmetic shift >>> nb (round-half-up, signed).
//   S7    : clamp to [-127, 127].
//   S8    : ReLU (Conv4 only).
// nb lives entirely in this module (both round_add fold and the >>> nb shift).

module cp_accumulate_rescale (
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  pool_rst,

    // MAC output
    input  wire signed [19:0]    tree_out,

    // Bias (INT32, from bias_rom)
    input  wire signed [31:0]    bias_in,

    // Controller signals
    input  wire [3:0]            a_in,           // channel counter delayed 5 cycles (a_d5)
    input  wire [3:0]            in_ch,          // IN_CH for current layer (1/4/4/8)
    input  wire                  compute_en_in,  // pipeline enable delayed 5 cycles (ce_d5)
    input  wire [4:0]            nb,             // rescale shift amount
    input  wire                  relu_en,        // 1 = Conv4 only

    // Outputs to cp_pool
    output reg signed [7:0]      relu_out,
    output reg                   relu_v
);

    // ── out_valid ──────────────────────────────────────────────────────────
    wire out_valid = compute_en_in && (a_in == in_ch - 4'd1);

    // ── S5: Accumulator ────────────────────────────────────────────────────
    reg signed [31:0] acc;
    wire signed [31:0] tree_sext = {{12{tree_out[19]}}, tree_out};

    // round_add (round-half-up bias) — folded into the accumulator init below.
    // nb is a per-layer constant → Quartus constant-folds.
    wire signed [31:0] round_add;
    assign round_add = (nb > 5'd0) ? (32'sd1 << (nb - 5'd1)) : 32'sd0;

    always @(posedge clk) begin
        if (rst || pool_rst) acc <= 32'sd0;
        else if (compute_en_in) begin
            // Fold per-layer constants (bias + round_add) into the init term so the
            // downstream S_bias add and S6 round-add drop off the critical chain.
            // Associative → bit-identical; bias≤139, round_add≤128, acc≤~4.19M ⟹ no signed-32 overflow.
            if (a_in == 0) acc <= tree_sext + bias_in + round_add;
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
                acc_final_r <= (a_in == 0) ? (tree_sext + bias_in + round_add)
                                           : (acc + tree_sext);
        end
    end

    // ── S_bias: pure register move (bias folded into acc-init at S5/S5b) ─────
    // Kept as a pipeline stage (not removed) to preserve depth + valid-bit timing.
    reg signed [31:0] biased;
    reg               bias_valid;
    always @(posedge clk) begin
        if (rst || pool_rst) begin
            bias_valid <= 1'b0;
            biased     <= 32'sd0;
        end else begin
            bias_valid <= acc_final_v;
            if (acc_final_v)
                biased <= acc_final_r;   // bias already in acc_final_r
        end
    end

    // ── S6: Rescale stage-1 (pure arithmetic shift, round-half-up) ──────────
    // round_add is folded into the accumulator init (S5/S5b), so this stage is
    // just a barrel shift — the +round add no longer sits on the critical path.
    reg signed [31:0] shifted;
    reg               rescale_v1;
    always @(posedge clk) begin
        if (rst || pool_rst) begin
            rescale_v1 <= 1'b0;
            shifted    <= 32'sd0;
        end else begin
            rescale_v1 <= bias_valid;
            if (bias_valid)
                shifted <= biased >>> nb;
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

endmodule
