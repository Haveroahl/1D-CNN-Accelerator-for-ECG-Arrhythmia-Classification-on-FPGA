// cp_accumulate_rescale.v
// Accumulate + bias + rescale + ReLU (S5→S8)
//
// Takes tree_out from cp_mac and produces an INT8 activation:
//   S5    : accumulate tree_out over in_ch cycles (per a_in). bias and round_add
//           (round-half-up) are FOLDED into the acc-init term (a_in==0) so the
//           downstream stages carry no extra add on the critical path.
//   S_bias: delay out_valid by 1 cycle, then capture acc directly (no adder —
//           acc already holds the completed sum one cycle after out_valid,
//           right before S5's a_in==0 branch overwrites it for the next
//           window). Replaces the old two-stage S5b+S_bias split: same total
//           pipeline depth (2 cycles from acc to biased), one fewer 32-bit
//           adder, bit-exact and cycle-count identical (verified).
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
    input  wire [3:0]            nb,             // rescale shift amount (0..15; max used = 8)
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
    assign round_add = (nb > 4'd0) ? (32'sd1 << (nb - 4'd1)) : 32'sd0;

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

    // ── S_bias: delay-then-capture (S5b folded in, adder removed) ────────────
    // Delay out_valid by 1 cycle, then capture acc directly (no recompute). At
    // that cycle, acc already holds the completed sum committed by S5 on the
    // previous edge — one cycle before S5's a_in==0 branch overwrites it for
    // the next accumulation window. No delay at all would read acc one cycle
    // too early (missing the final tap) — verified to break bit-exact from
    // Conv2 onward (in_ch>1). This single stage replaces the old S5b+S_bias
    // pair: identical 2-cycle depth from acc to biased, one 32-bit adder saved.
    reg signed [31:0] biased;
    reg               bias_valid;
    reg               out_valid_d1;
    always @(posedge clk) begin
        if (rst || pool_rst) begin
            bias_valid   <= 1'b0;
            biased       <= 32'sd0;
            out_valid_d1 <= 1'b0;
        end else begin
            out_valid_d1 <= out_valid;
            bias_valid   <= out_valid_d1;
            if (out_valid_d1)
                biased <= acc;
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
                           (shifted < -32'sd127) ? -8'sd127 : 8'(shifted[7:0]);
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
