// cp_mac.v
// Conv MAC datapath (S1→S4): 5 signed multipliers → 3-stage adder tree → tree_out.
// Feed-forward only; dot-product of 5 taps × 5 weights, 4 cycles latency.
// All two's-complement; $signed maps `*` to DSP18 (FPGA) / Baugh-Wooley (ASIC).
// Widths grow 16→17→18→20 (sign-extended) to avoid overflow summing 5 products.

module cp_mac (
    input  wire                  clk,

    input  wire [39:0]           x_in,    // 5 window samples, packed 5×8b
    input  wire [39:0]           w,          // 5 weights, packed 5×8b
    output reg signed [19:0]     tree_out
);

    // ── S1: 5 signed multipliers, 8×8 → 16b (lane k = bits [k*8 +: 8]) ─────
    // $signed() casts each unsigned slice to two's complement at the multiply.
    // Equivalent alternative: declare the lanes as `wire signed [7:0]` and drop
    // the per-op cast (signed then comes from the declaration). LHS prod is 16b,
    // so the multiply is signed 8b × signed 8b → signed 16b.
    reg signed [15:0] prod0, prod1, prod2, prod3, prod4;
    always @(posedge clk) begin
        prod0 <= $signed(x_in[ 7: 0]) * $signed(w[ 7: 0]);
        prod1 <= $signed(x_in[15: 8]) * $signed(w[15: 8]);
        prod2 <= $signed(x_in[23:16]) * $signed(w[23:16]);
        prod3 <= $signed(x_in[31:24]) * $signed(w[31:24]);
        prod4 <= $signed(x_in[39:32]) * $signed(w[39:32]);
    end

    // ── S2: Adder stage-1 (16b + 16b → 17b; operands sign-extended to 17b) ──
    reg signed [16:0] sum01, sum23;
    reg signed [15:0] p4_d1;
    always @(posedge clk) begin
        sum01 <= {prod0[15], prod0} + {prod1[15], prod1};
        sum23 <= {prod2[15], prod2} + {prod3[15], prod3};
        p4_d1 <= prod4;
    end

    // ── S3: Adder stage-2 (17b + 17b → 18b; operands sign-extended to 18b) ──
    reg signed [17:0] sum0123;
    reg signed [15:0] p4_d2;
    always @(posedge clk) begin
        sum0123 <= {sum01[16], sum01} + {sum23[16], sum23};
        p4_d2   <= p4_d1;
    end

    // ── S4: Adder stage-3 (18b + 16b → 20b; both sign-extended to 20b) ──────
    always @(posedge clk)
        tree_out <= {{2{sum0123[17]}}, sum0123} + {{4{p4_d2[15]}}, p4_d2};

endmodule
