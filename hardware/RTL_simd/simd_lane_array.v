// simd_lane_array.v
// SIMD-20 position-parallel MAC + accumulate + requantize for ONE output channel.
//
// 20 lanes run in lockstep. Lane l computes output position (block_base + l) of the
// CURRENT output channel oc, accumulating over input channels a = 0..IN_CH-1.
//   - taps_in[l] : 5 tap values for lane l, window [l..l+4] of line-buffer[ic=a]
//                  (static wiring from line_buffer_engine — NO dynamic channel MUX).
//   - w          : 5 weights w[oc][a], broadcast (same) to all 20 lanes.
//   - acc[l]     : accumulates 5-tap dot product across a; bias+round folded at a==0.
//   - requantize : >>nb, clamp[-127,127], ReLU(conv4) — 20 independent copies, 0 DSP.
//
// BIT-EXACT contract — copied verbatim from production cp_block.v (SIMD.md §12):
//   S1 MULT(reg) -> TREE 3-stage(reg) -> ACC(fold bias+round at a==0) -> S6 >>>nb
//   -> S7 clamp -> S8 ReLU. round_add = (nb>0)?(1<<(nb-1)):0. tree_sext sign-ext 20->32.
//   The ONLY difference vs cp_block: NO rolling MaxPool stage (S9) — 20 lanes emit
//   synchronously, pooling is combinational in simd_pool.v.
//
// Pipeline depth mux->acc-edge = 5 cycles (taps_s1 + MULT + TREE×3), identical to
// production. The array delays a/compute_en/in_ch by 5 internally so the acc
// conditional at edge cy N+5 matches the taps/weight that drove cy N.
//
// Weight fan-out (SIMD.md §2b): W1 default = replicate w into 4 group registers
// (lanes 0-4, 5-9, 10-14, 15-19) -> fan-out 25/copy. Set W3_REPLICATE for 20 copies
// (1/lane, fan-out 5 = production) if synth weight path fails. Numerically identical.

module simd_lane_array #(
    parameter L         = 20,
    parameter W3_REPLICATE = 0   // 0 = W1 (4 group copies), 1 = W3 (20 lane copies)
) (
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  blk_rst,        // per-block clear (acc + valid pipe)

    // Tap inputs: 20 lanes × 5 taps × 8b, packed taps_in[l*40 + tap*8 +: 8]
    input  wire [L*40-1:0]       taps_in,        // already registered (= cp_engine mux_s1)
    // Weight: 5 taps × 8b, w[tap*8 +: 8] — broadcast to all lanes
    input  wire [39:0]           w,              // already registered (= cp_engine w_packed)
    // Bias INT32 for current oc
    input  wire signed [31:0]    bias_in,

    // Control (RAW, undelayed — array applies the 5-cy delay internally)
    input  wire [3:0]            a,              // in-channel counter 0..IN_CH-1
    input  wire [3:0]            in_ch,          // IN_CH for layer (1/4/4/8)
    input  wire                  compute_en,     // pipeline enable
    input  wire [4:0]            nb,             // rescale shift
    input  wire                  relu_en,        // 1 = Conv4

    // Outputs: 20 lane INT8 results + valid strobe (1 cy, all lanes together)
    output wire [L*8-1:0]        lane_out,       // lane_out[l*8 +: 8]
    output wire                  lane_valid      // result row ready this cycle
);

    // ── round_add (round-half-up bias), folded into acc init (Fix B) ─────────
    wire signed [31:0] round_add = (nb > 5'd0) ? (32'sd1 << (nb - 5'd1)) : 32'sd0;

    // ── Weight fan-out staging (W1 default / W3 fallback) ────────────────────
    // w arrives already registered upstream; here we add the replicated group/lane
    // registers so the heavy 100-destination net is split. +0 pipeline latency:
    // these registers sit in parallel with taps_s1 (both feed S1 MULT next cycle).
    wire [39:0] w_lane [0:L-1];
    generate
        if (W3_REPLICATE) begin : g_w3
            // 20 copies, 1/lane (fan-out 5 = production). +800 FF.
            reg [39:0] w_rep [0:L-1];
            integer wi;
            always @(posedge clk) begin
                for (wi = 0; wi < L; wi = wi + 1)
                    w_rep[wi] <= w;
            end
            genvar gw;
            for (gw = 0; gw < L; gw = gw + 1) begin : g_w3map
                assign w_lane[gw] = w_rep[gw];
            end
        end else begin : g_w1
            // 4 group copies (lanes 0-4,5-9,10-14,15-19 → 4 pool trees). fan-out 25/copy.
            reg [39:0] w_grp0, w_grp1, w_grp2, w_grp3;
            always @(posedge clk) begin
                w_grp0 <= w; w_grp1 <= w; w_grp2 <= w; w_grp3 <= w;
            end
            genvar gl;
            for (gl = 0; gl < L; gl = gl + 1) begin : g_assign
                assign w_lane[gl] = (gl < 5)  ? w_grp0 :
                                    (gl < 10) ? w_grp1 :
                                    (gl < 15) ? w_grp2 : w_grp3;
            end
        end
    endgenerate

    // ── taps_s1: register taps to align with w_lane (1 cy) ───────────────────
    // CONTRACT: taps_in, w, a, in_ch, compute_en are all presented COMBINATIONALLY
    // at cycle n (line_buffer drives the static tap window for the given `a`; the
    // controller drives `a/w/ce` same cycle). taps_s1 and w_lane each add ONE
    // register so taps and weight reach S1 MULT the same cycle.
    reg [L*40-1:0] taps_s1;
    always @(posedge clk) taps_s1 <= taps_in;

    // ── Control delay chain: a/in_ch/compute_en delayed to match acc edge ────
    // Pipeline depth from cycle-n presentation to the acc-register-update edge = 5:
    //   n   : taps_in/w/a comb
    //   n+1 : taps_s1 <= taps_in ; w_lane <= w
    //   n+2 : prod    <= taps_s1 * w_lane           (S1 MULT)
    //   n+3 : s01/s23                               (S2)
    //   n+4 : s0123                                 (S3)
    //   n+5 : tree_out  → acc edge @ n+5 reads a_d5/ce_d5/ic_d5 (= a from cy n)
    reg [3:0] a_d1,a_d2,a_d3,a_d4,a_d5;
    reg [3:0] ic_d1,ic_d2,ic_d3,ic_d4,ic_d5;
    reg       ce_d1,ce_d2,ce_d3,ce_d4,ce_d5;
    // bias_in is per-oc, combinational from the controller's CURRENT oc. With continuous
    // pipelined issue, oc advances every in_ch cycles, so bias must be delayed 5 cy to
    // match the acc edge (acc folds bias at acc_init). nb/relu_en are layer-constant.
    reg signed [31:0] b_d1,b_d2,b_d3,b_d4,b_d5;
    always @(posedge clk) begin
        if (rst || blk_rst) begin
            a_d1<=0;a_d2<=0;a_d3<=0;a_d4<=0;a_d5<=0;
            ic_d1<=0;ic_d2<=0;ic_d3<=0;ic_d4<=0;ic_d5<=0;
            ce_d1<=0;ce_d2<=0;ce_d3<=0;ce_d4<=0;ce_d5<=0;
            b_d1<=0;b_d2<=0;b_d3<=0;b_d4<=0;b_d5<=0;
        end else begin
            a_d1<=a;     a_d2<=a_d1;  a_d3<=a_d2;  a_d4<=a_d3;  a_d5<=a_d4;
            ic_d1<=in_ch;ic_d2<=ic_d1;ic_d3<=ic_d2;ic_d4<=ic_d3;ic_d5<=ic_d4;
            ce_d1<=compute_en;ce_d2<=ce_d1;ce_d3<=ce_d2;ce_d4<=ce_d3;ce_d5<=ce_d4;
            b_d1<=bias_in;b_d2<=b_d1;b_d3<=b_d2;b_d4<=b_d3;b_d5<=b_d4;
        end
    end

    // out_valid: last input channel accumulated, at the acc edge → use d5.
    wire out_valid = ce_d5 && (a_d5 == ic_d5 - 4'd1);
    // a==0 init detector at the acc edge.
    wire acc_init  = (a_d5 == 4'd0);

    // ── Per-lane datapath: S1 MULT → TREE → ACC → S6 → S7 → S8 ───────────────
    wire signed [7:0] lane_int8 [0:L-1];
    reg               lo_valid;   // requantized-row valid (1 cy)

    genvar l;
    generate
        for (l = 0; l < L; l = l + 1) begin : lanes
            // tap/weight slices
            wire signed [7:0] t0 = taps_s1[l*40 + 0*8 +: 8];
            wire signed [7:0] t1 = taps_s1[l*40 + 1*8 +: 8];
            wire signed [7:0] t2 = taps_s1[l*40 + 2*8 +: 8];
            wire signed [7:0] t3 = taps_s1[l*40 + 3*8 +: 8];
            wire signed [7:0] t4 = taps_s1[l*40 + 4*8 +: 8];
            wire signed [7:0] w0 = w_lane[l][0*8 +: 8];
            wire signed [7:0] w1 = w_lane[l][1*8 +: 8];
            wire signed [7:0] w2 = w_lane[l][2*8 +: 8];
            wire signed [7:0] w3 = w_lane[l][3*8 +: 8];
            wire signed [7:0] w4 = w_lane[l][4*8 +: 8];

            // S1 MULT
            reg signed [15:0] p0,p1,p2,p3,p4;
            always @(posedge clk) begin
                p0 <= t0*w0; p1 <= t1*w1; p2 <= t2*w2; p3 <= t3*w3; p4 <= t4*w4;
            end
            // S2 tree-1
            reg signed [16:0] s01, s23;
            reg signed [15:0] p4_d1;
            always @(posedge clk) begin
                s01 <= p0+p1; s23 <= p2+p3; p4_d1 <= p4;
            end
            // S3 tree-2
            reg signed [17:0] s0123;
            reg signed [15:0] p4_d2;
            always @(posedge clk) begin
                s0123 <= s01+s23; p4_d2 <= p4_d1;
            end
            // S4 tree-3
            reg signed [19:0] tree_out;
            always @(posedge clk) tree_out <= s0123 + p4_d2;

            // S5 ACC (bias+round folded at a==0)
            wire signed [31:0] tree_sext = {{12{tree_out[19]}}, tree_out};
            reg signed [31:0] acc;
            always @(posedge clk) begin
                if (rst || blk_rst) acc <= 32'sd0;
                else if (ce_d5) begin
                    if (acc_init) acc <= tree_sext + b_d5 + round_add;
                    else          acc <= acc + tree_sext;
                end
            end

            // S5b acc_final (break 2-cascade adder, like production)
            reg signed [31:0] acc_final;
            always @(posedge clk) begin
                if (rst || blk_rst) acc_final <= 32'sd0;
                else if (out_valid)
                    acc_final <= acc_init ? (tree_sext + b_d5 + round_add)
                                          : (acc + tree_sext);
            end

            // S_bias passthrough (bias already folded) — keep stage for depth/timing
            reg signed [31:0] biased;
            always @(posedge clk) begin
                if (rst || blk_rst) biased <= 32'sd0;
                else                biased <= acc_final;
            end

            // S6 rescale >>>nb
            reg signed [31:0] shifted;
            always @(posedge clk) begin
                if (rst || blk_rst) shifted <= 32'sd0;
                else                shifted <= biased >>> nb;
            end

            // S7 clamp [-127,127]
            reg signed [7:0] clamped;
            always @(posedge clk) begin
                if (rst || blk_rst) clamped <= 8'sd0;
                else clamped <= (shifted > 32'sd127)  ?  8'sd127 :
                                (shifted < -32'sd127) ? -8'sd127 : shifted[7:0];
            end

            // S8 ReLU (Conv4 only)
            reg signed [7:0] relu_out;
            always @(posedge clk) begin
                if (rst || blk_rst) relu_out <= 8'sd0;
                else relu_out <= (relu_en && clamped[7]) ? 8'sd0 : clamped;
            end

            assign lane_int8[l] = relu_out;
            assign lane_out[l*8 +: 8] = relu_out;
        end
    endgenerate

    // ── lane_valid: out_valid propagated through S5b→S_bias→S6→S7→S8 = 5 cy ──
    // out_valid marks the acc edge (S5). lane_out (S8) is valid 5 cycles later:
    //   S5b(+1) S_bias(+2) S6(+3) S7(+4) S8(+5).
    reg v1,v2,v3,v4,v5;
    always @(posedge clk) begin
        if (rst || blk_rst) begin
            v1<=0;v2<=0;v3<=0;v4<=0;v5<=0;
        end else begin
            v1<=out_valid; v2<=v1; v3<=v2; v4<=v3; v5<=v4;
        end
    end
    assign lane_valid = v5;

endmodule
