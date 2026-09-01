// simd_pool.v
// MaxPool K=5, stride=5 over the 20 synchronous lane outputs → 4 pooled values.
//   pooled[0] = max(lane0..4), pooled[1] = max(lane5..9),
//   pooled[2] = max(lane10..14), pooled[3] = max(lane15..19)
//
// Lanes are signed INT8. max() over 5 signed values, 0 DSP. Combinational tree of
// depth 3 (4 comparators) split into 2 PIPELINE STAGES to cut critical path
// (SIMD.md §2b/§5):
//   stage 1 (reg): m_a=max(l0,l1), m_b=max(l2,l3), pass l4
//   stage 2 (reg): pooled = max( max(m_a,m_b), l4 )
//
// out_valid follows pool_valid_in delayed 2 cycles (the 2 pipeline stages).
//
// lane_valid_mask: for blocks where (block_base + lane) >= out_len, that lane is
// out of range and must NOT influence the max (SIMD.md §7 lane-valid gate). Out-of-
// range lanes are forced to -128 (most-negative) so they never win a signed max.
// NB: for this model every out_len is divisible by 20, so masks are normally all-1;
// kept for generality + to make the gate explicit/verifiable.

module simd_pool #(
    parameter L = 20,
    parameter NP = 4   // pooled outputs = L / 5
) (
    input  wire              clk,
    input  wire              rst,

    input  wire [L*8-1:0]    lane_in,        // 20 signed INT8 lane_in[l*8+:8]
    input  wire              pool_valid_in,  // = lane_valid from lane array
    input  wire [L-1:0]      lane_valid_mask,// 1 = lane in range; 0 = force -128

    output wire [NP*8-1:0]   pooled_out,     // 4 signed INT8 pooled_out[p*8+:8]
    output wire              pooled_valid     // 1-cy strobe, 2 cy after pool_valid_in
);

    // masked lane values: out-of-range → -128 (loses every signed max)
    wire signed [7:0] lv [0:L-1];
    genvar g;
    generate
        for (g = 0; g < L; g = g + 1) begin : mask
            assign lv[g] = lane_valid_mask[g] ? $signed(lane_in[g*8 +: 8]) : -8'sd128;
        end
    endgenerate

    // signed max helper
    function signed [7:0] smax;
        input signed [7:0] x, y;
        smax = (x > y) ? x : y;
    endfunction

    // ── Stage 1: pairwise max + pass l4, per group of 5 ──────────────────────
    reg signed [7:0] s1_ab [0:NP-1];   // max(l0,l1) ... for group p
    reg signed [7:0] s1_cd [0:NP-1];   // max(l2,l3)
    reg signed [7:0] s1_e  [0:NP-1];   // l4
    reg              s1_v;

    integer p;
    always @(posedge clk) begin
        if (rst) begin
            s1_v <= 1'b0;
            for (p = 0; p < NP; p = p + 1) begin
                s1_ab[p] <= 8'sd0; s1_cd[p] <= 8'sd0; s1_e[p] <= 8'sd0;
            end
        end else begin
            s1_v <= pool_valid_in;
            for (p = 0; p < NP; p = p + 1) begin
                s1_ab[p] <= smax(lv[p*5+0], lv[p*5+1]);
                s1_cd[p] <= smax(lv[p*5+2], lv[p*5+3]);
                s1_e[p]  <= lv[p*5+4];
            end
        end
    end

    // ── Stage 2: max( max(ab,cd), e ) ────────────────────────────────────────
    reg signed [7:0] s2_pooled [0:NP-1];
    reg              s2_v;
    always @(posedge clk) begin
        if (rst) begin
            s2_v <= 1'b0;
            for (p = 0; p < NP; p = p + 1) s2_pooled[p] <= 8'sd0;
        end else begin
            s2_v <= s1_v;
            for (p = 0; p < NP; p = p + 1)
                s2_pooled[p] <= smax(smax(s1_ab[p], s1_cd[p]), s1_e[p]);
        end
    end

    genvar gp;
    generate
        for (gp = 0; gp < NP; gp = gp + 1) begin : outp
            assign pooled_out[gp*8 +: 8] = s2_pooled[gp];
        end
    endgenerate
    assign pooled_valid = s2_v;

endmodule
