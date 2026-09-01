// gap_fc_argmax_simd.v
// GAP / FC / Argmax for the SIMD variant. FC and Argmax are IDENTICAL to production
// gap_fc_argmax.v (SIMD.md §9 — "Y HỆT"); only GAP differs: it reads the wide
// pong word (4 positions/word) and sums the 4 unpacked INT8 in one read instead of
// 4 sequential byte reads.
//
// Conv4 pooled output = 8 ch × 4 positions = exactly ONE 32-bit word per channel.
// pong_dout_wide[ch*32 +: 32] = {pos3,pos2,pos1,pos0} (byte little-endian within word).
//   gap[ch] = floor( (pos0+pos1+pos2+pos3) / 4 ) = sum >> 2  (positive after Conv4 ReLU)
//
// GAP cadence (driven by simd_controller gap_step, same sub-FSM as production):
//   gap_step 0 : issue read addr 0
//   gap_step 1 : word arrives (1-cy read latency) → sum 4 bytes into gap_acc
//   gap_step 2..4: idle (kept for cadence parity with production 6-step GAP)
//   gap_step 5 : gap_reg <= gap_acc[9:2]
//
// FC: bias pre-scaled 2^w_shift[fc] seeded into fc_acc, nb=0, raw INT32 logits.
// Argmax: 4-way compare → result[1:0].

module gap_fc_argmax_simd (
    input  wire        clk,
    input  wire        rst,

    input  wire [2:0]  fc_sub_state,
    input  wire [3:0]  gap_step,
    input  wire [3:0]  fc_step,
    input  wire [1:0]  argmax_step,

    // Wide pong read data: 8 ch × 32b word (= 4 INT8 positions/ch)
    input  wire [255:0] pong_dout_wide,

    // Pong wide read address (word). For Conv4 GAP only word 0 is needed.
    output wire [6:0]  gap_rd_addr,

    output wire [1:0]  result
);
    localparam GAP_S=3'd1, FC_S=3'd2, FC_FLUSH=3'd3, ARGMAX_S=3'd4, DONE_S=3'd5;

    reg signed [7:0]  fc_w [0:31];
    reg signed [31:0] fc_b [0:3];
    initial begin
        $readmemh("fc_weights.hex", fc_w);
        $readmemh("fc_bias.hex", fc_b);
    end

    // ── GAP ──────────────────────────────────────────────────────────────────
    reg signed [9:0]  gap_acc [0:7];
    reg signed [7:0]  gap_reg [0:7];

    assign gap_rd_addr = 7'd0;   // Conv4 pool output = 1 word per channel (word 0)

    // sign-extend a byte of the wide word to 10b
    function signed [9:0] sx;
        input [7:0] b;
        sx = {{2{b[7]}}, b};
    endfunction

    integer ci;
    always @(posedge clk) begin
        if (rst) begin
            for (ci=0; ci<8; ci=ci+1) begin gap_acc[ci]<=0; gap_reg[ci]<=0; end
        end else if (fc_sub_state == GAP_S) begin
            case (gap_step)
                // step 0,1: settle bank_sel_d after the Conv4→GAP bank toggle and
                // the 1-cy pong read latency. Sum the 4 packed positions at step 2.
                4'd0: for (ci=0; ci<8; ci=ci+1) gap_acc[ci] <= 10'sd0;
                4'd2: for (ci=0; ci<8; ci=ci+1)
                          gap_acc[ci] <= sx(pong_dout_wide[ci*32 +  0 +: 8])
                                       + sx(pong_dout_wide[ci*32 +  8 +: 8])
                                       + sx(pong_dout_wide[ci*32 + 16 +: 8])
                                       + sx(pong_dout_wide[ci*32 + 24 +: 8]);
                4'd5: for (ci=0; ci<8; ci=ci+1) gap_reg[ci] <= gap_acc[ci][9:2];
                default: ;
            endcase
        end
    end

    // ── FC (identical to production) ──────────────────────────────────────────
    reg signed [7:0]  fc_gap_pipe;
    reg [2:0]         fc_w_idx;
    reg signed [15:0] fc_prod [0:3];
    reg signed [31:0] fc_acc  [0:3];
    reg               prod_valid;

    integer ki;
    always @(posedge clk) begin
        if (rst) begin
            prod_valid<=0; fc_w_idx<=0; fc_gap_pipe<=0;
            for (ki=0; ki<4; ki=ki+1) begin fc_prod[ki]<=0; fc_acc[ki]<=0; end
        end else if (fc_sub_state == FC_S) begin
            case (fc_step)
                4'd0: begin
                    prod_valid<=0; fc_w_idx<=0;
                    for (ki=0; ki<4; ki=ki+1) fc_acc[ki] <= fc_b[ki];
                end
                default: begin
                    if (fc_step <= 4'd8) begin
                        fc_gap_pipe <= gap_reg[fc_step - 4'd1];
                        fc_w_idx    <= fc_step[2:0] - 3'd1;
                    end
                    for (ki=0; ki<4; ki=ki+1)
                        fc_prod[ki] <= $signed(fc_gap_pipe)
                                     * $signed(fc_w[{ki[1:0], fc_w_idx}]);
                    if (prod_valid)
                        for (ki=0; ki<4; ki=ki+1)
                            fc_acc[ki] <= fc_acc[ki] + {{16{fc_prod[ki][15]}}, fc_prod[ki]};
                    prod_valid <= (fc_step >= 4'd2);
                end
            endcase
        end else if (fc_sub_state == FC_FLUSH) begin
            for (ki=0; ki<4; ki=ki+1)
                fc_acc[ki] <= fc_acc[ki] + {{16{fc_prod[ki][15]}}, fc_prod[ki]};
        end
    end

    // ── Argmax (identical) ────────────────────────────────────────────────────
    reg signed [31:0] argmax_max;
    reg [1:0]         argmax_idx;
    always @(posedge clk) begin
        if (rst) begin argmax_max<=0; argmax_idx<=0; end
        else if (fc_sub_state == ARGMAX_S) begin
            case (argmax_step)
                2'd0: begin argmax_max<=fc_acc[0]; argmax_idx<=2'b00; end
                2'd1: if ($signed(fc_acc[1])>$signed(argmax_max)) begin argmax_max<=fc_acc[1]; argmax_idx<=2'b01; end
                2'd2: if ($signed(fc_acc[2])>$signed(argmax_max)) begin argmax_max<=fc_acc[2]; argmax_idx<=2'b10; end
                2'd3: if ($signed(fc_acc[3])>$signed(argmax_max)) begin argmax_max<=fc_acc[3]; argmax_idx<=2'b11; end
            endcase
        end
    end

    assign result = argmax_idx;
endmodule
