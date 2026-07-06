// fc_unit.v
// Fully-connected stage + FC weight/bias storage (split out of gap_fc_argmax).
//
// FC weights: stored in fc_w[k][i], k=0..3 output neurons, i=0..7 inputs.
// NB=0 → no output rescale (argmax is scale-invariant). FC logits live at scale
// 2^w_shift[fc], so the FC bias is pre-scaled by 2^w_shift[fc] (fc_bias.hex,
// INT32) to be commensurate with fc_acc, and seeded into fc_acc before the MACs.
//
// 4 parallel multipliers, 2-cycle pipeline (latch gap → multiply → accumulate).
// fc_w is a register array (0-cycle read latency), addr = k*8 + i.
//
// gap_reg comes in flattened (gap_reg_flat[ch*8+:8]); fc_acc goes out flattened
// (fc_acc_flat[k*32+:32]).
//
// ROM single-load build: fc_w/fc_b are baked in via $readmemh; no runtime bus reload.

module fc_unit (
    input  wire        clk,
    input  wire        rst,

    // Sub-FSM control (from cnn_controller)
    input  wire [2:0]  fc_sub_state,   // GAP_S/FC_S/FC_FLUSH/ARGMAX_S/DONE_S
    input  wire [3:0]  fc_step,        // 0..9

    // GAP output (INT8 per channel), flattened
    input  wire [63:0] gap_reg_flat,

    // FC logits (INT32 per neuron), flattened
    output wire [127:0] fc_acc_flat
);

    // ── Sub-state encoding (must match cnn_controller) ─────────────────────
    localparam FC_S     = 3'd2,
               FC_FLUSH = 3'd3;

    // ── Unpack gap_reg_flat back into an array (body copied verbatim) ────────
    wire signed [7:0] gap_reg [0:7];
    genvar u;
    generate
        for (u = 0; u < 8; u = u + 1) begin : gap_unpack
            assign gap_reg[u] = $signed(gap_reg_flat[u*8 +: 8]);
        end
    endgenerate

    // ── FC weight storage ──────────────────────────────────────────────────
    // 4 output neurons × 8 inputs × INT8, 1D flat (Verilog-2001)
    // addr = k*8 + i,  fc_weights.hex layout [k][i] row-major
    reg signed [7:0] fc_w [0:31];
    // FC bias: 4 × INT32, scaled by 2^w_shift[fc] (logit domain). Seeded into
    // fc_acc at fc_step==0 so the MAC accumulation adds it for free.
    reg signed [31:0] fc_b [0:3];
    initial begin
        $readmemh("fc_weights.hex", fc_w);
        $readmemh("fc_bias.hex", fc_b);
    end

    // ── FC datapath ────────────────────────────────────────────────────────
    // 4 parallel multipliers, 2-cycle pipeline (latch gap → multiply → accumulate)
    // ROM addr: use fc_step directly as index into fc_w (combinational, no latency)
    // — fc_w is a register array, not a synchronous ROM → 0-cycle latency
    // Pipeline: step N latches gap[N-1], step N+1 multiplies, step N+2 accumulates

    reg signed [7:0]  fc_gap_pipe;       // gap_reg[i] latched for multiplication
    reg [2:0]         fc_w_idx;          // weight column index, tracks fc_gap_pipe
    reg signed [15:0] fc_prod [0:3];     // multiply result (registered)
    reg signed [31:0] fc_acc  [0:3];     // INT32 accumulator per output neuron
    reg               prod_valid;        // fc_prod holds valid data

    // FC pipeline (2-cycle latency: latch → multiply → accumulate):
    //   step 0:  clear acc; prod_valid=0
    //   step 1:  latch gap[0]; fc_w_idx=0
    //   step 2:  latch gap[1]; fc_w_idx=1; multiply gap[0]×w[k][0]
    //   step 3:  latch gap[2]; fc_w_idx=2; multiply gap[1]×w[k][1]; acc+=prod[0]
    //   ...
    //   step 9:  hold gap[7]; fc_w_idx=7; multiply gap[7]×w[k][7]
    //   FC_FLUSH:                                                    acc+=prod[7]

    integer k_i;
    always @(posedge clk) begin
        if (rst) begin
            prod_valid  <= 1'b0;
            fc_w_idx    <= 3'd0;
            fc_gap_pipe <= 8'sd0;
            for (k_i = 0; k_i < 4; k_i = k_i + 1) begin
                fc_prod[k_i] <= 16'sd0;
                fc_acc[k_i]  <= 32'sd0;
            end
        end else if (fc_sub_state == FC_S) begin
            case (fc_step)
                4'd0: begin
                    prod_valid <= 1'b0;
                    fc_w_idx   <= 3'd0;
                    // Seed accumulator with pre-scaled bias (logit domain).
                    for (k_i = 0; k_i < 4; k_i = k_i + 1)
                        fc_acc[k_i] <= fc_b[k_i];
                end
                default: begin
                    // Stage 1: latch gap[fc_step-1] for steps 1..8; hold at step 9
                    if (fc_step <= 4'd8) begin
                        fc_gap_pipe <= gap_reg[fc_step - 4'd1];
                        fc_w_idx    <= fc_step[2:0] - 3'd1;
                    end

                    // Stage 2: multiply fc_gap_pipe × fc_w[k][fc_w_idx]
                    // fc_w_idx = (fc_step-1) from previous cycle = column for fc_gap_pipe
                    // Index 5-bit: k_i[1:0] (2-bit) || fc_w_idx (3-bit) = k*8 + i
                    for (k_i = 0; k_i < 4; k_i = k_i + 1)
                        fc_prod[k_i] <= $signed(fc_gap_pipe)
                                      * $signed(fc_w[{k_i[1:0], fc_w_idx}]);

                    // Stage 3: accumulate valid products (from step 3 onward)
                    if (prod_valid) begin
                        for (k_i = 0; k_i < 4; k_i = k_i + 1)
                            fc_acc[k_i] <= fc_acc[k_i]
                                         + {{16{fc_prod[k_i][15]}}, fc_prod[k_i]};
                    end

                    prod_valid <= (fc_step >= 4'd2);
                end
            endcase
        end else if (fc_sub_state == FC_FLUSH) begin
            // Drain last product: gap[7]×w[k][7] computed at fc_step=9
            for (k_i = 0; k_i < 4; k_i = k_i + 1)
                fc_acc[k_i] <= fc_acc[k_i]
                             + {{16{fc_prod[k_i][15]}}, fc_prod[k_i]};
        end
    end

    // ── Flatten fc_acc for the module boundary ────────────────────────────────
    genvar f;
    generate
        for (f = 0; f < 4; f = f + 1) begin : fc_pack
            assign fc_acc_flat[f*32 +: 32] = fc_acc[f];
        end
    endgenerate

endmodule
