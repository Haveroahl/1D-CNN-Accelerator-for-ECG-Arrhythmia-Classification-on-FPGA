// gap_fc_argmax.v
// Sequential engine: GAP (6cy) → FC (10cy + 1 flush) → Argmax (4cy) → Done (1cy)
// Total: 22 cycles after entry from Conv4 layer_done
//
// Reads Conv4 output from Ping bank (after bank_sel swap in controller).
// ping_dout[0..7] are driven by ping_pong_sram with pong_rd_addr as read address.
//
// FC weights: stored in fc_w[k][i], k=0..3 output neurons, i=0..7 inputs
// NB=0 → no output rescale (argmax is scale-invariant). FC logits live at scale
// 2^w_shift[fc], so the FC bias is pre-scaled by 2^w_shift[fc] (fc_bias.hex,
// INT32) to be commensurate with fc_acc, and seeded into fc_acc before the MACs.

module gap_fc_argmax (
    input  wire        clk,
    input  wire        rst,

    // Sub-FSM control (driven by cnn_controller)
    input  wire [2:0]  fc_sub_state,   // GAP_S/FC_S/FC_FLUSH/ARGMAX_S/DONE_S
    input  wire [3:0]  gap_step,       // 0..5
    input  wire [3:0]  fc_step,        // 0..9
    input  wire [1:0]  argmax_step,    // 0..3

    // Ping SRAM data (Conv4 output, 8 channels × 4 entries)
    input  wire [63:0] ping_dout,               // packed: ping_dout[ch*8+:8], 1-cy latency

    // Ping SRAM read address (to top-level → ping_pong_sram rd_addr)
    output reg  [8:0]  gap_rd_addr,    // broadcast to all 8 channels (0..3)

    // FC weight ROM init (loaded at elaboration)
    // fc_w[k][i]: output neuron k=0..3, input i=0..7
    // Loaded from fc_weights.hex — layout row-major [k][i]

    // Outputs
    output wire [1:0]  result          // argmax class index
);

    // ── Sub-state encoding (must match cnn_controller) ─────────────────────
    localparam GAP_S    = 3'd1,
               FC_S     = 3'd2,
               FC_FLUSH = 3'd3,
               ARGMAX_S = 3'd4,
               DONE_S   = 3'd5;

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

    // ── GAP datapath ───────────────────────────────────────────────────────
    reg signed [9:0]  gap_acc [0:7];   // 10-bit: max 4×127=508
    reg signed [7:0]  gap_reg [0:7];   // INT8 GAP output

    // gap_rd_addr: combinational from gap_step → ping_pong 1cy read latency
    // → data arrives next cycle (step N+1) when we accumulate.
    always @* begin
        case (gap_step)
            4'd0: gap_rd_addr = 9'd0;
            4'd1: gap_rd_addr = 9'd1;
            4'd2: gap_rd_addr = 9'd2;
            4'd3: gap_rd_addr = 9'd3;
            default: gap_rd_addr = 9'd0;
        endcase
    end

    integer ch_i;
    always @(posedge clk) begin
        if (rst) begin
            for (ch_i = 0; ch_i < 8; ch_i = ch_i + 1) begin
                gap_acc[ch_i] <= 10'sd0;
                gap_reg[ch_i] <= 8'sd0;
            end
        end else if (fc_sub_state == GAP_S) begin
            case (gap_step)
                4'd0: begin
                    for (ch_i = 0; ch_i < 8; ch_i = ch_i + 1)
                        gap_acc[ch_i] <= 10'sd0;
                end
                4'd1: begin
                    for (ch_i = 0; ch_i < 8; ch_i = ch_i + 1)
                        gap_acc[ch_i] <= gap_acc[ch_i]
                                       + {{2{ping_dout[ch_i*8+7]}}, ping_dout[ch_i*8 +: 8]};
                end
                4'd2: begin
                    for (ch_i = 0; ch_i < 8; ch_i = ch_i + 1)
                        gap_acc[ch_i] <= gap_acc[ch_i]
                                       + {{2{ping_dout[ch_i*8+7]}}, ping_dout[ch_i*8 +: 8]};
                end
                4'd3: begin
                    for (ch_i = 0; ch_i < 8; ch_i = ch_i + 1)
                        gap_acc[ch_i] <= gap_acc[ch_i]
                                       + {{2{ping_dout[ch_i*8+7]}}, ping_dout[ch_i*8 +: 8]};
                end
                4'd4: begin
                    // last sample arrives (no new addr issue)
                    for (ch_i = 0; ch_i < 8; ch_i = ch_i + 1)
                        gap_acc[ch_i] <= gap_acc[ch_i]
                                       + {{2{ping_dout[ch_i*8+7]}}, ping_dout[ch_i*8 +: 8]};
                end
                4'd5: begin
                    // Conv4 RELU_EN=1 → gap_acc ∈ [0,508] → gap_acc[9:2] ∈ [0,127], no clamp
                    for (ch_i = 0; ch_i < 8; ch_i = ch_i + 1)
                        gap_reg[ch_i] <= gap_acc[ch_i][9:2];
                end
                default: ;
            endcase
        end
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

    // ── Argmax datapath ────────────────────────────────────────────────────
    reg signed [31:0] argmax_max;
    reg [1:0]         argmax_idx;

    always @(posedge clk) begin
        if (rst) begin
            argmax_max <= 32'sd0;
            argmax_idx <= 2'b00;
        end else if (fc_sub_state == ARGMAX_S) begin
            case (argmax_step)
                2'd0: begin
                    argmax_max <= fc_acc[0];
                    argmax_idx <= 2'b00;
                end
                2'd1: if ($signed(fc_acc[1]) > $signed(argmax_max)) begin
                    argmax_max <= fc_acc[1];
                    argmax_idx <= 2'b01;
                end
                2'd2: if ($signed(fc_acc[2]) > $signed(argmax_max)) begin
                    argmax_max <= fc_acc[2];
                    argmax_idx <= 2'b10;
                end
                2'd3: if ($signed(fc_acc[3]) > $signed(argmax_max)) begin
                    argmax_max <= fc_acc[3];
                    argmax_idx <= 2'b11;
                end
            endcase
        end
    end

    // ── Outputs ────────────────────────────────────────────────────────────
    // Note: cnn_controller tracks completion via fc_sub_state==DONE_SUB directly,
    // so no separate `done` output needed here.
    assign result = argmax_idx;

endmodule
