// gap_unit.v
// Global Average Pooling stage (split out of gap_fc_argmax).
//
// Reads Conv4 output from the Ping bank (8 channels × 4 entries), accumulates
// over the 4 spatial positions, then GAP = floor(sum/4) = sum[9:2] (Conv4 has
// ReLU so values are non-negative → no clamp). Inactive Conv4 channels
// (out_ch_mask[ch]=0) are forced to 0 so a reduced out_ch is bit-exact.
//
// Output gap_reg is flattened (Verilog-2001 forbids array ports):
//   gap_reg_flat[ch*8 +: 8] — INT8 GAP output for channel ch.

module gap_unit (
    input  wire        clk,
    input  wire        rst,

    // Sub-FSM control (from cnn_controller)
    input  wire [2:0]  fc_sub_state,   // GAP_S/FC_S/FC_FLUSH/ARGMAX_S/DONE_S
    input  wire [3:0]  gap_step,       // 0..5

    // Ping SRAM data (Conv4 output, 8 channels × 4 entries)
    input  wire [63:0] ping_dout,      // packed: ping_dout[ch*8+:8], 1-cy latency

    // Conv4 active-output mask (= cp_en of Conv4). Default 8'hFF = all active.
    input  wire [7:0]  out_ch_mask,

    // Ping SRAM read address (to top-level → ping_pong_sram rd_addr)
    output reg  [8:0]  gap_rd_addr,    // broadcast to all 8 channels (0..3)

    // GAP output (INT8 per channel), flattened
    output wire [63:0] gap_reg_flat
);

    // ── Sub-state encoding (must match cnn_controller) ─────────────────────
    localparam GAP_S = 3'd1;

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
                    // Conv4 RELU_EN=1 → gap_acc ∈ [0,508] → gap_acc[9:2] ∈ [0,127], no clamp.
                    // Inactive Conv4 channels (out_ch_mask[ch]=0) hold stale ping
                    // data → force 0 so reduced out_ch is bit-exact.
                    for (ch_i = 0; ch_i < 8; ch_i = ch_i + 1)
                        gap_reg[ch_i] <= out_ch_mask[ch_i] ? gap_acc[ch_i][9:2] : 8'sd0;
                end
                default: ;
            endcase
        end
    end

    // ── Flatten gap_reg for the module boundary ──────────────────────────────
    genvar g;
    generate
        for (g = 0; g < 8; g = g + 1) begin : gap_pack
            assign gap_reg_flat[g*8 +: 8] = gap_reg[g];
        end
    endgenerate

endmodule
