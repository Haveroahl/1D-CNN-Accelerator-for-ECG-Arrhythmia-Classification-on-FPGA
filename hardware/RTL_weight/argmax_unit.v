// argmax_unit.v
// Argmax stage (split out of gap_fc_argmax).
//
// Sequential max over the 4 FC logits (INT32, signed). Walks argmax_step 0..3,
// tracking running max and its index → result[1:0] = class 0..3.
//
// fc_acc comes in flattened (fc_acc_flat[k*32+:32]).

module argmax_unit (
    input  wire        clk,
    input  wire        rst,

    // Sub-FSM control (from cnn_controller)
    input  wire [2:0]  fc_sub_state,   // GAP_S/FC_S/FC_FLUSH/ARGMAX_S/DONE_S
    input  wire [1:0]  argmax_step,    // 0..3

    // FC logits (INT32 per neuron), flattened
    input  wire [127:0] fc_acc_flat,

    // Output
    output wire [1:0]  result          // argmax class index
);

    // ── Sub-state encoding (must match cnn_controller) ─────────────────────
    localparam ARGMAX_S = 3'd4;

    // ── Unpack fc_acc_flat back into an array (body copied verbatim) ─────────
    wire signed [31:0] fc_acc [0:3];
    genvar u;
    generate
        for (u = 0; u < 4; u = u + 1) begin : fc_unpack
            assign fc_acc[u] = $signed(fc_acc_flat[u*32 +: 32]);
        end
    endgenerate

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

    assign result = argmax_idx;

endmodule
