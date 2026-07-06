// cp_weight_store.v  (ROM-only variant — thesis single-load build)
// Conv weight + bias storage for the 8-PE cp_engine.
//
// Hard-loaded ROM: 4 per-layer FF-array ROMs (async combinational MUX) + a
// w_packed FF stage. Weights are baked in at elaboration via $readmemh; there is
// no runtime bus reload and no topology reconfiguration (fixed Chapman topology,
// layer_state selects the ROM). w_packed valid at cycle N+1.
//
// Outputs are flattened (Verilog-2001 forbids array ports):
//   w_packed_flat[oc*40 +: 40] — 40-bit packed 5-tap weight for output channel oc
//   b_cur_flat  [oc*32 +: 32] — INT32 bias for output channel oc (current layer)

module cp_weight_store (
    input  wire        clk,
    input  wire        rst,

    // Datapath selectors (from controller/cp_engine)
    input  wire [2:0]  layer_state,  // CONV1=2 .. CONV4=5
    input  wire [3:0]  a,            // input-channel counter (weight word index)

    // Outputs (registered at cycle N+1)
    output wire [319:0] w_packed_flat,  // 8 × 40b
    output wire [255:0] b_cur_flat      // 8 × 32b
);

    // ── Layer state encoding (must match cnn_controller) ──────────────────
    localparam CONV1 = 3'd2;
    localparam CONV2 = 3'd3;
    localparam CONV3 = 3'd4;
    localparam CONV4 = 3'd5;

    // CONV1=2→0, CONV2=3→1, CONV3=4→2, CONV4=5→3
    wire [1:0] layer_idx = layer_state[1:0] - 2'd2;

    // ── Bias: INT32, 8 oc × 4 layer = 32 entries, addr = oc*4 + layer_idx ───
    (* ramstyle = "MLAB" *) reg signed [31:0] b_store [0:31];

    reg [39:0] w_packed [0:7];

    // ═══════════════════════ Hard ROM (FF arrays) ═══════════════════════════
    // 4 per-layer FF arrays, each entry = 40-bit packed 5 taps.
    //   w_rom_conv1[0:3]   — 4 oc × 1 ic
    //   w_rom_conv2[0:15]  — 4 oc × 4 ic   addr = oc*4 + ic
    //   w_rom_conv3[0:31]  — 8 oc × 4 ic   addr = oc*4 + ic
    //   w_rom_conv4[0:63]  — 8 oc × 8 ic   addr = oc*8 + ic
    reg [39:0] w_rom_conv1 [0:3];
    reg [39:0] w_rom_conv2 [0:15];
    reg [39:0] w_rom_conv3 [0:31];
    reg [39:0] w_rom_conv4 [0:63];

    initial begin
        $readmemh("conv1_w.hex", w_rom_conv1);
        $readmemh("conv2_w.hex", w_rom_conv2);
        $readmemh("conv3_w.hex", w_rom_conv3);
        $readmemh("conv4_w.hex", w_rom_conv4);
        $readmemh("conv_bias.hex", b_store);
    end

    // Weight combinational MUX: w_rom[oc, a] → w_comb (async).
    // Per-oc: 4:1 layer MUX + 8:1 ic MUX (a[2:0]) — combinational, ~2 LUT levels.
    wire [39:0] w_comb [0:7];

    assign w_comb[0] = (layer_state == CONV1) ? w_rom_conv1[2'd0]           :
                       (layer_state == CONV2) ? w_rom_conv2[{2'd0, a[1:0]}] :
                       (layer_state == CONV3) ? w_rom_conv3[{3'd0, a[1:0]}] :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd0, a[2:0]}] : 40'd0;
    assign w_comb[1] = (layer_state == CONV1) ? w_rom_conv1[2'd1]           :
                       (layer_state == CONV2) ? w_rom_conv2[{2'd1, a[1:0]}] :
                       (layer_state == CONV3) ? w_rom_conv3[{3'd1, a[1:0]}] :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd1, a[2:0]}] : 40'd0;
    assign w_comb[2] = (layer_state == CONV1) ? w_rom_conv1[2'd2]           :
                       (layer_state == CONV2) ? w_rom_conv2[{2'd2, a[1:0]}] :
                       (layer_state == CONV3) ? w_rom_conv3[{3'd2, a[1:0]}] :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd2, a[2:0]}] : 40'd0;
    assign w_comb[3] = (layer_state == CONV1) ? w_rom_conv1[2'd3]           :
                       (layer_state == CONV2) ? w_rom_conv2[{2'd3, a[1:0]}] :
                       (layer_state == CONV3) ? w_rom_conv3[{3'd3, a[1:0]}] :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd3, a[2:0]}] : 40'd0;
    assign w_comb[4] = (layer_state == CONV3) ? w_rom_conv3[{3'd4, a[1:0]}] :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd4, a[2:0]}] : 40'd0;
    assign w_comb[5] = (layer_state == CONV3) ? w_rom_conv3[{3'd5, a[1:0]}] :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd5, a[2:0]}] : 40'd0;
    assign w_comb[6] = (layer_state == CONV3) ? w_rom_conv3[{3'd6, a[1:0]}] :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd6, a[2:0]}] : 40'd0;
    assign w_comb[7] = (layer_state == CONV3) ? w_rom_conv3[{3'd7, a[1:0]}] :
                       (layer_state == CONV4) ? w_rom_conv4[{3'd7, a[2:0]}] : 40'd0;

    // w_packed: register w_comb (1 FF stage, aligns with mux_s1).
    always @(posedge clk) begin
        w_packed[0] <= w_comb[0];
        w_packed[1] <= w_comb[1];
        w_packed[2] <= w_comb[2];
        w_packed[3] <= w_comb[3];
        w_packed[4] <= w_comb[4];
        w_packed[5] <= w_comb[5];
        w_packed[6] <= w_comb[6];
        w_packed[7] <= w_comb[7];
    end

    // ── Bias: registered per layer (only changes on layer transition) ───────
    reg signed [31:0] b_cur [0:7];
    integer bi;
    always @(posedge clk) begin
        for (bi = 0; bi < 8; bi = bi + 1)
            b_cur[bi] <= b_store[{bi[2:0], layer_idx}];  // oc[2:0]*4 + layer_idx[1:0]
    end

    // ── Flatten outputs ─────────────────────────────────────────────────────
    genvar oc;
    generate
        for (oc = 0; oc < 8; oc = oc + 1) begin : flat
            assign w_packed_flat[oc*40 +: 40] = w_packed[oc];
            assign b_cur_flat  [oc*32 +: 32]  = b_cur[oc];
        end
    endgenerate

endmodule
