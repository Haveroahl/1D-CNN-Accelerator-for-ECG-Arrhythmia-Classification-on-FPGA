// cp_weight_store.v
// Conv weight + bias storage for the 8-PE cp_engine, factored out as its own
// module. Two build-time variants share IDENTICAL ports and IDENTICAL pipeline
// timing (w_packed valid at cycle N+1), so both are bit-exact against the golden
// checkpoints. Select with `+define+WEIGHT_ROM` at vlog/Quartus time:
//
//   V2 (default, Phase B01) — runtime-reloadable weight RAM
//     8 per-oc M10K RAMs (40b × 32), SYNC read (q register = the w_packed stage),
//     bus write port (w_wr_*, b_wr_*), topology base from cfg_base. This is the
//     production behavior used by tb_top/tb_topo.
//
//   V1 (`+define+WEIGHT_ROM`) — hard-loaded ROM (pre-B01 baseline)
//     4 per-layer FF-array ROMs (async combinational MUX) + a w_packed FF stage.
//     No bus write path, fixed Chapman topology (layer_state selects the ROM).
//     Used only for the V1-vs-V2 resource/Fmax comparison (paper RQ4/E1). The
//     bus/cfg ports still exist but are unused → synthesis trims them.
//
// Both variants read weights hex at elaboration ($readmemh) unless the datapath
// is overridden at runtime (V2 bus). Compile with +define+NO_WEIGHT_INIT to skip
// the ROM init (V2 only — proves the bus load path stands alone).
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

    // Topology base per layer (V2 only): 4 × 5-bit weight-RAM word bases
    input  wire [19:0] cfg_base,

    // ── Bus write port (V2 Phase B01 runtime reload; ignored in ROM mode) ───
    input  wire        w_wr_en,
    input  wire [2:0]  w_wr_oc,
    input  wire [4:0]  w_wr_word,
    input  wire [39:0] w_wr_data,
    input  wire        b_wr_en,
    input  wire [4:0]  b_wr_addr,
    input  wire [31:0] b_wr_data,

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

`ifdef WEIGHT_ROM
    // ═══════════════════════ V1 — hard ROM (FF arrays) ═══════════════════════
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

`else
    // ═══════════════════ V2 — Phase B01 weight RAM (M10K) ════════════════════
    // 8 per-oc M10K RAMs, 40-bit × 32 words (word = layer_base + ic). SYNC read:
    // addr issued cy N → q valid cy N+1 (that q register IS the w_packed stage).
    (* ramstyle = "M10K" *) reg [39:0] w_ram0 [0:31];
    (* ramstyle = "M10K" *) reg [39:0] w_ram1 [0:31];
    (* ramstyle = "M10K" *) reg [39:0] w_ram2 [0:31];
    (* ramstyle = "M10K" *) reg [39:0] w_ram3 [0:31];
    (* ramstyle = "M10K" *) reg [39:0] w_ram4 [0:31];
    (* ramstyle = "M10K" *) reg [39:0] w_ram5 [0:31];
    (* ramstyle = "M10K" *) reg [39:0] w_ram6 [0:31];
    (* ramstyle = "M10K" *) reg [39:0] w_ram7 [0:31];

`ifndef NO_WEIGHT_INIT
    initial begin
        $readmemh("w_ram0.hex", w_ram0);
        $readmemh("w_ram1.hex", w_ram1);
        $readmemh("w_ram2.hex", w_ram2);
        $readmemh("w_ram3.hex", w_ram3);
        $readmemh("w_ram4.hex", w_ram4);
        $readmemh("w_ram5.hex", w_ram5);
        $readmemh("w_ram6.hex", w_ram6);
        $readmemh("w_ram7.hex", w_ram7);
        $readmemh("conv_bias.hex", b_store);
    end
`endif

    // Conv weight write (bus).
    always @(posedge clk) begin
        if (w_wr_en) begin
            case (w_wr_oc)
                3'd0: w_ram0[w_wr_word] <= w_wr_data;
                3'd1: w_ram1[w_wr_word] <= w_wr_data;
                3'd2: w_ram2[w_wr_word] <= w_wr_data;
                3'd3: w_ram3[w_wr_word] <= w_wr_data;
                3'd4: w_ram4[w_wr_word] <= w_wr_data;
                3'd5: w_ram5[w_wr_word] <= w_wr_data;
                3'd6: w_ram6[w_wr_word] <= w_wr_data;
                3'd7: w_ram7[w_wr_word] <= w_wr_data;
            endcase
        end
    end

    // Conv weight read: word = layer_base + a, issued cycle N (async addr).
    wire [4:0] w_layer_base = cfg_base[layer_idx*5 +: 5];
    wire [4:0] w_rd_word    = w_layer_base + {1'b0, a[3:0]};

    // M10K synchronous read: q registered 1 cycle (this IS the w_packed stage).
    always @(posedge clk) begin
        w_packed[0] <= w_ram0[w_rd_word];
        w_packed[1] <= w_ram1[w_rd_word];
        w_packed[2] <= w_ram2[w_rd_word];
        w_packed[3] <= w_ram3[w_rd_word];
        w_packed[4] <= w_ram4[w_rd_word];
        w_packed[5] <= w_ram5[w_rd_word];
        w_packed[6] <= w_ram6[w_rd_word];
        w_packed[7] <= w_ram7[w_rd_word];
    end
`endif

    // ── Bias write (bus). V2 only — in ROM mode b_store is init-only. ───────
`ifndef WEIGHT_ROM
    always @(posedge clk) begin
        if (b_wr_en)
            b_store[b_wr_addr] <= b_wr_data;
    end
`endif

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
