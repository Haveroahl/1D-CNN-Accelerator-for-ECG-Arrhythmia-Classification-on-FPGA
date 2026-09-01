// ecg_core_simd.v
// Bus-agnostic SIMD-20 accelerator core. Wires the SIMD datapath:
//   input_buffer (Conv1 stream) + pong_sram_wide (inter-layer) + line_buffer_engine
//   + simd_lane_array + simd_pool + gap_fc_argmax_simd + simd_controller.
//
// Interface = same 8 wires as production ecg_core (sram_wr_addr/din/we, start,
// busy/done/result, isram_free) so the bus adapter (avalon_slave) is reused as-is.
//
// SOURCE-DATA ALIGNMENT (the classic SIMD trap, SIMD.md §7 risk):
//   The controller drives pos_load + shift_en in cycle n. The source memory
//   (input_buffer or pong_sram_wide) has 1-cycle read latency, so the streamed
//   sample arrives in cycle n+1. We therefore REGISTER shift_en and pad_zero by one
//   cycle (shift_en_d/pad_zero_d) so they reach the line-buffer the same cycle the
//   data does. This mirrors production's pad_zero_r alignment.
//
//   Logical position = pos_load - 2 (controller streams pos_load = block_base+cnt;
//   the -2 left-pad offset is applied here when forming the source address). When
//   logical position < 0 or >= in_len, the controller already raises pad_zero, so
//   the source address is don't-care (forced 0).

module ecg_core_simd (
    input  wire        clk,
    input  wire        rst,

    input  wire [11:0] sram_wr_addr,
    input  wire [7:0]  sram_din,
    input  wire        sram_we,

    input  wire        start,
    output wire        busy,
    output wire        done,
    output wire [1:0]  result,
    output wire        isram_free
);
    localparam L = 20;

    // ── Controller signals ─────────────────────────────────────────────────
    wire        c_srw_rst, c_blk_rst, c_shift_en, c_pad_zero, c_compute_en;
    wire        c_wide_load;            // wide 4-pos load strobe
    wire [9:0]  c_wprime_word;          // word address for wide read (0..624 Conv1)
    wire [3:0]  c_wpad4;                // per-position right-pad mask
    wire [11:0] c_pos_load;
    wire [3:0]  c_a, c_in_ch, c_oc;
    wire [11:0] c_in_len, c_block_base;
    wire [4:0]  c_nb;
    wire        c_relu_en, c_src_is_conv1, c_bank_sel;
    wire [6:0]  c_pong_waddr;
    wire [7:0]  c_pong_we;
    wire [2:0]  c_fc_sub, c_layer_state;
    wire [3:0]  c_gap_step, c_fc_step;
    wire [1:0]  c_argmax_step, c_argmax_result;

    // ── Datapath nets ────────────────────────────────────────────────────────
    wire [255:0] pong_dout;            // 8 ch × 32b word (Ping read)
    wire [31:0]  ibuf_word;            // input_buffer 32b word (4 pos, Conv1)
    wire [L*40-1:0] taps_out;
    wire [39:0]  w_packed;             // weight for current (oc, a)
    wire signed [31:0] bias_cur;
    wire [L*8-1:0] lane_out;
    wire           lane_valid;
    wire [L*8-1:0] pooled_lanes_unused;
    wire [4*8-1:0] pooled_out;
    wire           pooled_valid;
    wire [6:0]     gap_rd_addr;

    // ── Source address + 1-cy alignment ──────────────────────────────────────
    // SINGLE-SHIFT path (boundary fixups): logical position = pos_load - 2.
    wire [11:0] log_pos    = c_pos_load - 12'd2;
    wire [9:0]  prime_word = log_pos[11:2];  // word containing the position (0..624)
    wire [1:0]  prime_byte = log_pos[1:0];   // byte within word
    // WIDE path: word address straight from controller.
    // input_buffer word address (Conv1, 0..624) and pong word address (Conv2-4, 0..124)
    // share the read index: wide load uses c_wprime_word; single-shift uses prime_word.
    wire [9:0]  ibuf_rd_word = c_wide_load ? c_wprime_word : prime_word;
    // pong read address MUX (7-bit): GAP; else wide/single word (low 7 bits suffice).
    wire [6:0]  pong_rd_addr = (c_layer_state == 3'd6) ? gap_rd_addr :
                               c_wide_load             ? c_wprime_word[6:0] : prime_word[6:0];

    // delay control by 1 cy to align with 1-cy source read latency
    reg         shift_d, pad_d, wide_d, src_c1_d;
    reg  [1:0]  byte_d;
    reg  [3:0]  wpad4_d;
    always @(posedge clk) begin
        if (rst) begin shift_d<=0; pad_d<=0; byte_d<=0; src_c1_d<=0; wide_d<=0; wpad4_d<=0; end
        else begin
            shift_d  <= c_shift_en;
            pad_d    <= c_pad_zero;
            byte_d   <= prime_byte;
            src_c1_d <= c_src_is_conv1;
            wide_d   <= c_wide_load;
            wpad4_d  <= c_wpad4;
        end
    end

    // ── Build line-buffer inputs (after 1-cy read latency) ───────────────────
    // WIDE (4-pos word): Conv1 = input_buffer word on ch0 (others 0); Conv2-4 = pong
    //   word per channel. SINGLE (1-pos byte): Conv1 = input_buffer byte at byte_d on
    //   ch0; Conv2-4 = pong byte at byte_d per channel.
    wire [255:0] lbuf_din4;            // 8ch × 32b wide word
    wire [63:0]  lbuf_din;             // 8ch × 8b single byte
    genvar ch;
    generate
        for (ch = 0; ch < 8; ch = ch + 1) begin : din_sel
            // wide word for this channel
            assign lbuf_din4[ch*32 +: 32] =
                src_c1_d ? ((ch==0) ? ibuf_word : 32'h0) : pong_dout[ch*32 +: 32];
            // single byte for this channel
            wire [7:0] src_word_byte =
                (byte_d == 2'd0) ? lbuf_din4[ch*32 +  0 +: 8] :
                (byte_d == 2'd1) ? lbuf_din4[ch*32 +  8 +: 8] :
                (byte_d == 2'd2) ? lbuf_din4[ch*32 + 16 +: 8] :
                                   lbuf_din4[ch*32 + 24 +: 8];
            assign lbuf_din[ch*8 +: 8] = src_word_byte;
        end
    endgenerate

    // ── input_buffer (Conv1 wide source) ──────────────────────────────────────
    input_buffer u_ibuf (
        .clk(clk),
        .wr_addr(sram_wr_addr), .din(sram_din), .we(sram_we),
        .rd_word(ibuf_rd_word), .word_out(ibuf_word)
    );

    // ── line_buffer_engine (1-pos shift + 4-pos wide load) ────────────────────
    line_buffer_engine #(.L(L), .DEPTH(24)) u_lbuf (
        .clk(clk), .rst(rst), .srw_rst(c_srw_rst),
        .din_ch(lbuf_din), .shift_en(shift_d), .pad_zero(pad_d),
        .wide_load(wide_d), .din4_ch(lbuf_din4), .pad4_zero(wpad4_d),
        .a(c_a), .taps_out(taps_out)
    );

    // ── Weight + bias ROM (same layout as production) ─────────────────────────
    simd_weight_rom u_wrom (
        .clk(clk),
        .layer_state(c_layer_state), .oc(c_oc), .a(c_a),
        .w_packed(w_packed), .bias_cur(bias_cur)
    );

    // ── lane array ────────────────────────────────────────────────────────────
    simd_lane_array #(.L(L)) u_lane (
        .clk(clk), .rst(rst), .blk_rst(c_blk_rst),
        .taps_in(taps_out), .w(w_packed), .bias_in(bias_cur),
        .a(c_a), .in_ch(c_in_ch), .compute_en(c_compute_en),
        .nb(c_nb), .relu_en(c_relu_en),
        .lane_out(lane_out), .lane_valid(lane_valid)
    );

    // ── pool ────────────────────────────────────────────────────────────────
    simd_pool #(.L(L), .NP(4)) u_pool (
        .clk(clk), .rst(rst), .lane_in(lane_out),
        .pool_valid_in(lane_valid), .lane_valid_mask({L{1'b1}}),
        .pooled_out(pooled_out), .pooled_valid(pooled_valid)
    );

    // ── Pong-wide write: pack 4 pooled INT8 → word, route to current oc ───────
    // pooled_out[p*8+:8] = pooled position p of the current block, oc fixed.
    // The controller registers pong_we (asserted 1 cy AFTER pooled_valid), so we
    // must LATCH the pooled word on pooled_valid and present it the next cycle to
    // keep data + write-enable aligned (otherwise pooled_out has already changed).
    wire [31:0] pong_word_c = { pooled_out[3*8+:8], pooled_out[2*8+:8],
                                pooled_out[1*8+:8], pooled_out[0*8+:8] };
    reg  [31:0] pong_word_r;
    always @(posedge clk) if (pooled_valid) pong_word_r <= pong_word_c;
    wire [255:0] pong_din;
    generate
        for (ch = 0; ch < 8; ch = ch + 1) begin : pong_din_g
            assign pong_din[ch*32 +: 32] = pong_word_r;  // latched; we[oc] gates channel
        end
    endgenerate

    pong_sram_wide #(.WORDS(128)) u_pong (
        .clk(clk), .bank_sel(c_bank_sel),
        .wr_addr(c_pong_waddr), .din(pong_din), .we(c_pong_we), .wmask(4'hF),
        .rd_addr(pong_rd_addr), .dout(pong_dout)
    );

    // ── GAP / FC / Argmax ─────────────────────────────────────────────────────
    gap_fc_argmax_simd u_gfa (
        .clk(clk), .rst(rst),
        .fc_sub_state(c_fc_sub), .gap_step(c_gap_step),
        .fc_step(c_fc_step), .argmax_step(c_argmax_step),
        .pong_dout_wide(pong_dout), .gap_rd_addr(gap_rd_addr),
        .result(c_argmax_result)
    );

    // ── Controller ────────────────────────────────────────────────────────────
    simd_controller #(.L(L)) u_ctrl (
        .clk(clk), .rst(rst), .start(start), .pooled_valid(pooled_valid),
        .srw_rst(c_srw_rst), .blk_rst(c_blk_rst), .shift_en(c_shift_en),
        .pad_zero(c_pad_zero), .pos_load(c_pos_load),
        .wide_load(c_wide_load), .wprime_word(c_wprime_word), .wpad4(c_wpad4),
        .a(c_a), .in_ch(c_in_ch),
        .in_len(c_in_len), .compute_en(c_compute_en), .nb(c_nb), .relu_en(c_relu_en),
        .oc(c_oc), .block_base(c_block_base),
        .bank_sel(c_bank_sel), .pong_waddr(c_pong_waddr), .pong_we(c_pong_we),
        .src_is_conv1(c_src_is_conv1),
        .fc_sub_state(c_fc_sub), .gap_step(c_gap_step), .fc_step(c_fc_step),
        .argmax_step(c_argmax_step), .argmax_result(c_argmax_result),
        .layer_state(c_layer_state), .busy(busy), .done(done),
        .result(result), .isram_free(isram_free)
    );

endmodule
