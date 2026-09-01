// simd_controller.v
// FSM for the SIMD-20 position-parallel accelerator (SIMD.md §8).
//
// Loop nesting (block-outer, oc-inner — line-buffer primed/slid ONCE per block and
// reused across all OUT_CH×IN_CH → 0 redundant reload):
//   for layer in {Conv1..Conv4}:
//     PRIME first block (stream L+4=24 positions; left-pad first 2)
//     for block in 0..num_blocks-1:        (block_base += L)
//       COMPUTE: hold buffer; for oc { reset acc; for a drive lane; wait pooled; write }
//       SLIDE  : stream L=20 new positions for the next block (right-pad past in_len)
//
// Streaming positions ascending. pos_load = SOURCE position to fetch (the core maps
// it to input_buffer addr for Conv1 or pong-wide word for Conv2-4). pad_zero forces 0
// for logical position <0 (left, PP4) or >=in_len (right, PP1).
//
// All out_len divisible by L=20 and by 4 → no partial blocks/words.

module simd_controller #(
    parameter L = 20
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,

    input  wire        pooled_valid,   // one strobe per oc result from datapath

    // Line-buffer / stream control
    output reg         srw_rst,
    output reg         blk_rst,        // per-oc clear of lane-array acc/valid pipe
    output reg         shift_en,
    output reg         pad_zero,
    output reg  [11:0] pos_load,       // logical source position to fetch
    // Wide-load control (4-pos/cy): for PRIME/SLIDE, fetch a 4-pos word (input_buffer
    // for Conv1, pong-wide for Conv2-4). Cuts slide 20cy→~7cy/block.
    output reg         wide_load,      // 1 = wide 4-pos load this cycle
    output reg  [9:0]  wprime_word,    // word address to read (0..624 for Conv1)
    output reg  [3:0]  wpad4,          // per-position right-pad mask for the wide word
    output reg  [3:0]  a,
    output reg  [3:0]  in_ch,
    output reg  [11:0] in_len,
    output reg         compute_en,
    output reg  [4:0]  nb,
    output reg         relu_en,
    output reg  [3:0]  oc,
    output reg  [11:0] block_base,

    // Pong-wide write
    output reg         bank_sel,
    output reg  [6:0]  pong_waddr,
    output reg  [7:0]  pong_we,

    // Source select
    output reg         src_is_conv1,

    // GAP/FC/Argmax sub-FSM
    output reg  [2:0]  fc_sub_state,
    output reg  [3:0]  gap_step,
    output reg  [3:0]  fc_step,
    output reg  [1:0]  argmax_step,
    input  wire [1:0]  argmax_result,

    // Status
    output reg  [2:0]  layer_state,
    output reg         busy,
    output reg         done,
    output reg  [1:0]  result,
    output reg         isram_free
);
    localparam IDLE=3'd0, LOAD=3'd1, CONV1=3'd2, CONV2=3'd3, CONV3=3'd4,
               CONV4=3'd5, GAP_FC=3'd6, DONE_S=3'd7;
    localparam GAP_SUB=3'd1, FC_SUB=3'd2, FC_FLUSH=3'd3, ARGMAX_SUB=3'd4, DONE_SUB=3'd5;
    localparam NB1=5'd8, NB2=5'd6, NB3=5'd6, NB4=5'd7;

    // per-block phases
    localparam PH_PRIME=4'd0, PH_OC_RST=4'd1, PH_SWEEP=4'd2, PH_DRAIN=4'd3,
               PH_SLIDE=4'd4, PH_TRANS=4'd5, PH_PRIME_SETTLE=4'd6, PH_SLIDE_SETTLE=4'd7,
               PH_WPRIME=4'd8, PH_WSLIDE=4'd9;
    reg [3:0]  phase;
    reg [2:0]  settle_cnt;
    reg        sweep_armed;

    reg [3:0]  out_ch_n;
    reg [6:0]  num_blocks;
    reg [6:0]  block_idx;
    reg [4:0]  prime_cnt;
    reg [4:0]  slide_cnt;
    reg [3:0]  wstep;   // step through the wide (4-pos word + 2 single fixup) sequence

    // ── Pipelined issue↔writeback (decouple — bỏ PH_DRAIN stall) ─────────────
    // PH_SWEEP issues (oc,a) continuously (pipeline stays full). Results emerge ~12cy
    // later as pooled_valid strobes IN ISSUE ORDER. The writeback side REPLAYS the same
    // (block,oc) order with its own counters → pong write address, no FIFO race.
    reg [6:0]  wb_block;   // block index of next result to write back
    reg [3:0]  wb_oc;      // oc of next result to write back
    reg [7:0]  in_flight;  // issued-but-not-written count (drain gate for PH_TRANS)
    wire       wb_empty = (in_flight == 8'd0);
    reg        issue_done; // pulse: one oc fully issued this cycle

    function [6:0] nblk;
        input [11:0] ilen;
        case (ilen)
            12'd2500: nblk=7'd125;
            12'd500 : nblk=7'd25;
            12'd100 : nblk=7'd5;
            12'd20  : nblk=7'd1;
            default : nblk=7'd1;
        endcase
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            layer_state<=IDLE; busy<=0; done<=0; result<=0; isram_free<=0;
            srw_rst<=0; blk_rst<=0; shift_en<=0; pad_zero<=0; compute_en<=0;
            a<=0; oc<=0; block_base<=0; pos_load<=0;
            bank_sel<=0; pong_waddr<=0; pong_we<=0; src_is_conv1<=0;
            in_ch<=0; in_len<=0; nb<=0; relu_en<=0;
            fc_sub_state<=0; gap_step<=0; fc_step<=0; argmax_step<=0;
            phase<=PH_PRIME; block_idx<=0; prime_cnt<=0; slide_cnt<=0;
            num_blocks<=1; out_ch_n<=0;
            wide_load<=0; wprime_word<=0; wpad4<=0; wstep<=0;
            wb_block<=0; wb_oc<=0; in_flight<=0; issue_done<=0;
        end else begin
            srw_rst<=0; blk_rst<=0; shift_en<=0; done<=0; pong_we<=0; compute_en<=0;
            wide_load<=0; issue_done<=0;

            // ── Writeback: replay (block,oc) order on each pooled_valid (any phase) ──
            if (issue_done) in_flight <= in_flight + 8'd1 - (pooled_valid ? 8'd1 : 8'd0);
            else if (pooled_valid) in_flight <= in_flight - 8'd1;
            if (pooled_valid) begin
                pong_we    <= (8'd1 << wb_oc);
                pong_waddr <= wb_block;
                if (wb_oc == out_ch_n - 4'd1) begin
                    wb_oc <= 4'd0;
                    wb_block <= (wb_block == num_blocks-7'd1) ? 7'd0 : wb_block + 7'd1;
                end else
                    wb_oc <= wb_oc + 4'd1;
            end

            case (layer_state)
            IDLE: begin busy<=0; if (start) begin layer_state<=LOAD; busy<=1; end end

            LOAD: begin
                layer_state<=CONV1; isram_free<=0; bank_sel<=0;
                in_ch<=4'd1; out_ch_n<=4'd4; in_len<=12'd2500; nb<=NB1; relu_en<=0;
                src_is_conv1<=1; num_blocks<=nblk(12'd2500);
                phase<=PH_WPRIME; block_idx<=0; block_base<=0; prime_cnt<=0;
                wstep<=0; srw_rst<=1; oc<=0; a<=0;
            end

            CONV1, CONV2, CONV3, CONV4: begin
                case (phase)
                // ── PRIME block 0: stream L+4 positions (logical -2..L+1) ───
                PH_PRIME: begin
                    // logical pos = block_base + prime_cnt - 2
                    pos_load <= block_base + prime_cnt;  // core subtracts 2 offset
                    pad_zero <= (prime_cnt < 5'd2) ||
                                (({1'b0,block_base} + prime_cnt) >= (in_len + 12'd2));
                    shift_en <= 1;
                    if (prime_cnt == (L+4-1)) begin
                        prime_cnt<=0; phase<=PH_PRIME_SETTLE; settle_cnt<=0; oc<=0;
                    end else
                        prime_cnt<=prime_cnt+5'd1;
                end

                // Drain the shift_en→shift_d (1cy) + pong-read (1cy) pipeline so the
                // last primed positions land in the line buffer before compute reads taps.
                // Settle the source-read pipeline, then preset a=0,oc=0,compute_en=1 and
                // jump straight into the continuous sweep (first SWEEP cycle issues
                // oc=0,a=0 with compute_en=1 — no wasted cycle).
                PH_PRIME_SETTLE: begin
                    shift_en<=0;
                    if (settle_cnt == 3'd1) begin phase<=PH_SWEEP; a<=4'd0; oc<=4'd0; compute_en<=1'b1; end
                    else settle_cnt<=settle_cnt+3'd1;
                end

                // PH_OC_RST kept only as a label (unused in pipelined flow).
                PH_OC_RST: phase<=PH_SWEEP;

                // ── PH_SWEEP: continuous pipelined issue (no per-oc drain) ───
                // Drive compute_en=1 every cycle; sweep a=0..in_ch-1, bump oc when a
                // wraps; mark issue_done on each oc's last a (writeback counts it).
                // No blk_rst per oc — the lane array's acc_init=(a_d5==0) starts each
                // oc cleanly (production-style). On the last oc's last a, stop and go
                // slide/trans; in-flight results drain via the writeback logic.
                PH_SWEEP: begin
                    compute_en <= 1'b1;
                    if (a == in_ch-4'd1) begin
                        issue_done <= 1'b1;
                        a <= 4'd0;
                        if (oc == out_ch_n-4'd1) begin
                            compute_en <= 1'b0; oc <= 4'd0;
                            if (block_idx == num_blocks-7'd1)
                                phase<=PH_TRANS;
                            else begin
                                phase<=PH_WSLIDE; wstep<=0;
                            end
                        end else
                            oc <= oc+4'd1;
                    end else
                        a <= a+4'd1;
                end

                // ── SLIDE: stream L new positions for next block ────────────
                // Core computes logical_pos = pos_load - 2. Buffer currently holds
                // logical [block_base-2 .. block_base+21]; the next block (base+L)
                // needs logical up to (block_base+L)+21 = block_base+41, so stream L
                // NEW logical positions [block_base+22 .. block_base+41].
                //   pos_load = logical + 2 = block_base + 24 + slide_cnt.
                PH_SLIDE: begin
                    pos_load <= block_base + 12'd24 + slide_cnt;
                    pad_zero <= ((block_base + 12'd22 + slide_cnt) >= in_len);
                    shift_en <= 1;
                    if (slide_cnt == (L-1)) begin
                        block_idx <= block_idx + 7'd1;
                        block_base<= block_base + L;
                        phase<=PH_SLIDE_SETTLE; settle_cnt<=0; oc<=0;
                    end else
                        slide_cnt<=slide_cnt+5'd1;
                end

                PH_SLIDE_SETTLE: begin
                    shift_en<=0;
                    if (settle_cnt == 3'd1) begin phase<=PH_SWEEP; a<=4'd0; oc<=4'd0; compute_en<=1'b1; end
                    else settle_cnt<=settle_cnt+3'd1;
                end

                // ── WIDE PRIME (block 0): fill positions 0..21 via 4-pos wide words
                // + 2 single-shift fixups (window offset -2 mod 4 makes pos 20,21 the
                // two newest, not word-aligned). srw_rst cleared the top 2 slots = left
                // pad (pos -1,-2). block_base==0 here. ~7 cy vs 24 for 1-pos PRIME.
                //   wstep 0..4 : wide words 0..4  (positions 0..19)
                //   wstep 5,6  : single shifts pos 20, 21 (newest)
                PH_WPRIME: begin
                    if (wstep <= 4'd4) begin
                        wide_load   <= 1'b1;
                        wprime_word <= {6'd0, wstep};
                        wpad4[0]    <= ((wstep*4 + 0) >= in_len);
                        wpad4[1]    <= ((wstep*4 + 1) >= in_len);
                        wpad4[2]    <= ((wstep*4 + 2) >= in_len);
                        wpad4[3]    <= ((wstep*4 + 3) >= in_len);
                        wstep       <= wstep + 4'd1;
                    end else begin
                        pos_load <= 12'd2 + (12'd15 + wstep); // log_pos = 15+wstep (20,21)
                        pad_zero <= ((12'd15 + wstep) >= in_len);
                        shift_en <= 1'b1;
                        if (wstep == 4'd6) begin
                            wstep<=0; phase<=PH_PRIME_SETTLE; settle_cnt<=0; oc<=0;
                        end else
                            wstep<=wstep+4'd1;
                    end
                end

                // ── WIDE SLIDE: stream 20 NEW positions [B+22..B+41] for next block.
                // Window starts offset 2 (mod 4): 2 single (B+22,B+23), 4 wide words
                // (B+24..B+39), 2 single (B+40,B+41). ~8 cy vs 20 for 1-pos SLIDE.
                PH_WSLIDE: begin
                    if (wstep <= 4'd1) begin
                        pos_load <= block_base + 12'd24 + wstep; // (B+22+wstep)+2
                        pad_zero <= ((block_base + 12'd22 + wstep) >= in_len);
                        shift_en <= 1'b1;
                        wstep    <= wstep + 4'd1;
                    end else if (wstep <= 4'd5) begin
                        wide_load   <= 1'b1;
                        wprime_word <= ((block_base + 12'd24) >> 2) + (wstep - 4'd2);
                        wpad4[0] <= ((block_base + 12'd24 + (wstep-4'd2)*4 + 0) >= in_len);
                        wpad4[1] <= ((block_base + 12'd24 + (wstep-4'd2)*4 + 1) >= in_len);
                        wpad4[2] <= ((block_base + 12'd24 + (wstep-4'd2)*4 + 2) >= in_len);
                        wpad4[3] <= ((block_base + 12'd24 + (wstep-4'd2)*4 + 3) >= in_len);
                        wstep    <= wstep + 4'd1;
                    end else begin
                        pos_load <= block_base + 12'd2 + (12'd34 + wstep); // (B+34+wstep)+2
                        pad_zero <= ((block_base + 12'd34 + wstep) >= in_len);
                        shift_en <= 1'b1;
                        if (wstep == 4'd7) begin
                            block_idx <= block_idx + 7'd1;
                            block_base<= block_base + L;
                            wstep<=0; phase<=PH_SLIDE_SETTLE; settle_cnt<=0; oc<=0;
                        end else
                            wstep<=wstep+4'd1;
                    end
                end

                // ── Layer transition ───────────────────────────────────────
                // Wait for all in-flight writebacks of the last block to drain before
                // flipping bank_sel (else a late write lands in the new bank).
                PH_TRANS: if (wb_empty && !pooled_valid) begin
                    bank_sel<=~bank_sel;
                    block_idx<=0; block_base<=0; prime_cnt<=0; oc<=0; a<=0;
                    wstep<=0; srw_rst<=1; phase<=PH_WPRIME;
                    wb_block<=0; wb_oc<=0; in_flight<=0;
                    blk_rst<=1'b1;   // flush lane pipeline between layers
                    case (layer_state)
                    CONV1: begin layer_state<=CONV2; isram_free<=1;
                        in_ch<=4'd4; out_ch_n<=4'd4; in_len<=12'd500; nb<=NB2;
                        relu_en<=0; src_is_conv1<=0; num_blocks<=nblk(12'd500); end
                    CONV2: begin layer_state<=CONV3;
                        in_ch<=4'd4; out_ch_n<=4'd8; in_len<=12'd100; nb<=NB3;
                        relu_en<=0; src_is_conv1<=0; num_blocks<=nblk(12'd100); end
                    CONV3: begin layer_state<=CONV4;
                        in_ch<=4'd8; out_ch_n<=4'd8; in_len<=12'd20; nb<=NB4;
                        relu_en<=1; src_is_conv1<=0; num_blocks<=nblk(12'd20); end
                    CONV4: begin layer_state<=GAP_FC; phase<=PH_PRIME;
                        fc_sub_state<=GAP_SUB; gap_step<=0; srw_rst<=0; end
                    endcase
                end
                endcase
            end

            GAP_FC: begin
                case (fc_sub_state)
                GAP_SUB: begin
                    gap_step<=gap_step+4'd1;
                    if (gap_step==4'd5) begin fc_sub_state<=FC_SUB; fc_step<=0; end
                end
                FC_SUB: begin
                    fc_step<=fc_step+4'd1;
                    if (fc_step==4'd9) fc_sub_state<=FC_FLUSH;
                end
                FC_FLUSH: begin fc_sub_state<=ARGMAX_SUB; argmax_step<=0; end
                ARGMAX_SUB: begin
                    argmax_step<=argmax_step+2'd1;
                    if (argmax_step==2'd3) fc_sub_state<=DONE_SUB;
                end
                DONE_SUB: begin layer_state<=DONE_S; done<=1; result<=argmax_result; end
                endcase
            end

            DONE_S: begin busy<=0; if (start) begin layer_state<=LOAD; busy<=1; end end
            default: layer_state<=IDLE;
            endcase
        end
    end

endmodule
