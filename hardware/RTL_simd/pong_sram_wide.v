// pong_sram_wide.v
// Wide ping-pong feature-map memory for the SIMD variant (SIMD.md §6, Phương án B).
//
// WHY WIDE: the SIMD pool emits 4 pooled values per block in ONE cycle. A byte-wide
// channel-major SRAM (1 value/cycle) would overflow for Conv1 (n=IN_CH=1 cycle/block
// → 4 in, 1 out). Storing 4 positions per 32-bit word lets the whole pool result
// land in 1 write/cycle. Bonus: a word read = 4 consecutive positions, which suits
// both line-buffer priming and GAP (sum of 4 pos/word) directly.
//
// LAYOUT: 8 channels × 2 banks, each an independent 32-bit-wide M10K.
//   word = {pos+3, pos+2, pos+1, pos+0} little-endian within the word:
//     word[7:0]=pos0, word[15:8]=pos1, word[23:16]=pos2, word[31:24]=pos3.
//   max words = max_out_len/4 = 500/4 = 125 → depth 128 (M10K 128×32).
//
// bank_sel=0: A=Ping(read), B=Pong(write).  bank_sel=1: swapped.
// Read: synchronous 1-cy, all 8 channels parallel → 256-bit dout (8 × 32b word).
// Write: per-channel we; lane-valid byte mask (wmask[3:0]) for partial last word
//        (unused for this model — every out_len divisible by 4 — kept for safety).

module pong_sram_wide #(
    parameter WORDS = 128   // 125 needed (500/4), pad to 128
) (
    input  wire        clk,
    input  wire        bank_sel,

    // Write port (Pong) — one 32-bit word (4 positions) per channel
    input  wire [6:0]  wr_addr,        // word address 0..124
    input  wire [255:0] din,           // 8 ch × 32b: din[ch*32 +: 32]
    input  wire [7:0]  we,             // per-channel word write enable
    input  wire [3:0]  wmask,          // per-position byte mask within the word (1=write)

    // Read port (Ping) — 8 ch × 32b word
    input  wire [6:0]  rd_addr,
    output reg  [255:0] dout            // dout[ch*32 +: 32]
);

    // 16 independent 32-bit M10K arrays (8 ch × 2 banks)
    (* ramstyle = "M10K" *) reg [31:0] mem_a0 [0:WORDS-1];
    (* ramstyle = "M10K" *) reg [31:0] mem_a1 [0:WORDS-1];
    (* ramstyle = "M10K" *) reg [31:0] mem_a2 [0:WORDS-1];
    (* ramstyle = "M10K" *) reg [31:0] mem_a3 [0:WORDS-1];
    (* ramstyle = "M10K" *) reg [31:0] mem_a4 [0:WORDS-1];
    (* ramstyle = "M10K" *) reg [31:0] mem_a5 [0:WORDS-1];
    (* ramstyle = "M10K" *) reg [31:0] mem_a6 [0:WORDS-1];
    (* ramstyle = "M10K" *) reg [31:0] mem_a7 [0:WORDS-1];
    (* ramstyle = "M10K" *) reg [31:0] mem_b0 [0:WORDS-1];
    (* ramstyle = "M10K" *) reg [31:0] mem_b1 [0:WORDS-1];
    (* ramstyle = "M10K" *) reg [31:0] mem_b2 [0:WORDS-1];
    (* ramstyle = "M10K" *) reg [31:0] mem_b3 [0:WORDS-1];
    (* ramstyle = "M10K" *) reg [31:0] mem_b4 [0:WORDS-1];
    (* ramstyle = "M10K" *) reg [31:0] mem_b5 [0:WORDS-1];
    (* ramstyle = "M10K" *) reg [31:0] mem_b6 [0:WORDS-1];
    (* ramstyle = "M10K" *) reg [31:0] mem_b7 [0:WORDS-1];

    reg [31:0] ra0,ra1,ra2,ra3,ra4,ra5,ra6,ra7;
    reg [31:0] rb0,rb1,rb2,rb3,rb4,rb5,rb6,rb7;
    reg        bank_sel_d;

    wire [7:0] we_a = bank_sel ? we : 8'h00;
    wire [7:0] we_b = bank_sel ? 8'h00 : we;

    // byte-masked write helper applied per memory below via function
    function [31:0] merge;
        input [31:0] old_w, new_w;
        input [3:0]  m;
        begin
            merge[7:0]   = m[0] ? new_w[7:0]   : old_w[7:0];
            merge[15:8]  = m[1] ? new_w[15:8]  : old_w[15:8];
            merge[23:16] = m[2] ? new_w[23:16] : old_w[23:16];
            merge[31:24] = m[3] ? new_w[31:24] : old_w[31:24];
        end
    endfunction

    always @(posedge clk) begin
        if (we_a[0]) mem_a0[wr_addr] <= merge(mem_a0[wr_addr], din[0*32+:32], wmask);
        ra0 <= mem_a0[rd_addr];
    end
    always @(posedge clk) begin
        if (we_a[1]) mem_a1[wr_addr] <= merge(mem_a1[wr_addr], din[1*32+:32], wmask);
        ra1 <= mem_a1[rd_addr];
    end
    always @(posedge clk) begin
        if (we_a[2]) mem_a2[wr_addr] <= merge(mem_a2[wr_addr], din[2*32+:32], wmask);
        ra2 <= mem_a2[rd_addr];
    end
    always @(posedge clk) begin
        if (we_a[3]) mem_a3[wr_addr] <= merge(mem_a3[wr_addr], din[3*32+:32], wmask);
        ra3 <= mem_a3[rd_addr];
    end
    always @(posedge clk) begin
        if (we_a[4]) mem_a4[wr_addr] <= merge(mem_a4[wr_addr], din[4*32+:32], wmask);
        ra4 <= mem_a4[rd_addr];
    end
    always @(posedge clk) begin
        if (we_a[5]) mem_a5[wr_addr] <= merge(mem_a5[wr_addr], din[5*32+:32], wmask);
        ra5 <= mem_a5[rd_addr];
    end
    always @(posedge clk) begin
        if (we_a[6]) mem_a6[wr_addr] <= merge(mem_a6[wr_addr], din[6*32+:32], wmask);
        ra6 <= mem_a6[rd_addr];
    end
    always @(posedge clk) begin
        if (we_a[7]) mem_a7[wr_addr] <= merge(mem_a7[wr_addr], din[7*32+:32], wmask);
        ra7 <= mem_a7[rd_addr];
    end

    always @(posedge clk) begin
        if (we_b[0]) mem_b0[wr_addr] <= merge(mem_b0[wr_addr], din[0*32+:32], wmask);
        rb0 <= mem_b0[rd_addr];
    end
    always @(posedge clk) begin
        if (we_b[1]) mem_b1[wr_addr] <= merge(mem_b1[wr_addr], din[1*32+:32], wmask);
        rb1 <= mem_b1[rd_addr];
    end
    always @(posedge clk) begin
        if (we_b[2]) mem_b2[wr_addr] <= merge(mem_b2[wr_addr], din[2*32+:32], wmask);
        rb2 <= mem_b2[rd_addr];
    end
    always @(posedge clk) begin
        if (we_b[3]) mem_b3[wr_addr] <= merge(mem_b3[wr_addr], din[3*32+:32], wmask);
        rb3 <= mem_b3[rd_addr];
    end
    always @(posedge clk) begin
        if (we_b[4]) mem_b4[wr_addr] <= merge(mem_b4[wr_addr], din[4*32+:32], wmask);
        rb4 <= mem_b4[rd_addr];
    end
    always @(posedge clk) begin
        if (we_b[5]) mem_b5[wr_addr] <= merge(mem_b5[wr_addr], din[5*32+:32], wmask);
        rb5 <= mem_b5[rd_addr];
    end
    always @(posedge clk) begin
        if (we_b[6]) mem_b6[wr_addr] <= merge(mem_b6[wr_addr], din[6*32+:32], wmask);
        rb6 <= mem_b6[rd_addr];
    end
    always @(posedge clk) begin
        if (we_b[7]) mem_b7[wr_addr] <= merge(mem_b7[wr_addr], din[7*32+:32], wmask);
        rb7 <= mem_b7[rd_addr];
    end

    always @(posedge clk) bank_sel_d <= bank_sel;

    always @(*) begin
        dout[0*32+:32] = bank_sel_d ? rb0 : ra0;
        dout[1*32+:32] = bank_sel_d ? rb1 : ra1;
        dout[2*32+:32] = bank_sel_d ? rb2 : ra2;
        dout[3*32+:32] = bank_sel_d ? rb3 : ra3;
        dout[4*32+:32] = bank_sel_d ? rb4 : ra4;
        dout[5*32+:32] = bank_sel_d ? rb5 : ra5;
        dout[6*32+:32] = bank_sel_d ? rb6 : ra6;
        dout[7*32+:32] = bank_sel_d ? rb7 : ra7;
    end

endmodule
