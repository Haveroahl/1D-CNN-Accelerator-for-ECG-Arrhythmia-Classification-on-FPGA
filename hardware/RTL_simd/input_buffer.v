// input_buffer.v
// Wrapper-side input buffer for Conv1 (SIMD.md §3b).
//
// WIDE-4 (4 pos/word): exposes 625 × 32-bit words = 4 INT8 positions/word
//   word[7:0]=pos(4w+0), word[15:8]=pos(4w+1), word[23:16]=pos(4w+2), word[31:24]=pos(4w+3).
// This lets Conv1 priming/slide read 4 consecutive positions per cycle (matching the
// pong-wide source for Conv2-4), so the line-buffer slide is ~5cy/block instead of 20.
//
// STORAGE: 4 independent byte banks of 625×8b, one per (position % 4). Each bank is a
// simple-dual-port byte RAM → infers M10K. (The earlier single 625×32b array used a
// byte-granular read-modify-write write port, which Quartus cannot infer as RAM — it
// fell back to ~20k flip-flops = 64% of the chip's ALMs. Splitting by lane removes the
// sub-word write entirely: each write touches exactly one bank's full 8-bit word.)
//
// Write port: byte-at-a-time from avalon_slave (HPS/JTAG/UART), addressed by full
// position index (0..2499). bank = pos%4, word = pos/4.
// Read port: word address (0..624), 1-cy synchronous → 32-bit word (4 positions),
// assembled from the 4 banks at the same word address.

module input_buffer (
    input  wire        clk,

    // Write (from bus adapter) — per-POSITION byte (addr 0..2499)
    input  wire [11:0] wr_addr,      // position 0..2499
    input  wire [7:0]  din,
    input  wire        we,

    // Wide read (to core) — word address 0..624, 1-cy synchronous → 4-pos word
    input  wire [9:0]  rd_word,      // word 0..624 (needs 10 bits)
    output wire [31:0] word_out
);
    // 4 byte banks, one per (position % 4). Each infers an M10K.
    reg [7:0] mem0 [0:624];
    reg [7:0] mem1 [0:624];
    reg [7:0] mem2 [0:624];
    reg [7:0] mem3 [0:624];

    wire [9:0] w_word = wr_addr[11:2];   // position / 4 (0..624)
    wire [1:0] w_byte = wr_addr[1:0];    // position % 4 → bank select

    always @(posedge clk) begin
        if (we && w_byte == 2'd0) mem0[w_word] <= din;
    end
    always @(posedge clk) begin
        if (we && w_byte == 2'd1) mem1[w_word] <= din;
    end
    always @(posedge clk) begin
        if (we && w_byte == 2'd2) mem2[w_word] <= din;
    end
    always @(posedge clk) begin
        if (we && w_byte == 2'd3) mem3[w_word] <= din;
    end

    reg [7:0] q0, q1, q2, q3;
    always @(posedge clk) q0 <= mem0[rd_word];
    always @(posedge clk) q1 <= mem1[rd_word];
    always @(posedge clk) q2 <= mem2[rd_word];
    always @(posedge clk) q3 <= mem3[rd_word];

    assign word_out = {q3, q2, q1, q0};  // byte0=pos%4==0 (oldest) ... byte3=pos%4==3
endmodule
