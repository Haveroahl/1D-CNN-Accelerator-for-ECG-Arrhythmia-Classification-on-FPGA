// ping_pong_sram.v
// 2 sets x 8 channels x 500 entries x 8-bit
// bank_sel=0: Set-A = Ping (read), Set-B = Pong (write)
// bank_sel=1: Set-B = Ping (read), Set-A = Pong (write)
//
// Read port  (Ping set): synchronous, 1-cycle latency, all 8 channels in parallel
// Write port (Pong set): synchronous, gated by we[ch] per channel
//
// Flat layout: mem_a[ch*500 + offset], mem_b[ch*500 + offset]
// Quartus typically infers 4 M10K (2 banks x 2 due to packing).

module ping_pong_sram (
    input  wire        clk,
    input  wire        bank_sel,   // 0: A=Ping B=Pong | 1: B=Ping A=Pong

    // Write port - Pong set (from cp_engine pool output)
    input  wire [8:0]  wr_addr,    // 0..499 (9-bit, ch*500+offset later)
    input  wire [63:0] din,        // 8 channels packed: din[ch*8+:8]
    input  wire [7:0]  we,         // per-channel write enable

    // Read port - Ping set (to cp_engine SRW / GAP engine)
    input  wire [8:0]  rd_addr,    // 0..499
    output reg  [63:0] dout        // 8 channels packed: dout[ch*8+:8]
);

    reg [7:0] mem_a [0:3999];   // 8ch x 500
    reg [7:0] mem_b [0:3999];

    integer ch;

    // Write port: write to Pong set
    always @(posedge clk) begin
        for (ch = 0; ch < 8; ch = ch + 1) begin
            if (we[ch]) begin
                if (bank_sel == 1'b0)
                    mem_b[ch*500 + wr_addr] <= din[ch*8 +: 8];
                else
                    mem_a[ch*500 + wr_addr] <= din[ch*8 +: 8];
            end
        end
    end

    // Read port: read from Ping set, 1-cycle synchronous latency
    always @(posedge clk) begin
        for (ch = 0; ch < 8; ch = ch + 1) begin
            if (bank_sel == 1'b0)
                dout[ch*8 +: 8] <= mem_a[ch*500 + rd_addr];
            else
                dout[ch*8 +: 8] <= mem_b[ch*500 + rd_addr];
        end
    end

endmodule
