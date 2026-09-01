// ping_pong_sram.v — DE0-Nano (Cyclone IV E) fork of ../../RTL/ping_pong_sram.v
// ----------------------------------------------------------------------------
// ONLY difference vs the Cyclone V original: ramstyle hint "M10K" -> "M9K".
// Cyclone IV E has no M10K block (that is a Cyclone V primitive); its block RAM
// is the M9K (8192/9216 bit). Each mem here is 512x8 = 4096 bit, fits 1 M9K.
// The inference template (1 mem ref, sync write + sync read into a dedicated
// register, per memory) is unchanged and maps to an M9K simple-dual-port.
// Keep this file byte-identical to the original APART from the hint so behaviour
// stays bit-exact; re-sync if the original changes.
// ----------------------------------------------------------------------------
// 2 sets x 8 channels x 500 entries x 8-bit (pad to 512 for M9K mode 512x8/9)
// bank_sel=0: Set-A = Ping (read), Set-B = Pong (write)
// bank_sel=1: Set-B = Ping (read), Set-A = Pong (write)
//
// Read port  (Ping set): synchronous, 1-cycle latency, all 8 channels parallel
// Write port (Pong set): synchronous, gated by we[ch] per channel

module ping_pong_sram (
    input  wire        clk,
    input  wire        bank_sel,   // 0: A=Ping B=Pong | 1: B=Ping A=Pong

    // Write port - Pong set (from cp_engine pool output)
    input  wire [8:0]  wr_addr,    // 0..499
    input  wire [63:0] din,        // 8 channels packed: din[ch*8+:8]
    input  wire [7:0]  we,         // per-channel write enable

    // Read port - Ping set (to cp_engine SRW / GAP engine)
    input  wire [8:0]  rd_addr,    // 0..499
    output reg  [63:0] dout        // 8 channels packed: dout[ch*8+:8]
);

    // ── 16 separate mem arrays with M9K ramstyle hint (Cyclone IV E) ────
    (* ramstyle = "M9K" *) reg [7:0] mem_a_ch0 [0:511];
    (* ramstyle = "M9K" *) reg [7:0] mem_a_ch1 [0:511];
    (* ramstyle = "M9K" *) reg [7:0] mem_a_ch2 [0:511];
    (* ramstyle = "M9K" *) reg [7:0] mem_a_ch3 [0:511];
    (* ramstyle = "M9K" *) reg [7:0] mem_a_ch4 [0:511];
    (* ramstyle = "M9K" *) reg [7:0] mem_a_ch5 [0:511];
    (* ramstyle = "M9K" *) reg [7:0] mem_a_ch6 [0:511];
    (* ramstyle = "M9K" *) reg [7:0] mem_a_ch7 [0:511];

    (* ramstyle = "M9K" *) reg [7:0] mem_b_ch0 [0:511];
    (* ramstyle = "M9K" *) reg [7:0] mem_b_ch1 [0:511];
    (* ramstyle = "M9K" *) reg [7:0] mem_b_ch2 [0:511];
    (* ramstyle = "M9K" *) reg [7:0] mem_b_ch3 [0:511];
    (* ramstyle = "M9K" *) reg [7:0] mem_b_ch4 [0:511];
    (* ramstyle = "M9K" *) reg [7:0] mem_b_ch5 [0:511];
    (* ramstyle = "M9K" *) reg [7:0] mem_b_ch6 [0:511];
    (* ramstyle = "M9K" *) reg [7:0] mem_b_ch7 [0:511];

    // ── Dedicated read-data registers (one per memory) ─────────────────
    reg [7:0] rd_a_ch0, rd_a_ch1, rd_a_ch2, rd_a_ch3;
    reg [7:0] rd_a_ch4, rd_a_ch5, rd_a_ch6, rd_a_ch7;
    reg [7:0] rd_b_ch0, rd_b_ch1, rd_b_ch2, rd_b_ch3;
    reg [7:0] rd_b_ch4, rd_b_ch5, rd_b_ch6, rd_b_ch7;
    reg       bank_sel_d;

    // ── Bank-gated write enables (precompute outside always) ───────────
    // bank_sel=0 → Pong=B, bank_sel=1 → Pong=A
    wire [7:0] we_a = bank_sel ? we : 8'h00;
    wire [7:0] we_b = bank_sel ? 8'h00 : we;

    // ── Per-memory always blocks: standard M9K inference template ──────
    // Each block: 1 mem ref, sync write, sync read into dedicated register.
    always @(posedge clk) begin
        if (we_a[0]) mem_a_ch0[wr_addr] <= din[0*8 +: 8];
        rd_a_ch0 <= mem_a_ch0[rd_addr];
    end
    always @(posedge clk) begin
        if (we_a[1]) mem_a_ch1[wr_addr] <= din[1*8 +: 8];
        rd_a_ch1 <= mem_a_ch1[rd_addr];
    end
    always @(posedge clk) begin
        if (we_a[2]) mem_a_ch2[wr_addr] <= din[2*8 +: 8];
        rd_a_ch2 <= mem_a_ch2[rd_addr];
    end
    always @(posedge clk) begin
        if (we_a[3]) mem_a_ch3[wr_addr] <= din[3*8 +: 8];
        rd_a_ch3 <= mem_a_ch3[rd_addr];
    end
    always @(posedge clk) begin
        if (we_a[4]) mem_a_ch4[wr_addr] <= din[4*8 +: 8];
        rd_a_ch4 <= mem_a_ch4[rd_addr];
    end
    always @(posedge clk) begin
        if (we_a[5]) mem_a_ch5[wr_addr] <= din[5*8 +: 8];
        rd_a_ch5 <= mem_a_ch5[rd_addr];
    end
    always @(posedge clk) begin
        if (we_a[6]) mem_a_ch6[wr_addr] <= din[6*8 +: 8];
        rd_a_ch6 <= mem_a_ch6[rd_addr];
    end
    always @(posedge clk) begin
        if (we_a[7]) mem_a_ch7[wr_addr] <= din[7*8 +: 8];
        rd_a_ch7 <= mem_a_ch7[rd_addr];
    end

    always @(posedge clk) begin
        if (we_b[0]) mem_b_ch0[wr_addr] <= din[0*8 +: 8];
        rd_b_ch0 <= mem_b_ch0[rd_addr];
    end
    always @(posedge clk) begin
        if (we_b[1]) mem_b_ch1[wr_addr] <= din[1*8 +: 8];
        rd_b_ch1 <= mem_b_ch1[rd_addr];
    end
    always @(posedge clk) begin
        if (we_b[2]) mem_b_ch2[wr_addr] <= din[2*8 +: 8];
        rd_b_ch2 <= mem_b_ch2[rd_addr];
    end
    always @(posedge clk) begin
        if (we_b[3]) mem_b_ch3[wr_addr] <= din[3*8 +: 8];
        rd_b_ch3 <= mem_b_ch3[rd_addr];
    end
    always @(posedge clk) begin
        if (we_b[4]) mem_b_ch4[wr_addr] <= din[4*8 +: 8];
        rd_b_ch4 <= mem_b_ch4[rd_addr];
    end
    always @(posedge clk) begin
        if (we_b[5]) mem_b_ch5[wr_addr] <= din[5*8 +: 8];
        rd_b_ch5 <= mem_b_ch5[rd_addr];
    end
    always @(posedge clk) begin
        if (we_b[6]) mem_b_ch6[wr_addr] <= din[6*8 +: 8];
        rd_b_ch6 <= mem_b_ch6[rd_addr];
    end
    always @(posedge clk) begin
        if (we_b[7]) mem_b_ch7[wr_addr] <= din[7*8 +: 8];
        rd_b_ch7 <= mem_b_ch7[rd_addr];
    end

    // ── Bank_sel delay 1cy to align with sync-read latency ─────────────
    always @(posedge clk) bank_sel_d <= bank_sel;

    // ── Output mux: combinational, 1 LUT level (bank_sel 2:1 per bit) ──
    always @(*) begin
        dout[0*8 +: 8] = bank_sel_d ? rd_b_ch0 : rd_a_ch0;
        dout[1*8 +: 8] = bank_sel_d ? rd_b_ch1 : rd_a_ch1;
        dout[2*8 +: 8] = bank_sel_d ? rd_b_ch2 : rd_a_ch2;
        dout[3*8 +: 8] = bank_sel_d ? rd_b_ch3 : rd_a_ch3;
        dout[4*8 +: 8] = bank_sel_d ? rd_b_ch4 : rd_a_ch4;
        dout[5*8 +: 8] = bank_sel_d ? rd_b_ch5 : rd_a_ch5;
        dout[6*8 +: 8] = bank_sel_d ? rd_b_ch6 : rd_a_ch6;
        dout[7*8 +: 8] = bank_sel_d ? rd_b_ch7 : rd_a_ch7;
    end

endmodule
