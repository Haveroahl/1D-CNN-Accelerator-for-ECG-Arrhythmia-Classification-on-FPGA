// input_sram_asic.v  —  ASIC (Sky130/OpenLane) variant of input_sram.
//
// 2500 x 8-bit simple dual-port SRAM. Logic is IDENTICAL to input_sram.v
// (sync write + 1-cycle sync read) — that module already uses no FPGA-specific
// ramstyle pragma, so the only change here is documentation: on ASIC this maps
// to ONE SRAM macro. OpenRAM typically wants a power-of-two depth, so the macro
// is sized 4096x8; only words 0..2499 are used (wr_addr/rd_addr never exceed 2499).
//
// Write port: from external loader (wr_addr/din/we). Read port: from cp_engine.

module input_sram_asic (
    input  wire        clk,

    // Write port
    input  wire [11:0] wr_addr,   // 0..2499
    input  wire [7:0]  din,
    input  wire        we,

    // Read port (synchronous, 1-cycle latency)
    input  wire [11:0] rd_addr,   // 0..2499
    output reg  [7:0]  dout
);

    // 4096x8 to match a power-of-two SRAM macro; words 2500..4095 unused.
    reg [7:0] mem [0:4095];

    always @(posedge clk) begin
        if (we)
            mem[wr_addr] <= din;
    end

    always @(posedge clk) begin
        dout <= mem[rd_addr];
    end

endmodule
