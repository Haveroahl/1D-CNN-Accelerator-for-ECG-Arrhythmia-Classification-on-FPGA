// input_sram.v
// 2500 x 8-bit simple dual-port SRAM
// Write port: from avalon_slave (HPS can write anytime)
// Read port:  from cp_engine (1-cycle synchronous latency)
// Target: Intel Cyclone V — infers M10K block RAM

module input_sram (
    input  wire        clk,

    // Write port (from avalon_slave)
    input  wire [11:0] wr_addr,   // 0..2499
    input  wire [7:0]  din,
    input  wire        we,

    // Read port (from cp_engine, 1-cycle latency)
    input  wire [11:0] rd_addr,   // 0..2499
    output reg  [7:0]  dout
);

    reg [7:0] mem [0:2499];

    // Write
    always @(posedge clk) begin
        if (we)
            mem[wr_addr] <= din;
    end

    // Read (synchronous, 1-cycle latency)
    always @(posedge clk) begin
        dout <= mem[rd_addr];
    end

endmodule
