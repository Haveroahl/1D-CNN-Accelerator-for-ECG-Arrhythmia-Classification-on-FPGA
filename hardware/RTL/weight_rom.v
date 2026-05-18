// weight_rom.v
// Weight ROM for one CP block (5 INT8 weights) — synchronous read, 1-cycle latency
// Instantiated once per (layer, output_channel) pair in cp_engine.
//
// Parameters:
//   INIT_FILE : path to .hex file with 5 signed bytes (w[0]..w[4])
//
// Address 0..4 → tap weight w[0]..w[4]
// Read is always enabled — addr driven by controller tap counter (0..4 during pre-fetch+compute)

module weight_rom #(
    parameter INIT_FILE = "weights.hex"
) (
    input  wire       clk,
    input  wire [2:0] addr,    // 0..4 (5 weights per output channel)
    output reg  signed [7:0] dout
);

    reg signed [7:0] mem [0:4];

    initial begin
        $readmemh(INIT_FILE, mem);
    end

    always @(posedge clk)
        dout <= mem[addr];

endmodule


// bias_rom.v inlined below as a separate module — bias is INT32, 1 entry per output channel.
// In cp_block, bias is loaded once at layer start (addr=0 always), so no address needed.
// Kept as ROM so Quartus can infer from .hex and pack into logic/M10K as needed.

module bias_rom #(
    parameter INIT_FILE = "bias.hex"
) (
    input  wire       clk,
    input  wire [2:0] addr,    // 0..7 (up to 8 output channels per layer)
    output reg  signed [31:0] dout
);

    reg signed [31:0] mem [0:7];

    initial begin
        $readmemh(INIT_FILE, mem);
    end

    always @(posedge clk)
        dout <= mem[addr];

endmodule
