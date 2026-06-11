// avalon_slave.v
// Avalon-MM slave interface for ECG accelerator on DE10-Standard.
// Bridges HPS Lightweight bridge to: Input SRAM write port + control/status registers.

module avalon_slave (
    input  wire        clk,
    input  wire        rst_n,

    // Avalon-MM Slave
    input  wire [4:0]  avs_address,
    input  wire        avs_write,
    input  wire        avs_read,
    input  wire [31:0] avs_writedata,
    output reg  [31:0] avs_readdata,

    // Input SRAM write port
    output reg  [11:0] sram_wr_addr,
    output reg  [7:0]  sram_din,
    output reg         sram_we,

    // Control/status
    output reg         start,
    input  wire        busy,
    input  wire        done,
    input  wire [1:0]  result,
    input  wire        isram_free   // core: input_sram free to reload (CONV2..DONE)
);

    reg done_latched;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done_latched <= 1'b0;
            start        <= 1'b0;
            sram_we      <= 1'b0;
            sram_din     <= 8'h00;
            sram_wr_addr <= 12'h000;
            avs_readdata <= 32'h0;
        end else begin
            start   <= 1'b0;
            sram_we <= 1'b0;

            if (done)
                done_latched <= 1'b1;

            if (avs_write)
                case (avs_address)
                    5'h00: sram_din     <= avs_writedata[7:0];
                    5'h01: sram_wr_addr <= avs_writedata[11:0];
                    5'h02: sram_we      <= avs_writedata[0];
                    5'h03: begin
                        start        <= avs_writedata[0];
                        done_latched <= 1'b0;
                    end
                endcase

            if (avs_read)
                case (avs_address)
                    5'h04: avs_readdata <= {29'b0, isram_free, done_latched, busy};
                    5'h05: avs_readdata <= {30'b0, result};
                    default: avs_readdata <= 32'b0;
                endcase
        end
    end

endmodule
