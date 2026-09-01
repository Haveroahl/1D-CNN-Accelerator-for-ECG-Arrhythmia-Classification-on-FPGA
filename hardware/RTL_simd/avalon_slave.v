// avalon_slave.v
// Avalon-MM slave interface for ECG accelerator on DE10-Standard.
// Bridges HPS Lightweight bridge to: Input SRAM write port + control/status registers.
//
// Address map (WORDS, 13-bit address):
//   Low registers (unchanged, byte-at-a-time path — used by tb_top.v):
//     0x0000 W : sram_din      [7:0]
//     0x0001 W : sram_wr_addr  [11:0]
//     0x0002 W : sram_we       [0]
//     0x0003 W : start         [0]   (clears done_latched)
//     0x0004 R : status        {isram_free, done_latched, busy}
//     0x0005 R : result        [1:0]
//   DATA WINDOW (burst path — used by JTAG host master_write_memory):
//     0x1000..0x19C3 W : write word -> sram_din<=wd[7:0], sram_wr_addr<=(addr-0x1000),
//                        sram_we<=1.  addr[12]=1 selects the window; index = addr[11:0]
//                        (0..2499). One System-Console block write loads a whole sample.
// The window is purely additive: the low-register path is byte-identical to the
// previous 5-bit slave, so tb_top.v (21/21 bit-exact) is unaffected.

module avalon_slave (
    input  wire        clk,
    input  wire        rst_n,

    // Avalon-MM Slave
    input  wire [12:0] avs_address,
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

            if (avs_write) begin
                if (avs_address[12]) begin
                    // ── DATA WINDOW: one word = one SRAM byte (auto din+addr+we) ──
                    sram_din     <= avs_writedata[7:0];
                    sram_wr_addr <= avs_address[11:0];
                    sram_we      <= 1'b1;
                end else begin
                    // ── Low registers (legacy byte-at-a-time path) ──
                    case (avs_address[2:0])
                        3'h0: sram_din     <= avs_writedata[7:0];
                        3'h1: sram_wr_addr <= avs_writedata[11:0];
                        3'h2: sram_we      <= avs_writedata[0];
                        3'h3: begin
                            start        <= avs_writedata[0];
                            done_latched <= 1'b0;
                        end
                    endcase
                end
            end

            if (avs_read)
                case ({avs_address[12], avs_address[2:0]})
                    4'h4: avs_readdata <= {29'b0, isram_free, done_latched, busy};
                    4'h5: avs_readdata <= {30'b0, result};
                    default: avs_readdata <= 32'b0;
                endcase
        end
    end

endmodule
