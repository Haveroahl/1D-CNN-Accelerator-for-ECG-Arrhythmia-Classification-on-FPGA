// ecg_accelerator_top_asic.v
// Avalon wrapper around ecg_core_asic — used ONLY for regression simulation so
// the existing tb_top driver (Avalon byte path) can exercise the ASIC core and
// its packed memories without rewriting the testbench stimulus.
//
// This is NOT the ASIC chip top (that is ecg_core_asic with parallel pins). It
// exists purely to prove the macro-friendly memory refactor is bit-exact: same
// avalon_slave + same FSM/datapath, only the two memory modules differ.

module ecg_accelerator_top_asic (
    input  wire        clk,
    input  wire        rst,
    input  wire        rst_n,
    input  wire [12:0] avs_address,
    input  wire        avs_write,
    input  wire        avs_read,
    input  wire [31:0] avs_writedata,
    output wire [31:0] avs_readdata
);

    wire [11:0] sram_wr_addr;
    wire [7:0]  sram_din;
    wire        sram_we;
    wire        start;
    wire        busy;
    wire        done;
    wire [1:0]  result;

    // input_sram read port (core ↔ wrapper-resident input_sram_asic)
    wire [11:0] input_rd_addr;
    wire [7:0]  input_dout;

    avalon_slave u_avs (
        .clk          (clk),
        .rst_n        (rst_n),
        .avs_address  (avs_address),
        .avs_write    (avs_write),
        .avs_read     (avs_read),
        .avs_writedata(avs_writedata),
        .avs_readdata (avs_readdata),
        .sram_wr_addr (sram_wr_addr),
        .sram_din     (sram_din),
        .sram_we      (sram_we),
        .start        (start),
        .busy         (busy),
        .done         (done),
        .result       (result)
    );

    // input_sram (macro variant) — input I/O buffer, lives in the wrapper
    input_sram_asic u_isram (
        .clk    (clk),
        .wr_addr(sram_wr_addr),
        .din    (sram_din),
        .we     (sram_we),
        .rd_addr(input_rd_addr),
        .dout   (input_dout)
    );

    ecg_core_asic u_core (
        .clk          (clk),
        .rst          (rst),
        .input_rd_addr(input_rd_addr),
        .input_dout   (input_dout),
        .start        (start),
        .busy         (busy),
        .done         (done),
        .result       (result)
    );

endmodule
