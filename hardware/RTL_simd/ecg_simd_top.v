// ecg_simd_top.v
// Thin wrapper for the SIMD-20 variant: bus adapter (avalon_slave, reused verbatim
// from production) + SIMD core (ecg_core_simd). Port list IDENTICAL to production
// ecg_accelerator_top so the same testbench Avalon helpers and Quartus project shell
// can drive it.
//
//   HPS ──avs_*──► avalon_slave ──8 wires──► ecg_core_simd
//
// The input 2500 buffer lives INSIDE ecg_core_simd (input_buffer.v) per SIMD.md §3b
// (Mô hình 2: buffer at wrapper boundary, core streams). avalon_slave writes it via
// the same sram_wr_addr/din/we path as production input_sram.

module ecg_simd_top (
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
    wire        start, busy, done, isram_free;
    wire [1:0]  result;

    avalon_slave u_avs (
        .clk(clk), .rst_n(rst_n),
        .avs_address(avs_address), .avs_write(avs_write), .avs_read(avs_read),
        .avs_writedata(avs_writedata), .avs_readdata(avs_readdata),
        .sram_wr_addr(sram_wr_addr), .sram_din(sram_din), .sram_we(sram_we),
        .start(start), .busy(busy), .done(done), .result(result),
        .isram_free(isram_free)
    );

    ecg_core_simd u_core (
        .clk(clk), .rst(rst),
        .sram_wr_addr(sram_wr_addr), .sram_din(sram_din), .sram_we(sram_we),
        .start(start), .busy(busy), .done(done), .result(result),
        .isram_free(isram_free)
    );
endmodule
