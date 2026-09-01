// ecg_uart_top.v
// UART-driven top level — sibling of ecg_accelerator_top.v (which uses Avalon).
// Same bus-agnostic ecg_core, but the adapter is uart_wrapper instead of
// avalon_slave, so the design can be driven from a PC serial line with no
// Nios V / HPS. ecg_core.v and all inner modules are reused unchanged.
//
//   PC ──RX/TX──► uart_wrapper ──8 wires──► ecg_core
//
// Pick this top in the Quartus project for a standalone UART board flow;
// pick ecg_accelerator_top for the Avalon/Nios V flow. Both share ecg_core.

module ecg_uart_top #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD     = 115_200
) (
    input  wire clk,
    input  wire rst,         // synchronous reset (active high)

    input  wire uart_rx,     // async serial in
    output wire uart_tx      // serial out
);

    // ── 8-wire core interface (driven/consumed by uart_wrapper) ────────
    wire [11:0] sram_wr_addr;
    wire [7:0]  sram_din;
    wire        sram_we;
    wire        start;
    wire        busy;
    wire        done;
    wire [1:0]  result;
    wire        isram_free;

    // ── uart_wrapper (bus adapter) ─────────────────────────────────────
    uart_wrapper #(.CLK_FREQ(CLK_FREQ), .BAUD(BAUD)) u_uart (
        .clk          (clk),
        .rst          (rst),
        .uart_rx      (uart_rx),
        .uart_tx      (uart_tx),
        .sram_wr_addr (sram_wr_addr),
        .sram_din     (sram_din),
        .sram_we      (sram_we),
        .start        (start),
        .busy         (busy),
        .done         (done),
        .result       (result),
        .isram_free   (isram_free)
    );

    // ── ecg_core (bus-agnostic accelerator, unchanged) ─────────────────
    ecg_core u_core (
        .clk          (clk),
        .rst          (rst),
        .sram_wr_addr (sram_wr_addr),
        .sram_din     (sram_din),
        .sram_we      (sram_we),
        .start        (start),
        .busy         (busy),
        .done         (done),
        .result       (result),
        .isram_free   (isram_free)
    );

endmodule
