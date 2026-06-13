// uart_board_top.v
// DE10-Standard board-level top for the UART-driven ECG accelerator variant.
// Maps the physical board pins (50 MHz osc, KEY0 reset push-button, UART pins)
// onto ecg_uart_top — the same role niosv_top.v plays for the Nios V variant.
//
//   FPGA_CLK1_50 ─► clk (50 MHz, no PLL — core Fmax ~105 MHz, comfortable)
//   KEY0_n (async, active-low) ─► 2-FF sync ─► rst (sync, active-high)
//   UART_RXD / UART_TXD ─► ecg_uart_top.uart_rx / uart_tx
//
// CLK_FREQ is set to 50e6 so the UART baud divisor matches the 50 MHz clock.
// ecg_uart_top + ecg_core + all inner modules are reused unchanged.

module uart_board_top (
    input  wire FPGA_CLK1_50,
    input  wire KEY0_n,        // async active-low reset push-button
    input  wire UART_RXD,      // serial in  (assign GPIO pin in .qsf)
    output wire UART_TXD       // serial out (assign GPIO pin in .qsf)
);

    // ── Reset synchronizer: async KEY0_n (active-low) → sync active-high ──
    reg rst_meta, rst_sync;
    always @(posedge FPGA_CLK1_50 or negedge KEY0_n) begin
        if (!KEY0_n) begin
            rst_meta <= 1'b1;
            rst_sync <= 1'b1;
        end else begin
            rst_meta <= 1'b0;
            rst_sync <= rst_meta;
        end
    end

    ecg_uart_top #(
        .CLK_FREQ (50_000_000),
        .BAUD     (115_200)
    ) u_top (
        .clk     (FPGA_CLK1_50),
        .rst     (rst_sync),
        .uart_rx (UART_RXD),
        .uart_tx (UART_TXD)
    );

endmodule
