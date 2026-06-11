// niosv_top.v
// ============================================================================
// Quartus TOP-LEVEL for DE10-Standard on-board demo — Nios V variant (Phase D, stage 2).
//
// Wraps the Qsys system `nios_system` (Nios V/m RISC-V soft-core + on-chip RAM +
// JTAG UART + ecg_core accelerator, all on an Avalon-MM interconnect) and adds a
// PLL (50->100 MHz) + reset glue. No DDR3, no HPS.
//
// The Nios V boots from on-chip RAM (initialised with the compiled main.c via
// nios_system_onchip_memory2_0.hex), loads the embedded ECG samples into the
// accelerator over Avalon, runs inference, and prints the class through the JTAG
// UART (read on the host with juart-terminal). To load fresh firmware without
// recompiling the bitstream, use niosv-download app.elf over JTAG.
//
// Ports of the generated `nios_system` (soc/nios_system/synthesis/nios_system.v):
//   clk_clk, reset_reset_n, ecg_reset_h_reset, intel_niosv_m_0platform_irq_rx_irq
// ============================================================================

module niosv_top (
    input  wire FPGA_CLK1_50,   // 50 MHz on-board oscillator
    input  wire KEY0_n          // push-button, active-low (manual reset)
    // No other pins: JTAG uses the dedicated JTAG TAP (USB-Blaster), not FPGA IO.
);

    // ── PLL: 50 MHz -> 100 MHz core clock ───────────────────────────────────
    // Reuses the same core_pll IP created for the JTAG variant
    // (soc/core_pll.qip, refclk 50 -> outclk_0 100, locked).
    wire core_clk;
    wire pll_locked;

    core_pll u_pll (
        .refclk   (FPGA_CLK1_50),
        .rst      (~KEY0_n),
        .outclk_0 (core_clk),
        .locked   (pll_locked)
    );

    // ── Reset glue ──────────────────────────────────────────────────────────
    // The whole SoC (Nios V + interconnect + core) runs on core_clk.
    //   reset_reset_n : async active-low system reset for Qsys (CPU, interconnect)
    //   ecg_reset_h   : sync  active-high reset conduit for ecg_core's .rst
    // Hold in reset until the PLL is locked and KEY0 is released.
    wire sys_rst_n = pll_locked & KEY0_n;

    reg rst_sync_0, rst_sync_1;
    always @(posedge core_clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            rst_sync_0 <= 1'b1;
            rst_sync_1 <= 1'b1;
        end else begin
            rst_sync_0 <= 1'b0;
            rst_sync_1 <= rst_sync_0;
        end
    end
    wire core_rst = rst_sync_1;   // sync active-high

    // ── Qsys SoC ──────────────────────────────────────────────────────────────
    wire irq;   // Nios V platform irq export (left open at top)

    nios_system u_sys (
        .clk_clk                            (core_clk),
        .reset_reset_n                      (sys_rst_n),
        .ecg_reset_h_reset                  (core_rst),
        .intel_niosv_m_0platform_irq_rx_irq (irq)
    );

endmodule
