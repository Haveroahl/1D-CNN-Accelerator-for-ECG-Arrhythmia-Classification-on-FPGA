// jtag_top.v
// ============================================================================
// Quartus TOP-LEVEL for DE10-Standard on-board demo — JTAG variant (Phase D).
//
// WHY this file instead of soc_top.v:
//   soc_top.v drives the core from the HPS via the Lightweight bridge. The HPS
//   (Cyclone V Hard Processor System) IP is NOT available in Quartus Prime LITE
//   — it is license-gated to the Standard edition. To run the verified core on
//   a board WITHOUT the HPS, we drive its Avalon-MM slave from the PC over JTAG.
//
//   PC (System Console, Tcl) ──USB-Blaster/JTAG──► JTAG-to-Avalon Master Bridge
//                                                        │ Avalon-MM
//                                                        ▼
//                                              ecg_core.avs (avalon_slave+core)
//
// This top wraps the Qsys system `jtag_system` (jtag master + clock + reset +
// ecg_core) and adds a PLL (50→100 MHz) + reset glue. No DDR3, no HPS pins.
//
// IMPORTANT — references a Qsys system named `jtag_system`. Build it in Platform
// Designer first (see JTAG_PHASE_D_STEPS.md), Generate HDL, then open
// soc/synthesis/jtag_system.v and reconcile the exact exported port names below
// (Qsys appends interface names like _clk_clk, _reset_reset, etc.).
// ============================================================================

module jtag_top (
    input  wire FPGA_CLK1_50,   // 50 MHz on-board oscillator
    input  wire KEY0_n          // push-button, active-low (manual reset)
    // No other pins: JTAG uses the dedicated JTAG TAP (USB-Blaster), not FPGA IO.
);

    // ── PLL: 50 MHz → 100 MHz core clock ────────────────────────────────
    // Core is constrained at 100 MHz (jtag_top.sdc). Add an "PLL Intel FPGA IP"
    // named `core_pll` (refclk 50 in → outclk_0 100 out, enable `locked`).
    wire core_clk;     // 100 MHz to the core
    wire pll_locked;

    core_pll u_pll (
        .refclk   (FPGA_CLK1_50),
        .rst      (~KEY0_n),
        .outclk_0 (core_clk),
        .locked   (pll_locked)
    );

    // ── Reset glue ──────────────────────────────────────────────────────
    // Core wants rst_n (async active-low) + rst (sync active-high).
    // Hold in reset until PLL locked and KEY0 released.
    wire core_rst_n = pll_locked & KEY0_n;

    reg  rst_sync_0, rst_sync_1;
    always @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            rst_sync_0 <= 1'b1;
            rst_sync_1 <= 1'b1;
        end else begin
            rst_sync_0 <= 1'b0;
            rst_sync_1 <= rst_sync_0;
        end
    end
    wire core_rst = rst_sync_1;   // sync active-high

    // ── Qsys system ─────────────────────────────────────────────────────
    // jtag_system contains: jtag master + clock/reset bridges + ecg_core.
    // The ecg_core Avalon slave is wired INSIDE Qsys to the JTAG master, so it
    // does not appear as a port here — only the clk/reset conduits.
    jtag_system u_sys (
        .ecg_clk_clk         (core_clk),     // core clock conduit (100 MHz)
        .ecg_reset_n_reset_n (core_rst_n),   // core async reset-n conduit
        .ecg_reset_h_reset   (core_rst)      // core sync  reset   conduit
    );

endmodule
