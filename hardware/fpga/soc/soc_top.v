// soc_top.v
// ============================================================================
// Quartus TOP-LEVEL for DE10-Standard on-board demo (Phase D).
//
// This wraps the Qsys system (HPS + Avalon interconnect + ecg_core) and adds:
//   - reset glue: core needs BOTH rst_n (async, active-low) and rst (sync,
//     active-high). HPS gives us one active-low reset; we derive the other.
//   - the HPS hard-IP external pins (DDR3, etc.) pass straight through.
//
// IMPORTANT — this file references a Qsys system named `soc_system`.
// You must FIRST build that system in Platform Designer (see README in this
// folder), then Generate HDL. Quartus then knows the `soc_system` module and
// its exact port list. The port list below is the CONVENTIONAL Cyclone V SoC
// shape — your generated `soc_system.v` may name ports slightly differently
// (Qsys appends the interface name). After Generate, open
// soc/synthesis/soc_system.v, copy the real port list, and reconcile.
//
// Do NOT treat this as drop-in: it is a TEMPLATE that compiles only after the
// Qsys system exists and its port names match. Mismatches are expected and
// must be fixed by hand against the generated module.
// ============================================================================

module soc_top (
    // ── Clock & on-board reset ──────────────────────────────────────────
    input  wire        FPGA_CLK1_50,     // 50 MHz on-board oscillator
    input  wire        KEY0_n,           // push-button, active-low (manual reset, optional)

    // ── HPS DDR3 (these MUST be brought to top — HPS hard IP needs them) ─
    output wire [14:0] HPS_DDR3_ADDR,
    output wire [2:0]  HPS_DDR3_BA,
    output wire        HPS_DDR3_CAS_n,
    output wire        HPS_DDR3_CKE,
    output wire        HPS_DDR3_CK_n,
    output wire        HPS_DDR3_CK_p,
    output wire        HPS_DDR3_CS_n,
    output wire [3:0]  HPS_DDR3_DM,
    inout  wire [31:0] HPS_DDR3_DQ,
    inout  wire [3:0]  HPS_DDR3_DQS_n,
    inout  wire [3:0]  HPS_DDR3_DQS_p,
    output wire        HPS_DDR3_ODT,
    output wire        HPS_DDR3_RAS_n,
    output wire        HPS_DDR3_RESET_n,
    input  wire        HPS_DDR3_RZQ,
    output wire        HPS_DDR3_WE_n
    // NOTE: a real DE10-Standard pin-out has many more HPS pins (Ethernet,
    // UART, SD, USB, I2C…). For a minimal "HPS boots Linux + talks to FPGA via
    // lightweight bridge" demo you only strictly need DDR3 + the HPS pins Qsys
    // marks as conduit. Add the rest from the DE10-Standard golden top when you
    // need those peripherals. Keeping this list minimal on purpose.
);

    // ── PLL: 50 MHz → 100 MHz core clock ────────────────────────────────
    // The accelerator core is constrained at 100 MHz (see 100mhz.sdc).
    // Generate an ALTERA PLL IP named `core_pll` (50 in → 100 out) and
    // instantiate it here. Placeholder wiring shown; replace with the real
    // PLL IP instance after you add it via IP Catalog.
    wire core_clk;     // 100 MHz to the core
    wire pll_locked;

    core_pll u_pll (
        .refclk   (FPGA_CLK1_50),
        .rst      (~KEY0_n),
        .outclk_0 (core_clk),
        .locked   (pll_locked)
    );

    // ── Reset glue ──────────────────────────────────────────────────────
    // HPS reset (h2f_reset) comes OUT of the Qsys system as hps_h2f_rst_n.
    // The accelerator core wants:
    //   rst_n : async active-low   → tie to (hps reset & pll_locked & KEY0)
    //   rst   : sync  active-high  → ~rst_n, synchronized into core_clk
    // We AND in pll_locked so the core stays in reset until the clock is stable.
    wire        hps_h2f_rst_n;            // from Qsys (HPS-to-FPGA reset)
    wire        core_rst_n = hps_h2f_rst_n & pll_locked & KEY0_n;

    // Synchronize the active-high sync reset into the core clock domain.
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
    wire core_rst = rst_sync_1;           // sync active-high

    // ── Qsys system ─────────────────────────────────────────────────────
    // soc_system contains: hps_0 (HPS hard IP) + mm_interconnect +
    // ecg_core (your accelerator, exported clk/reset/avalon interface).
    //
    // The ecg_core's Avalon slave is wired INSIDE Qsys to hps h2f_lw master,
    // so it does NOT appear as a port here. Only clk/reset conduits + HPS pins.
    soc_system u_soc (
        // clocks / resets fed into the system
        .ecg_clk_clk        (core_clk),       // core clock conduit (100 MHz)
        .ecg_reset_n_reset_n(core_rst_n),     // core async reset-n conduit
        .ecg_reset_h_reset  (core_rst),       // core sync  reset   conduit

        // HPS reset out (drive reset glue above)
        .hps_h2f_reset_reset_n(hps_h2f_rst_n),

        // HPS DDR3 conduit — pass straight to pins
        .memory_mem_a       (HPS_DDR3_ADDR),
        .memory_mem_ba      (HPS_DDR3_BA),
        .memory_mem_ck      (HPS_DDR3_CK_p),
        .memory_mem_ck_n    (HPS_DDR3_CK_n),
        .memory_mem_cke     (HPS_DDR3_CKE),
        .memory_mem_cs_n    (HPS_DDR3_CS_n),
        .memory_mem_ras_n   (HPS_DDR3_RAS_n),
        .memory_mem_cas_n   (HPS_DDR3_CAS_n),
        .memory_mem_we_n    (HPS_DDR3_WE_n),
        .memory_mem_reset_n (HPS_DDR3_RESET_n),
        .memory_mem_dq      (HPS_DDR3_DQ),
        .memory_mem_dqs     (HPS_DDR3_DQS_p),
        .memory_mem_dqs_n   (HPS_DDR3_DQS_n),
        .memory_mem_odt     (HPS_DDR3_ODT),
        .memory_mem_dm      (HPS_DDR3_DM),
        .oct_rzqin          (HPS_DDR3_RZQ)
    );

endmodule
