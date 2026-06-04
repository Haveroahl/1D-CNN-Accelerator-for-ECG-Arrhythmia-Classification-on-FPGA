# =============================================================================
# Synopsys Design Constraints (SDC) — Phase D (on-board HPS)
# Top      : soc_top  (Qsys soc_system: HPS + interconnect + ecg_core + PLL)
# Device   : Intel Cyclone V (DE10-Standard, 5CSXFC6D6F31C6, speed grade -6)
# =============================================================================
#
# DO NOT reuse the Phase C SDC (ecg_accelerator_top_100mhz.sdc) here.
# In Phase D:
#   - avs_* and the core clk are NOT top-level ports anymore — they live inside
#     the Qsys system. Constraining them with set_input/output_delay on
#     [get_ports ...] would fail (ports don't exist) or be wrong.
#   - The HPS hard IP + lightweight-bridge timing is handled by the
#     Qsys-generated soc_system.sdc (pulled in via the .qip). Do NOT re-constrain.
#
# This file only needs to:
#   1. Tell TimeQuest the board oscillator frequency (FPGA_CLK1_50, 50 MHz).
#   2. Let derive_pll_clocks discover core_clk (100 MHz) from the PLL IP.
#   3. Cut the async reset path (KEY0_n).
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Board oscillator — 50 MHz on-board clock into the PLL
# -----------------------------------------------------------------------------
create_clock -name FPGA_CLK1_50 -period 20.000 [get_ports {FPGA_CLK1_50}]

# -----------------------------------------------------------------------------
# 2. Derive PLL output clocks (core_pll: 50 MHz -> 100 MHz core_clk)
# -----------------------------------------------------------------------------
# derive_pll_clocks creates the generated clock on core_pll|...|outclk_0
# automatically from the PLL config — no manual create_generated_clock needed.
derive_pll_clocks
derive_clock_uncertainty

# -----------------------------------------------------------------------------
# 3. Async reset — false path
# -----------------------------------------------------------------------------
# KEY0_n is an async push-button; the core's resets are synchronized in soc_top
# (rst_sync_* registers) so the raw input edge is a false path.
set_false_path -from [get_ports {KEY0_n}]

# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------
# - HPS DDR3 / IO timing: handled entirely by the HPS hard IP + Qsys-generated
#   constraints. Nothing to add here.
# - If you used option (B) in the README (h2f_user0_clock instead of an FPGA
#   PLL), replace section 2 with a create_clock on that HPS-exported clock and
#   drop derive_pll_clocks.
# =============================================================================
