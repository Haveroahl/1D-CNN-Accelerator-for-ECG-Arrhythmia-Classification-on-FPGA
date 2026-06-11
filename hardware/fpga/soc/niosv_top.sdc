# =============================================================================
# SDC — Phase D Nios V variant (on-board, no HPS)
# Top      : niosv_top  (Qsys nios_system: Nios V/m + RAM + JTAG UART + ecg_core + PLL)
# Device   : Intel Cyclone V (DE10-Standard, 5CSXFC6D6F31C6, speed grade -6)
# =============================================================================
# Same constraints as jtag_top.sdc: 50 MHz board oscillator, PLL-derived 100 MHz
# core clock, async reset key. The Nios V JTAG debug TAP brings its own internal
# SLD clock constraints — nothing to add here for it.
# =============================================================================

# 1. Board oscillator — 50 MHz into the PLL
create_clock -name FPGA_CLK1_50 -period 20.000 [get_ports {FPGA_CLK1_50}]

# 2. Derive PLL output clocks (core_pll: 50 MHz -> 100 MHz core_clk)
derive_pll_clocks
derive_clock_uncertainty

# 3. Async reset push-button — false path (synchronized in niosv_top)
set_false_path -from [get_ports {KEY0_n}]
