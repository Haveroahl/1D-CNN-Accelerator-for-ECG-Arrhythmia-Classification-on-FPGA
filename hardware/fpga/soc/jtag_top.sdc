# =============================================================================
# Synopsys Design Constraints (SDC) — Phase D JTAG variant (on-board, no HPS)
# Top      : jtag_top  (Qsys jtag_system: JTAG master + interconnect + ecg_core + PLL)
# Device   : Intel Cyclone V (DE10-Standard, 5CSXFC6D6F31C6, speed grade -6)
# =============================================================================
#
# Same idea as soc_top.sdc but simpler: no HPS, no DDR3. Only the 50 MHz board
# oscillator, the PLL-derived 100 MHz core clock, and the async reset key.
#
# The JTAG master runs on the dedicated JTAG TAP clock (altera_jtag_avalon_master
# brings its own internal SLD clock constraints) — nothing to add here for it.
# =============================================================================

# 1. Board oscillator — 50 MHz into the PLL
create_clock -name FPGA_CLK1_50 -period 20.000 [get_ports {FPGA_CLK1_50}]

# 2. Derive PLL output clocks (core_pll: 50 MHz -> 100 MHz core_clk)
derive_pll_clocks
derive_clock_uncertainty

# 3. Async reset push-button — false path (synchronized in jtag_top)
set_false_path -from [get_ports {KEY0_n}]
