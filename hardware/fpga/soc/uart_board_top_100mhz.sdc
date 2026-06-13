# =============================================================================
# SDC — UART variant, 100 MHz SANITY constraint (NOT board-runnable)
# Top      : uart_board_top
# Device   : Intel Cyclone V (DE10-Standard, 5CSXFC6D6F31C6, speed grade -6)
# =============================================================================
# Forces a 10 ns (100 MHz) period directly onto the 50 MHz board oscillator
# port. This is a TIMING SANITY check only — the physical osc is 50 MHz, so a
# real board run at 100 MHz would need a PLL (50->100). Use this SDC to see
# whether the core path closes at 100 MHz without adding the PLL.
#
# For an actual 100 MHz board build, add core_pll.qip (50->100) and constrain
# the PLL output, like the niosv variant does.
# =============================================================================

# Oscillator port driven at 100 MHz (sanity — real osc is 50 MHz)
create_clock -name FPGA_CLK1_50 -period 10.000 [get_ports {FPGA_CLK1_50}]

derive_clock_uncertainty

set_false_path -from [get_ports {KEY0_n}]
set_false_path -from [get_ports {UART_RXD}]
set_false_path -to   [get_ports {UART_TXD}]
