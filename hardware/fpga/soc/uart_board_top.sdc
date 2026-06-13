# =============================================================================
# SDC — UART variant (on-board, no HPS, no Nios V)
# Top      : uart_board_top  (50 MHz direct → ecg_uart_top → ecg_core)
# Device   : Intel Cyclone V (DE10-Standard, 5CSXFC6D6F31C6, speed grade -6)
# =============================================================================
# No PLL — the core runs directly off the 50 MHz board oscillator (Fmax ~105 MHz,
# so 50 MHz has large slack). UART RX/TX are slow async serial lines (115200 baud)
# unrelated to the core timing; constrain them as false paths.
# =============================================================================

# 1. Board oscillator — 50 MHz, also the core clock
create_clock -name FPGA_CLK1_50 -period 20.000 [get_ports {FPGA_CLK1_50}]

derive_clock_uncertainty

# 2. Async reset push-button — false path (synchronized in uart_board_top)
set_false_path -from [get_ports {KEY0_n}]

# 3. UART serial pins — async to the core clock, baud-rate slow; false-path both.
#    uart_rx is double-flopped inside uart_wrapper; uart_tx is a slow registered out.
set_false_path -from [get_ports {UART_RXD}]
set_false_path -to   [get_ports {UART_TXD}]
