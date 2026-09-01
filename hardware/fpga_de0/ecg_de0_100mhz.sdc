# =============================================================================
# Synopsys Design Constraints (SDC) — DE0-Nano, 100 MHz TIMING-CHECK target
# Project  : CNN Accelerator — ECG Arrhythmia Classification
# Top      : ecg_accelerator_top
# Device   : Intel Cyclone IV E (DE0-Nano, EP4CE22F17C6, speed grade -6)
# Target   : 100 MHz — used ONLY to answer "does the design close timing at
#            100 MHz on this slower 60 nm fabric?". The on-board demo runs at
#            50 MHz (ecg_de0_50mhz.sdc) from the board's CLOCK_50 oscillator.
# =============================================================================
# Note: the DE0-Nano has no 100 MHz source. This SDC constrains `clk` to 100 MHz
# so the fitter optimizes for and reports timing at that period. To actually run
# at 100 MHz on hardware you would add a PLL (50 -> 100 MHz); not needed here
# since this is a timing-feasibility measurement, not the demo configuration.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Primary Clock — 100 MHz (period = 10.000 ns)
# -----------------------------------------------------------------------------
create_clock -name clk -period 10.000 [get_ports clk]

# Clock uncertainty: jitter + skew budget
set_clock_uncertainty -setup -to [get_clocks clk] 0.300
set_clock_uncertainty -hold  -to [get_clocks clk] 0.100

# -----------------------------------------------------------------------------
# 2. Avalon-MM Slave Input Delays
# -----------------------------------------------------------------------------
set_input_delay -clock clk -max 3.000 \
    [get_ports {avs_address[*] avs_write avs_read avs_writedata[*]}]
set_input_delay -clock clk -min 0.200 \
    [get_ports {avs_address[*] avs_write avs_read avs_writedata[*]}]

# -----------------------------------------------------------------------------
# 3. Avalon-MM Slave Output Delays
# -----------------------------------------------------------------------------
set_output_delay -clock clk -max 3.000 [get_ports {avs_readdata[*]}]
set_output_delay -clock clk -min 0.000 [get_ports {avs_readdata[*]}]

# -----------------------------------------------------------------------------
# 4. Reset Paths — async, false path
# -----------------------------------------------------------------------------
set_false_path -from [get_ports rst_n]
set_false_path -from [get_ports rst]

# End of SDC
