# =============================================================================
# Synopsys Design Constraints (SDC) — DE0-Nano port
# Project  : CNN Accelerator — ECG Arrhythmia Classification
# Top      : ecg_accelerator_top
# Device   : Intel Cyclone IV E (DE0-Nano, EP4CE22F17C6, speed grade -6)
# Target   : 50 MHz (= DE0-Nano on-board CLOCK_50 oscillator, PIN_R8)
# =============================================================================
#
# The DE0-Nano supplies a single 50 MHz oscillator (CLOCK_50 -> PIN_R8). The
# accelerator is clocked directly from it (no PLL), so the design clock IS 50 MHz.
# Cyclone IV E -6 is ~20-30% slower than the Cyclone V -6 used on the DE10, but
# the worst datapath observed at 100 MHz on Cyclone V is ~15.3 ns, leaving ample
# slack at the 20 ns period here. Inference still completes in 5216 cycles =
# ~104 us @ 50 MHz, four orders of magnitude faster than a heartbeat (~700 ms).
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Primary Clock — 50 MHz (period = 20.000 ns)
# -----------------------------------------------------------------------------
create_clock -name clk -period 20.000 [get_ports clk]

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
