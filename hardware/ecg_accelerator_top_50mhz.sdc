# =============================================================================
# Synopsys Design Constraints (SDC)
# Project  : CNN Accelerator — ECG Arrhythmia Classification
# Top      : ecg_accelerator_top
# Device   : Intel Cyclone V (DE10-Standard, 5CSXFC6D6F31C6, speed grade -6)
# Target   : 50 MHz (relaxed / debug target)
# =============================================================================
#
# Use this SDC when 100 MHz fails timing closure (e.g. ping_pong_sram not
# inferred as M10K → register-file implementation → long mux delays).
# 50 MHz = 20 ns period gives ~13 ns slack on the worst path observed at
# 100 MHz (data path 15.3 ns vs new budget 20 ns). Inference @ 50 MHz takes
# 2× longer (~52 µs vs ~26 µs per inference) but still well within real-time
# ECG requirements (single beat = ~700 ms).
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Primary Clock — 50 MHz (period = 20.000 ns)
# -----------------------------------------------------------------------------
create_clock -name clk -period 20.000 [get_ports clk]

# Clock uncertainty: jitter + skew budget (~2% of period — relaxed)
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

# -----------------------------------------------------------------------------
# 5. Multicycle Paths
# -----------------------------------------------------------------------------
# Not required at 50 MHz — comfortable margin on all paths.

# -----------------------------------------------------------------------------
# 6. Generated Clocks (PLL)
# -----------------------------------------------------------------------------
# If PLL derives clk from 50 MHz reference (1:1):
#   create_generated_clock -name clk_50 \
#       -source [get_pins u_pll|altpll_component|auto_generated|pll1|inclk[0]] \
#       -multiply_by 1 [get_pins u_pll|altpll_component|auto_generated|pll1|clk[0]]

# End of SDC
