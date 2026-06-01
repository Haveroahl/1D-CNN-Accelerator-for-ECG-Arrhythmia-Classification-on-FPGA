# =============================================================================
# Synopsys Design Constraints (SDC)
# Project  : CNN Accelerator — ECG Arrhythmia Classification
# Top      : ecg_accelerator_top
# Device   : Intel Cyclone V (DE10-Standard, 5CSXFC6D6F31C6, speed grade -6)
# Target   : 100 MHz (standard target)
# =============================================================================
#
# Standard design target per System_Design.md.
# After ping_pong_sram refactor (16 M10K inference), worst internal path:
#   cnn_controller.a[*] -> cp_engine.mux_s1[*]   ~10.7 ns
# Slack at 100 MHz ≈ -0.7 ns (borderline) — enable Quartus Performance High
# Effort + register retiming to close timing without RTL changes.
# Fallback: ecg_accelerator_top_100mhz.sdc.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Primary Clock — 100 MHz (period = 10.000 ns)
# -----------------------------------------------------------------------------
create_clock -name clk -period 10.000 [get_ports clk]

# Clock uncertainty: jitter + skew budget (~3% of period)
set_clock_uncertainty -setup -to [get_clocks clk] 0.300
set_clock_uncertainty -hold  -to [get_clocks clk] 0.100

# -----------------------------------------------------------------------------
# 2. Avalon-MM Slave Input Delays
# -----------------------------------------------------------------------------
set_input_delay -clock clk -max 1.500 \
    [get_ports {avs_address[*] avs_write avs_read avs_writedata[*]}]
set_input_delay -clock clk -min 0.200 \
    [get_ports {avs_address[*] avs_write avs_read avs_writedata[*]}]

# -----------------------------------------------------------------------------
# 3. Avalon-MM Slave Output Delays
# -----------------------------------------------------------------------------
set_output_delay -clock clk -max 1.500 [get_ports {avs_readdata[*]}]
set_output_delay -clock clk -min 0.000 [get_ports {avs_readdata[*]}]

# -----------------------------------------------------------------------------
# 4. Reset Paths — async, false path
# -----------------------------------------------------------------------------
set_false_path -from [get_ports rst_n]
set_false_path -from [get_ports rst]

# -----------------------------------------------------------------------------
# 5. Multicycle Paths
# -----------------------------------------------------------------------------
# Not required — Quartus retiming handles internal balance.

# -----------------------------------------------------------------------------
# 6. Generated Clocks (PLL)
# -----------------------------------------------------------------------------
# If PLL derives clk from 50 MHz reference, add:
#   create_generated_clock -name clk_100 \
#       -source [get_pins u_pll|altpll_component|auto_generated|pll1|inclk[0]] \
#       -multiply_by 2 [get_pins u_pll|altpll_component|auto_generated|pll1|clk[0]]

# End of SDC
