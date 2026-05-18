# =============================================================================
# Synopsys Design Constraints (SDC)
# Project  : CNN Accelerator — ECG Arrhythmia Classification
# Top      : ecg_accelerator_top
# Device   : Intel Cyclone V (DE10-Standard, 5CSXFC6D6F31C6, speed grade -6)
# Target   : 100 MHz (standard / fallback target)
# =============================================================================
#
# Standard design target per System_Design.md. Use this SDC if the 150 MHz
# experimental constraint (ecg_accelerator_top_150mhz.sdc) fails timing closure.
#
# All critical paths (S6 RESCALE, S5b ACC_FINAL, S_bias, FC ACC) have ~3 ns
# slack at 100 MHz — comfortable, no risk.
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
# At 100 MHz all internal paths have ~3 ns slack. No exceptions required.

# -----------------------------------------------------------------------------
# 6. Generated Clocks (PLL)
# -----------------------------------------------------------------------------
# If PLL derives clk from 50 MHz reference, add:
#   create_generated_clock -name clk_100 \
#       -source [get_pins u_pll|altpll_component|auto_generated|pll1|inclk[0]] \
#       -multiply_by 2 [get_pins u_pll|altpll_component|auto_generated|pll1|clk[0]]

# End of SDC
