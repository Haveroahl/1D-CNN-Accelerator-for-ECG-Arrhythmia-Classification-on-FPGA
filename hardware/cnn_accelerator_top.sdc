# =============================================================================
# Synopsys Design Constraints (SDC)
# Project  : CNN Accelerator — ECG Arrhythmia Classification
# Device   : Intel Cyclone V (DE10-Nano, 5CSEBA6U23I7)
# Target   : 100 MHz
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Clock Definition
# -----------------------------------------------------------------------------
# System clock — 100 MHz (period = 10.000 ns)
create_clock -name clk -period 10.000 [get_ports clk]

# Clock uncertainty: jitter + skew budget (~5% of period)
set_clock_uncertainty -setup -to [get_clocks clk] 0.300
set_clock_uncertainty -hold  -to [get_clocks clk] 0.100

# -----------------------------------------------------------------------------
# 3. Input Delays
# Avalon-ST source assumed to be in the same clock domain.
# Input setup/hold referenced to clk rising edge.
# Assumed PCB + source FF delay: 1.0 ns max, 0.2 ns min.
# -----------------------------------------------------------------------------
set_input_delay -clock clk -max 1.000 [get_ports {asi_data[*] asi_valid aso_ready rst_n}]
set_input_delay -clock clk -min 0.200 [get_ports {asi_data[*] asi_valid aso_ready rst_n}]

# -----------------------------------------------------------------------------
# 4. False Paths
# rst_n: asynchronous reset — no timing analysis needed.
# Output ports (aso_data, aso_valid, asi_ready): connect to Qsys interconnect
# internally; output pad delay (~2.4 ns) is not representative of real timing.
# Exclude entirely from STA via false path — no set_output_delay needed.
# -----------------------------------------------------------------------------
set_false_path -from [get_ports rst_n]
set_false_path -to   [get_ports {aso_data[*] aso_valid asi_ready}]

# -----------------------------------------------------------------------------
# 5. Multicycle Paths
# At 100 MHz (10 ns period), all internal paths have sufficient slack.
# No multicycle exceptions required.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 7. Derived/Generated Clocks
# No PLLs or clock dividers in this design — single clock domain.
# If a PLL is added in Qsys, add:
#   create_generated_clock -name clk_150 -source [get_pins u_pll|inclk[0]] \
#       -multiply_by 3 -divide_by 1 [get_pins u_pll|outclk[0]]
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 8. Operating Conditions (Cyclone V, slow corner for setup, fast for hold)
# -----------------------------------------------------------------------------
# Quartus applies worst-case automatically; no explicit set_operating_conditions
# needed for Cyclone V in TimeQuest.

# -----------------------------------------------------------------------------
# 9. Design-Specific Timing Exceptions
#
# w_addr_r path: FF(in_ch_out) → carry-chain(*5) → FF(w_addr_r) is a standard
# single-cycle path; no exception needed.
#
# weight_buf path: FF(w_addr_r) → 30-entry per-PE LUTRAM → FF(weight_buf).
# 30-entry LUTRAM has 5-bit address — MUX tree is 2-3 levels (~2-3 ns).
# Path budget: ~4-5 ns total. No exception needed; verify slack >= 0.
#
# shared_pixel_window: now registered (FF-to-FF path). No exception needed.
# -----------------------------------------------------------------------------

# End of SDC
