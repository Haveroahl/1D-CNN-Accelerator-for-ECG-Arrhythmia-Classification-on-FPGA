# =============================================================================
# Synopsys Design Constraints (SDC)
# Project  : CNN Accelerator — ECG Arrhythmia Classification
# Top      : ecg_accelerator_top
# Device   : Intel Cyclone V (DE10-Standard, 5CSXFC6D6F31C6, speed grade -6)
# Target   : 150 MHz (experimental — push above default 100 MHz baseline)
# =============================================================================
#
# Critical-path analysis (see System_Design.md "Critical Path Analysis"):
#   S6 RESCALE  (32b add + barrel shift)        — borderline at 6.67 ns
#   S5b acc_final + S_bias (32b adds)           — borderline
#   FC accumulator (32b sext add)               — moderate
#   S7 clamp, weight ROM MUX, GAP add, SRAM     — comfortable
#
# Goal: verify TimeQuest Fmax. If WNS < 0 (negative slack) → either fall back to
# 100 MHz or add pipeline stage after S6 (split add + shift across 2 registers).
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Primary Clock — 150 MHz (period = 6.667 ns)
# -----------------------------------------------------------------------------
create_clock -name clk -period 6.667 [get_ports clk]

# Clock uncertainty: jitter + skew budget (~7% of period for higher freq)
# Tighter than 100 MHz design (where 0.300 ns was 3%) to account for tighter PLL
# jitter contribution as freq scales up.
set_clock_uncertainty -setup -to [get_clocks clk] 0.450
set_clock_uncertainty -hold  -to [get_clocks clk] 0.100

# -----------------------------------------------------------------------------
# 2. Avalon-MM Slave Input Delays
# -----------------------------------------------------------------------------
# HPS Lightweight bridge → FPGA fabric. Conservative max input delay assumes
# ~1.5 ns FF + interconnect on HPS side.
# Inputs from HPS LW bridge (synchronous to clk):
#   avs_address[4:0], avs_write, avs_read, avs_writedata[31:0]
set_input_delay -clock clk -max 1.500 \
    [get_ports {avs_address[*] avs_write avs_read avs_writedata[*]}]
set_input_delay -clock clk -min 0.200 \
    [get_ports {avs_address[*] avs_write avs_read avs_writedata[*]}]

# -----------------------------------------------------------------------------
# 3. Avalon-MM Slave Output Delays
# -----------------------------------------------------------------------------
# avs_readdata returns to HPS LW bridge. Allow ~1.5 ns for downstream FF setup.
set_output_delay -clock clk -max 1.500 [get_ports {avs_readdata[*]}]
set_output_delay -clock clk -min 0.000 [get_ports {avs_readdata[*]}]

# -----------------------------------------------------------------------------
# 4. Reset Paths — async, false path
# -----------------------------------------------------------------------------
# rst_n: async active-low reset from HPS (used by avalon_slave).
# rst:   synchronous reset (active high) — internally synchronized.
# Both should be excluded from setup/hold STA — assume designer added
# reset synchronizers (1-2 FF) in downstream RTL.
set_false_path -from [get_ports rst_n]
set_false_path -from [get_ports rst]

# -----------------------------------------------------------------------------
# 5. Multicycle Paths
# -----------------------------------------------------------------------------
# w_packed register (cp_engine.v:237-245) is updated every cycle but only
# READ during compute (after 5-cycle prefetch latency). At 150 MHz the
# combinational w_comb MUX (4-way layer + 8-way ic) is short (~2-3 LUT
# levels, < 2 ns) — no multicycle exception needed.
#
# Bias b_cur (cp_engine.v:259-262) registered same cycle as use — no MC.
#
# If post-synth report shows w_comb → w_packed as critical, consider:
#   set_multicycle_path -setup 2 -from [get_pins -of [get_registers \
#       {u_top|u_cpe|w_rom_conv*[*]*}]] -to [get_pins -of [get_registers \
#       {u_top|u_cpe|w_packed[*][*]}]]

# -----------------------------------------------------------------------------
# 6. Memory Timing (M10K Block RAMs)
# -----------------------------------------------------------------------------
# input_sram (input_sram.v) and ping_pong_sram (ping_pong_sram.v) infer M10K
# blocks with 1-cycle synchronous read latency. Cyclone V M10K -6 supports
# > 300 MHz typical → comfortable at 150 MHz. No constraints needed.

# -----------------------------------------------------------------------------
# 7. Generated Clocks (PLL)
# -----------------------------------------------------------------------------
# If a PLL is instantiated to derive 150 MHz from the DE10-Standard 50 MHz
# reference oscillator, add:
#   create_generated_clock -name clk_150 \
#       -source [get_pins u_pll|altpll_component|auto_generated|pll1|inclk[0]] \
#       -multiply_by 3 [get_pins u_pll|altpll_component|auto_generated|pll1|clk[0]]
# Currently `clk` is treated as primary — adjust when PLL is added.

# -----------------------------------------------------------------------------
# 8. Critical Path Watchlist (for TimeQuest report-timing review)
# -----------------------------------------------------------------------------
# Run after fitting:
#   report_timing -setup -npaths 20 -detail full_path -panel "Worst setup"
#   report_clock_fmax_summary
#
# Watch:
#   - cp_block.v:118  biased <= acc_final_r + bias_in        (S_bias add)
#   - cp_block.v:136  shifted <= (biased + round_add) >>> nb (S6 RESCALE)
#   - cp_block.v:104  acc_final_r <= (acc + tree_sext)       (S5b cascaded add)
#   - gap_fc_argmax.v:168  fc_acc += sext(fc_prod)           (FC 32b add)
#
# Expected Fmax for current architecture:
#   - Speed grade -6 (DE10-Standard): ~150-180 MHz achievable
#   - Speed grade -7 (slower parts) : ~120-150 MHz
#
# If WNS < 0:
#   1. Try -O speed in Quartus Compiler settings.
#   2. Enable "Optimize timing" and "Optimize register-to-register".
#   3. Insert pipeline reg between (biased + round_add) and (>>> nb).
#   4. Fall back to clk period = 8.0 ns (125 MHz) or 10.0 ns (100 MHz).

# End of SDC
