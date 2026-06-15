# ecg_core_hw.tcl
# ============================================================================
# Platform Designer (Qsys) component definition for the ECG accelerator.
#
# Wraps RTL/ecg_accelerator_top.v as a Qsys component named "ecg_core" and
# declares its port groups:
#   - clk            : clock input
#   - reset_n        : async active-low reset (drives core .rst_n)
#   - reset_h        : sync  active-high reset (drives core .rst)   [conduit]
#   - avs (avalon)   : the 6-register Avalon-MM slave (avs_*)
#
# HOW TO USE:
#   1. Quartus → Tools → Platform Designer.
#   2. (one-time) Tools → Options → IP Search Path → add this folder, or just
#      keep this .tcl next to the project so Qsys auto-discovers it.
#   3. The component "ecg_core" then appears in the IP Catalog → drop it in.
#
# WHY a conduit for reset_h:
#   The core takes TWO resets (rst_n async-low + rst sync-high). Qsys reset
#   interfaces model ONE polarity each. We expose rst_n as a proper reset sink
#   and rst as a conduit driven from soc_top's reset-synchronizer. Cleaner than
#   forcing both into Qsys reset bridges.
# ============================================================================

package require qsys

set_module_property NAME                         ecg_core
set_module_property DISPLAY_NAME                 "ECG CNN Accelerator"
set_module_property VERSION                      1.0
set_module_property GROUP                        "Custom/ECG"
set_module_property DESCRIPTION                  "INT8 1D-CNN ECG arrhythmia classifier, Avalon-MM slave"
set_module_property AUTHOR                       "Le Duc"
set_module_property EDITABLE                     false
set_module_property ELABORATION_CALLBACK         elaborate

# ── HDL files ────────────────────────────────────────────────────────────
# Top of the component is ecg_accelerator_top; pull in every sub-module.
# Paths are relative to this .tcl (soc/ → ../../RTL).
#
# Two filesets share the SAME Verilog RTL + .hex:
#   QUARTUS_SYNTH — for Quartus compile (onboard .sof)
#   SIM_VERILOG   — for ModelSim/Questa system simulation (Nios V drives the core)
# The RTL is plain Verilog (synthesisable + simulatable), so both reuse one proc.
add_fileset          QUARTUS_SYNTH  QUARTUS_SYNTH  add_ecg_files
set_fileset_property QUARTUS_SYNTH  TOP_LEVEL ecg_accelerator_top
add_fileset          SIM_VERILOG    SIM_VERILOG    add_ecg_files
set_fileset_property SIM_VERILOG    TOP_LEVEL ecg_accelerator_top

proc add_ecg_files { entity } {
    set rtl ../../RTL
    add_fileset_file ecg_accelerator_top.v VERILOG PATH "$rtl/ecg_accelerator_top.v" TOP_LEVEL_FILE
    add_fileset_file ecg_core.v            VERILOG PATH "$rtl/ecg_core.v"
    add_fileset_file avalon_slave.v        VERILOG PATH "$rtl/avalon_slave.v"
    add_fileset_file input_sram.v          VERILOG PATH "$rtl/input_sram.v"
    add_fileset_file ping_pong_sram.v      VERILOG PATH "$rtl/ping_pong_sram.v"
    add_fileset_file cp_engine.v           VERILOG PATH "$rtl/cp_engine.v"
    add_fileset_file cp_block.v            VERILOG PATH "$rtl/cp_block.v"
    add_fileset_file cnn_controller.v      VERILOG PATH "$rtl/cnn_controller.v"
    add_fileset_file gap_fc_argmax.v       VERILOG PATH "$rtl/gap_fc_argmax.v"
    # Weight .hex are read by $readmemh in cp_engine. Qsys copies them next to
    # the generated synth/sim files; the tool then finds them on the search path.
    # Phase B01: cp_engine reads the 8 per-oc w_ram*.hex (M10K init); conv*_w.hex
    # are no longer $readmemh'd by cp_engine but kept for the jtag_system submodule
    # search path / legacy sim. conv_bias.hex is still read by cp_engine.
    add_fileset_file w_ram0.hex    OTHER PATH "$rtl/w_ram0.hex"
    add_fileset_file w_ram1.hex    OTHER PATH "$rtl/w_ram1.hex"
    add_fileset_file w_ram2.hex    OTHER PATH "$rtl/w_ram2.hex"
    add_fileset_file w_ram3.hex    OTHER PATH "$rtl/w_ram3.hex"
    add_fileset_file w_ram4.hex    OTHER PATH "$rtl/w_ram4.hex"
    add_fileset_file w_ram5.hex    OTHER PATH "$rtl/w_ram5.hex"
    add_fileset_file w_ram6.hex    OTHER PATH "$rtl/w_ram6.hex"
    add_fileset_file w_ram7.hex    OTHER PATH "$rtl/w_ram7.hex"
    add_fileset_file conv_bias.hex OTHER PATH "$rtl/conv_bias.hex"
    # gap_fc_argmax $readmemh's BOTH fc_weights.hex and fc_bias.hex (see
    # gap_fc_argmax.v:52-53). Both must be in the fileset so Qsys copies them
    # next to the generated submodules; otherwise synthesis fails with
    # "can't open Verilog Design File fc_bias.hex".
    add_fileset_file fc_weights.hex OTHER PATH "$rtl/fc_weights.hex"
    add_fileset_file fc_bias.hex    OTHER PATH "$rtl/fc_bias.hex"
}

# ── Clock interface ────────────────────────────────────────────────────────
add_interface           clk clock end
add_interface_port      clk clk clk Input 1

# ── Reset interface (async active-low) → drives core .rst_n ─────────────────
# associatedClock left empty: this reset is fully asynchronous (synchronousEdges
# NONE). Pairing it with a clock makes Qsys warn "No synchronous edges, but has
# associated clock" — harmless, but cleaner to leave the clock association off.
add_interface           reset_n reset end
set_interface_property  reset_n associatedClock ""
set_interface_property  reset_n synchronousEdges NONE
add_interface_port      reset_n rst_n reset_n Input 1

# ── Sync active-high reset as a conduit → drives core .rst ──────────────────
add_interface           reset_h conduit end
add_interface_port      reset_h rst reset Input 1
set_interface_property  reset_h associatedClock clk

# ── Avalon-MM slave ─────────────────────────────────────────────────────────
# 14-bit WORD address. Read latency = 1 cycle (avs_readdata is registered in
# avalon_slave). No waitrequest / no burst — simple slave.
#   Low registers 0x0000..0x0005 (unchanged 6-register map).
#   DATA WINDOW   0x1000..0x19C3 (addr[12]=1): one word = one SRAM byte, so a
#                 single System-Console block write loads a whole 2500-byte
#                 sample (replaces the ~7500 per-byte JTAG transactions).
#   WEIGHT WINDOW 0x2000..0x3FFF (addr[13]=1, Phase B01 runtime weight reload):
#                 conv weight / conv bias / FC w+bias regions (see avalon_slave.v).
add_interface           avs avalon end
set_interface_property  avs associatedClock      clk
set_interface_property  avs associatedReset      reset_n
set_interface_property  avs addressUnits         WORDS
set_interface_property  avs readLatency          1
set_interface_property  avs maximumPendingReadTransactions 0
set_interface_property  avs explicitAddressSpan  0

add_interface_port      avs avs_address   address    Input  14
add_interface_port      avs avs_write     write      Input  1
add_interface_port      avs avs_read      read       Input  1
add_interface_port      avs avs_writedata writedata  Input  32
add_interface_port      avs avs_readdata  readdata   Output 32

proc elaborate {} {
    # Address span = 2^14 words = 16384 words. Low regs 0x0..0x5, ECG window
    # 0x1000..0x19C3, weight window 0x2000..0x3FFF. 16384 covers all.
    set_interface_property avs explicitAddressSpan 16384
}
