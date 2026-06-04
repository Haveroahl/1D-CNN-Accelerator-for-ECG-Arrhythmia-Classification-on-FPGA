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

package require -exact qsys 21.1

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
add_fileset          QUARTUS_SYNTH  QUARTUS_SYNTH  generate_synth
set_fileset_property QUARTUS_SYNTH  TOP_LEVEL ecg_accelerator_top

proc generate_synth { entity } {
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
    # the generated synth files; Quartus then finds them on the search path.
    add_fileset_file conv1_w.hex   OTHER PATH "$rtl/conv1_w.hex"
    add_fileset_file conv2_w.hex   OTHER PATH "$rtl/conv2_w.hex"
    add_fileset_file conv3_w.hex   OTHER PATH "$rtl/conv3_w.hex"
    add_fileset_file conv4_w.hex   OTHER PATH "$rtl/conv4_w.hex"
    add_fileset_file conv_bias.hex OTHER PATH "$rtl/conv_bias.hex"
    # fc_weights.hex: include if gap_fc_argmax $readmemh's it (check your RTL).
    add_fileset_file fc_weights.hex OTHER PATH "$rtl/fc_weights.hex"
}

# ── Clock interface ────────────────────────────────────────────────────────
add_interface           clk clock end
add_interface_port      clk clk clk Input 1

# ── Reset interface (async active-low) → drives core .rst_n ─────────────────
add_interface           reset_n reset end
set_interface_property  reset_n associatedClock clk
set_interface_property  reset_n synchronousEdges NONE
add_interface_port      reset_n rst_n reset_n Input 1

# ── Sync active-high reset as a conduit → drives core .rst ──────────────────
add_interface           reset_h conduit end
add_interface_port      reset_h rst reset Input 1
set_interface_property  reset_h associatedClock clk

# ── Avalon-MM slave ─────────────────────────────────────────────────────────
# 6 word registers (5-bit address). Read latency = 1 cycle (avs_readdata is
# registered in avalon_slave). No waitrequest / no burst — simple slave.
add_interface           avs avalon end
set_interface_property  avs associatedClock      clk
set_interface_property  avs associatedReset      reset_n
set_interface_property  avs addressUnits         WORDS
set_interface_property  avs readLatency          1
set_interface_property  avs maximumPendingReadTransactions 0
set_interface_property  avs explicitAddressSpan  0

add_interface_port      avs avs_address   address    Input  5
add_interface_port      avs avs_write     write      Input  1
add_interface_port      avs avs_read      read       Input  1
add_interface_port      avs avs_writedata writedata  Input  32
add_interface_port      avs avs_readdata  readdata   Output 32

proc elaborate {} {
    # Address span = 2^5 words = 32 words (6 used: 0x00..0x05).
    set_interface_property avs explicitAddressSpan 32
}
