# ============================================================================
# ecg_jtag_console.tcl  —  System Console driver for the ECG accelerator
#                          over the JTAG-to-Avalon Master Bridge (Phase D, no HPS)
#
# Replaces the HPS C driver (ecg_classify.c) for the LITE-edition / JTAG flow.
# Drives the SAME verified avalon_slave register map from the PC over JTAG.
#
# USAGE (from a shell, board programmed with jtag_top.sof):
#   system-console --script=ecg_jtag_console.tcl
#       -> runs the whole Chapman test set, prints accuracy.
#   To run a single sample only, set ::MAX_SAMPLES below.
#
# Register map (avalon_slave.v), WORD address -> BYTE address (master_* use bytes):
#   word 0x00 (byte 0x00) W : sram_din      [7:0]   one ECG byte
#   word 0x01 (byte 0x04) W : sram_wr_addr  [11:0]  SRAM address
#   word 0x02 (byte 0x08) W : sram_we       [0]     write-enable pulse
#   word 0x03 (byte 0x0C) W : start         [0]     start + clear done
#   word 0x04 (byte 0x10) R : status        {done_latched, busy}
#   word 0x05 (byte 0x14) R : result        [1:0]   predicted class
# ============================================================================

# ---- byte addresses ----
set A_DIN   0x00
set A_ADDR  0x04
set A_WE    0x08
set A_START 0x0C
set A_STAT  0x10
set A_RES   0x14

# ---- config ----
set ::ECG_FILE    "demo_data/chapman_test_ecg_int8.bin"
set ::LBL_FILE    "demo_data/chapman_test_labels.bin"
set ::SAMPLE_LEN  2500
set ::MAX_SAMPLES 0        ;# 0 = all samples; set e.g. 3 for a quick check

# ----------------------------------------------------------------------------
# Open the JTAG master service.
# ----------------------------------------------------------------------------
proc open_master {} {
    set masters [get_service_paths master]
    if {[llength $masters] == 0} {
        error "No JTAG Avalon master found. Is the board programmed with jtag_top.sof and USB-Blaster connected?"
    }
    set m [lindex $masters 0]
    set claimed [claim_service master $m ""]
    puts "Claimed master: $m"
    return $claimed
}

# ----------------------------------------------------------------------------
# Load one 2500-byte ECG sample into the input SRAM.
#   For each byte: write din, write addr, pulse we=1.
#   (we self-clears in HW the cycle after, but we write 1 then move on — the
#    slave registers sram_we for exactly one clock, matching tb_top behavior.)
# ----------------------------------------------------------------------------
proc load_ecg {m bytes} {
    global A_DIN A_ADDR A_WE
    set i 0
    foreach b $bytes {
        # int8 -> unsigned byte for the 8-bit din field
        set ub [expr {$b & 0xFF}]
        master_write_32 $m $A_DIN  $ub
        master_write_32 $m $A_ADDR $i
        master_write_32 $m $A_WE   1
        incr i
    }
}

# ----------------------------------------------------------------------------
# Pulse start, poll done, return predicted class.
# ----------------------------------------------------------------------------
proc run_inference {m} {
    global A_START A_STAT A_RES
    master_write_32 $m $A_START 1
    # poll done_latched (bit 1 of status)
    set tries 0
    while {1} {
        set st [master_read_32 $m $A_STAT 1]
        set st [expr {[lindex $st 0]}]
        if {($st & 0x2) != 0} break
        incr tries
        if {$tries > 100000} { error "Timeout waiting for done" }
    }
    set res [master_read_32 $m $A_RES 1]
    return [expr {[lindex $res 0] & 0x3}]
}

# ----------------------------------------------------------------------------
# Read a binary file as a list of signed/unsigned bytes.
# ----------------------------------------------------------------------------
proc read_bytes {path} {
    set f [open $path rb]
    set data [read $f]
    close $f
    binary scan $data c* signed   ;# signed int8 list
    return $signed
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
proc main {} {
    global A_RES
    set m [open_master]

    set ecg_all [read_bytes $::ECG_FILE]
    set lbl_all [read_bytes $::LBL_FILE]
    set n_total [expr {[llength $ecg_all] / $::SAMPLE_LEN}]
    set n_run   $n_total
    if {$::MAX_SAMPLES > 0 && $::MAX_SAMPLES < $n_run} { set n_run $::MAX_SAMPLES }

    puts "Total samples in file: $n_total ; running: $n_run"
    set correct 0
    for {set s 0} {$s < $n_run} {incr s} {
        set off [expr {$s * $::SAMPLE_LEN}]
        set sample [lrange $ecg_all $off [expr {$off + $::SAMPLE_LEN - 1}]]
        load_ecg $m $sample
        set pred [run_inference $m]
        set truth [expr {[lindex $lbl_all $s] & 0xFF}]
        if {$pred == $truth} { incr correct }
        if {$n_run <= 10 || ($s % 50) == 0} {
            puts [format "sample %4d : pred=%d truth=%d %s" \
                  $s $pred $truth [expr {$pred==$truth ? "OK" : "X"}]]
        }
    }
    set acc [expr {100.0 * $correct / $n_run}]
    puts "----------------------------------------------"
    puts [format "Accuracy: %d/%d = %.2f%%" $correct $n_run $acc]
    close_service master $m
}

main
