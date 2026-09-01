# ============================================================================
# ecg_jtag_rom.tcl — System Console driver for the ROM single-load build
#                    (hardware/RTL/, thesis variant) over JTAG-to-Avalon.
#
# Difference vs ecg_jtag_console.tcl (weight-load build, hardware/RTL_weight/):
#   - NO load_weights   : weights are baked into the bitstream ($readmemh ROM).
#   - NO load_topology  : topology is hard-coded in cnn_controller (1,4,4,8;
#                         nb = 8,7,6,7). There is no CONFIG window in this slave.
#   - NO overlap reload : this slave exposes only {done_latched, busy}; there is
#                         no isram_free bit, so each sample runs strictly
#                         load -> start -> wait done -> read result.
# Keep BOTH scripts: this one matches jtag_top.sof built from hardware/RTL/.
#
# USAGE (board programmed with output_files/jtag_top.sof, USB-Blaster attached):
#   cd d:\Thesis101\hardware\fpga\soc
#   <quartus>\bin64\system-console --script=ecg_jtag_rom.tcl
#
# Writes a timestamped log next to this script: ecg_rom_<date>_<time>.log
#
# Register map (RTL/avalon_slave.v), WORD address -> BYTE address (master_* use bytes):
#   word 0x0003 (byte 0x000C) W : start   [0]  start + clear done_latched
#   word 0x0004 (byte 0x0010) R : status  bit0=busy, bit1=done_latched
#   word 0x0005 (byte 0x0014) R : result  [1:0] predicted class 0..3
#   DATA WINDOW word 0x1000..0x19C3 (byte 0x4000..0x6710) W : one word per SRAM
#       byte; the slave drives din+addr+we from the word address. A 2500-byte
#       sample ships in ONE master_write_32 block call.
# ============================================================================

# ---- byte addresses ----
set A_START  0x0C
set A_STAT   0x10
set A_RES    0x14
set A_WINDOW 0x4000   ;# word 0x1000 << 2 — base of the 2500-word data window

# ---- dataset (must match the weights baked into the .sof) ----
# ROM = Chapman QAT-INT8 (nb 8,7,6,7). BOTH datasets run on the SAME bitstream:
# Georgia is a zero-shot far-transfer test, so it needs no weight reload — which
# is exactly why the ROM build can run it at all.
#
#   Chapman (in-distribution) : expect 94.27% (4688/4973)
#   Georgia (zero-shot)       : expect 93.00% (5077/5459)
#
# NOTE — display label only. The underlying files keep their on-disk names
# (ningba_test_*.bin, built from ningbo_dataset_clip16.npz); nothing about the
# data changed. "Chapman" here is the name used in the thesis text.
#
# Paths are resolved against THIS script's own directory, not the shell's cwd:
# System Console launched from the Quartus GUI starts in the project directory
# (hardware/fpga/), so a bare "demo_data/..." would not resolve. The log is
# written next to the script for the same reason.
set ::HERE [file dirname [file normalize [info script]]]

# Datasets to run, in order: {tag  ecg.bin  labels.bin  expected_correct  expected_n}
# `tag` is the DISPLAY label only; the .bin paths are the real on-disk names.
set ::DATASETS {
    {Chapman  demo_data/ningba_test_ecg_int8.bin   demo_data/ningba_test_labels.bin   4688 4973}
    {Georgia  demo_data/georgia_test_ecg_int8.bin  demo_data/georgia_test_labels.bin  5077 5459}
}
set ::SAMPLE_LEN  2500

# 0 = all samples. For a quick smoke run, set this BEFORE sourcing the script:
#     set ::MAX_SAMPLES 20
#     source .../ecg_jtag_rom.tcl
# `info exists` keeps that value instead of clobbering it back to 0.
if {![info exists ::MAX_SAMPLES]} { set ::MAX_SAMPLES 0 }

set ::LOG_ENABLE 1
set ::log_fh     ""

# ----------------------------------------------------------------------------
# JTAG master service
# ----------------------------------------------------------------------------
proc open_master {} {
    set masters [get_service_paths master]
    if {[llength $masters] == 0} {
        error "No JTAG Avalon master found. Is the board programmed with jtag_top.sof and the USB-Blaster connected?"
    }
    set m [lindex $masters 0]
    open_service master $m
    puts "Opened master: $m"
    return $m
}

proc load_ecg {m bytes} {
    global A_WINDOW
    set words {}
    foreach b $bytes { lappend words [expr {$b & 0xFF}] }
    master_write_32 $m $A_WINDOW $words
}

proc read_status {m} {
    global A_STAT
    return [expr {[lindex [master_read_32 $m $A_STAT 1] 0]}]
}

# Block until done_latched (bit1) is set.
proc wait_done {m what} {
    set tries 0
    while {1} {
        if {([read_status $m] & 0x2) == 0x2} break
        incr tries
        if {$tries > 100000} { error "Timeout waiting for $what" }
    }
}

proc start_inference {m} {
    global A_START
    master_write_32 $m $A_START 1
}

proc read_result {m} {
    global A_RES
    return [expr {[lindex [master_read_32 $m $A_RES 1] 0] & 0x3}]
}

proc read_bytes {path} {
    set f [open $path rb]
    set data [read $f]
    close $f
    binary scan $data c* signed
    return $signed
}

proc get_sample {ecg_all s} {
    set off [expr {$s * $::SAMPLE_LEN}]
    return [lrange $ecg_all $off [expr {$off + $::SAMPLE_LEN - 1}]]
}

proc logputs {line} {
    puts $line
    if {$::log_fh ne ""} { puts $::log_fh $line; flush $::log_fh }
}

# ----------------------------------------------------------------------------
# Run ONE dataset. Strictly sequential per sample:
#   load (one block write) -> start -> wait done -> read result.
# Returns {correct n_run elapsed_s}.
# ----------------------------------------------------------------------------
proc run_dataset {m tag ecg_path lbl_path exp_ok exp_n} {
    set names {AFIB GSVT SB SR}

    set ecg_all [read_bytes $ecg_path]
    set lbl_all [read_bytes $lbl_path]
    set n_total [expr {[llength $ecg_all] / $::SAMPLE_LEN}]
    set n_run   $n_total
    if {$::MAX_SAMPLES > 0 && $::MAX_SAMPLES < $n_run} { set n_run $::MAX_SAMPLES }

    logputs ""
    logputs "=============================================="
    logputs "  DATASET: $tag"
    logputs "=============================================="
    logputs "Samples in file: $n_total ; running: $n_run"
    logputs [format "Expected (SW bit-exact): %d/%d = %.2f%%" \
             $exp_ok $exp_n [expr {100.0 * $exp_ok / $exp_n}]]
    logputs ""

    set correct 0
    for {set i 0} {$i < 4} {incr i} { set hit($i) 0 ; set sup($i) 0 }

    set t0 [clock milliseconds]
    for {set s 0} {$s < $n_run} {incr s} {
        load_ecg $m [get_sample $ecg_all $s]
        start_inference $m
        wait_done $m "done($tag s=$s)"
        set pred  [read_result $m]
        set truth [expr {[lindex $lbl_all $s] & 0xFF}]

        incr sup($truth)
        if {$pred == $truth} { incr correct ; incr hit($truth) }

        set line [format "%-8s sample %4d : pred=%d (%-4s) truth=%d (%-4s) %s" \
                  $tag $s $pred [lindex $names $pred] $truth [lindex $names $truth] \
                  [expr {$pred==$truth ? "OK" : "X"}]]
        # File: every sample. Console: thinned so the window stays readable.
        if {$::log_fh ne ""} { puts $::log_fh $line ; flush $::log_fh }
        if {$n_run <= 20 || ($s % 250) == 0} { puts $line }
    }
    set elapsed [expr {([clock milliseconds] - $t0) / 1000.0}]
    set acc [expr {100.0 * $correct / $n_run}]

    logputs ""
    logputs "  --- $tag result ---"
    foreach i {0 1 2 3} {
        if {$sup($i) > 0} {
            logputs [format "  %-4s : %4d/%4d recall = %6.2f%%" \
                     [lindex $names $i] $hit($i) $sup($i) \
                     [expr {100.0 * $hit($i) / $sup($i)}]]
        }
    }
    logputs "  ----------------------------------------------"
    logputs [format "  Accuracy      : %d/%d = %.2f%%" $correct $n_run $acc]
    if {$n_run == $exp_n} {
        logputs [format "  Expected (SW) : %d/%d = %.2f%%   -> %s" \
                 $exp_ok $exp_n [expr {100.0 * $exp_ok / $exp_n}] \
                 [expr {$correct == $exp_ok ? "MATCH (bit-exact)" : "DIFFERS"}]]
    } else {
        logputs "  Expected (SW) : n/a (partial run, MAX_SAMPLES=$::MAX_SAMPLES)"
    }
    logputs [format "  Wall time     : %.1f s  (%.1f ms/sample, JTAG-bound)" \
             $elapsed [expr {1000.0 * $elapsed / $n_run}]]

    return [list $correct $n_run $elapsed]
}

# ----------------------------------------------------------------------------
# Main — run every dataset in ::DATASETS on the SAME bitstream.
# ----------------------------------------------------------------------------
proc main {} {
    set log_path ""
    if {$::LOG_ENABLE} {
        set ts [clock format [clock seconds] -format "%Y%m%d_%H%M%S"]
        set log_path [file join $::HERE "ecg_rom_${ts}.log"]
        set ::log_fh [open $log_path w]
    }

    logputs "# ECG accelerator — ROM single-load build (hardware/RTL/) on DE10-Standard"
    logputs "# Weights baked into bitstream (Chapman QAT-INT8, nb=8,7,6,7); no runtime reload."
    logputs "# Georgia runs zero-shot on the SAME weights -> same bitstream, no reconfig."
    logputs "# Core latency 5216 cycles = 52.16 us @ 100 MHz (per tb_top.v)."
    logputs "# Run started: [clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}]"

    set m [open_master]

    set results {}
    foreach ds $::DATASETS {
        lassign $ds tag ecg lbl exp_ok exp_n
        set r [run_dataset $m $tag [file join $::HERE $ecg] [file join $::HERE $lbl] \
               $exp_ok $exp_n]
        lappend results [list $tag {*}$r $exp_ok $exp_n]
    }

    logputs ""
    logputs "=============================================="
    logputs "  SUMMARY — DE10-Standard (Cyclone V), 1 bitstream"
    logputs "=============================================="
    logputs [format "  %-9s %-14s %-14s %s" "dataset" "on-board" "expected(SW)" "match"]
    foreach r $results {
        lassign $r tag correct n_run elapsed exp_ok exp_n
        set mark [expr {($n_run == $exp_n && $correct == $exp_ok) ? "yes" : \
                        ($n_run == $exp_n ? "NO" : "partial")}]
        logputs [format "  %-9s %4d/%4d %5.2f%%  %4d/%4d %5.2f%%  %s" \
                 $tag $correct $n_run [expr {100.0*$correct/$n_run}] \
                 $exp_ok $exp_n [expr {100.0*$exp_ok/$exp_n}] $mark]
    }
    logputs "=============================================="
    logputs "# Run finished: [clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}]"

    close_service master $m
    if {$::log_fh ne ""} {
        close $::log_fh
        puts "Results written to $log_path"
    }
}

main
