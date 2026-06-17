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
#   word 0x0003 (byte 0x000C) W : start    [0]   start + clear done
#   word 0x0004 (byte 0x0010) R : status   {isram_free, done_latched, busy}
#                                 bit0=busy, bit1=done_latched, bit2=isram_free
#   word 0x0005 (byte 0x0014) R : result   [1:0]   predicted class
#   DATA WINDOW: word 0x1000..0x19C3 (byte 0x4000..0x6710) W : one word per SRAM
#       byte (the slave auto-drives din+addr+we from the word's address). A whole
#       2500-byte sample is shipped in ONE master_write_32 block call instead of
#       ~7500 per-byte transactions — drops the full set from ~10 h to minutes.
# ============================================================================

# ---- byte addresses ----
set A_START  0x0C
set A_STAT   0x10
set A_RES    0x14
set A_WINDOW 0x4000   ;# word 0x1000 << 2 — base of the 2500-word data window

# Weight window (avalon_slave.v, addr[13]=1). Byte = word << 2.
#   conv weight : word 0x2000 | (ramword<<4) | (oc<<1) | hi   -> byte 0x8000 ...
#   conv bias   : word 0x2800 | idx                            -> byte 0xA000 ...
#   FC w/bias   : word 0x3000 | addr ([5]=1 -> bias)           -> byte 0xC000 ...
set A_WCONV  0x8000   ;# (0x2000 << 2) conv weight base
set A_WBIAS  0xA000   ;# (0x2800 << 2) conv bias base
set A_WFC    0xC000   ;# (0x3000 << 2) FC weight base
set A_WFCB   0xC080   ;# (0x3020 << 2) FC bias base (fcw_addr[5]=1)

# Topology CONFIG window (avalon_slave.v addr[13]=1, addr[12:11]=11):
#   word = 0x3800 | (field<<2) | layer  ->  byte = word<<2 = 0xE000 | (field<<4) | (layer<<2)
#   field: 0=in_ch 1=cp_en 2=nb 3=base ; layer 0..3 = Conv1..4
set A_CFG    0xE000   ;# (0x3800 << 2) topology config base

# ---- weight config ----
set ::WEIGHT_DIR  "demo_data/chapman_weights"   ;# set to ptbxl_weights for C3 cross-dataset

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
# Load one 2500-byte ECG sample into the input SRAM via the DATA WINDOW.
#   Build a list of 2500 32-bit words (low byte = ECG sample) and ship them in
#   ONE master_write_32 block call. System Console increments the byte address
#   by 4 per element, which matches the word-addressed slave: word 0x1000+i ->
#   SRAM index i, with the slave auto-driving din+addr+we. This replaces the old
#   3-writes-per-byte loop (7500 transactions/sample) with a single block write.
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Read a $readmemh hex file (one hex word per line, '//' comments tolerated) into
# a Tcl list of integers.
# ----------------------------------------------------------------------------
proc read_hex_file {path} {
    set f [open $path r]
    set vals {}
    while {[gets $f line] >= 0} {
        set line [string trim $line]
        if {$line eq "" || [string match "//*" $line]} { continue }
        lappend vals [expr {"0x$line"}]
    }
    close $f
    return $vals
}

# ----------------------------------------------------------------------------
# Load ALL weights from $WEIGHT_DIR into the accelerator over the weight window.
# Runtime weight reload (Phase B01): the SAME bitstream runs Chapman or PTB-XL
# weights — point WEIGHT_DIR at the desired set before calling.
#
#   conv : 8 per-oc RAMs (w_ram0..7.hex), 17 words × 40 bit. Each 40-bit entry =
#          two 32-bit writes (lo bits[31:0], hi bits[39:32]). The slave assembles
#          them and writes the 40-bit RAM word on the hi half. Linear word-address
#          order (word-major, oc, lo/hi) matches the slave's offset decode.
#   bias : conv_bias.hex (32 × INT32), one write each.
#   FC   : fc_weights.hex (32 × INT8) + fc_bias.hex (4 × INT32, fcw_addr[5]=1).
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Read topo.txt (one line per layer Conv1..4: "in_ch cp_en nb base", '#' comments
# tolerated) into a list of 4 {in_ch cp_en nb base} sublists. Returns {} if the
# file is absent (caller falls back to the hard-coded 17-word Chapman layout).
# ----------------------------------------------------------------------------
proc read_topo {dir} {
    set path "$dir/topo.txt"
    if {![file exists $path]} { return {} }
    set f [open $path r]
    set rows {}
    while {[gets $f line] >= 0} {
        set line [string trim $line]
        if {$line eq "" || [string match "#*" $line]} { continue }
        lappend rows [lrange $line 0 3]   ;# in_ch cp_en nb base
    }
    close $f
    return $rows
}

# ----------------------------------------------------------------------------
# Write the per-layer topology into the CONFIG window so a SINGLE bitstream runs
# an arbitrary 1..8-channel-per-layer layout (verified bit-exact by tb_topo_sweep).
# topo = list of 4 {in_ch cp_en nb base}. With NO call the slave keeps its reset
# default (Chapman 1,4,4,8) — so the legacy flow is unchanged.
# ----------------------------------------------------------------------------
proc load_topology {m topo} {
    global A_CFG
    for {set L 0} {$L < 4} {incr L} {
        lassign [lindex $topo $L] ic ce nb bs
        # byte addr = A_CFG | (field<<4) | (layer<<2)
        master_write_32 $m [expr {$A_CFG | (0<<4) | ($L<<2)}] [expr {$ic & 0xF}]
        master_write_32 $m [expr {$A_CFG | (1<<4) | ($L<<2)}] [expr {$ce & 0xFF}]
        master_write_32 $m [expr {$A_CFG | (2<<4) | ($L<<2)}] [expr {$nb & 0x1F}]
        master_write_32 $m [expr {$A_CFG | (3<<4) | ($L<<2)}] [expr {$bs & 0x1F}]
    }
    puts "Loaded topology: $topo"
}

proc load_weights {m} {
    global A_WCONV A_WBIAS A_WFC A_WFCB
    set dir $::WEIGHT_DIR

    # Determine the number of conv weight RAM words from topo.txt (base[3]+in_ch[3]);
    # fall back to the Chapman 17 if no topo file is present.
    set topo [read_topo $dir]
    if {[llength $topo] == 4} {
        lassign [lindex $topo 3] ic3 ce3 nb3 bs3
        set nwords [expr {$bs3 + $ic3}]
    } else {
        set nwords 17
    }

    # --- conv weights: read 8 per-oc files (each $nwords x 40-bit hex) ---
    for {set oc 0} {$oc < 8} {incr oc} {
        set ram($oc) [read_hex_file "$dir/w_ram$oc.hex"]
    }
    # Build one block in linear offset order: for each ramword, for each oc, lo then hi.
    set conv_words {}
    for {set w 0} {$w < $nwords} {incr w} {
        for {set oc 0} {$oc < 8} {incr oc} {
            set e [lindex $ram($oc) $w]
            lappend conv_words [expr {$e & 0xFFFFFFFF}]            ;# lo (hi bit=0)
            lappend conv_words [expr {($e >> 32) & 0xFF}]          ;# hi (hi bit=1)
        }
    }
    master_write_32 $m $A_WCONV $conv_words

    # --- conv bias (32 × INT32) ---
    master_write_32 $m $A_WBIAS [read_hex_file "$dir/conv_bias.hex"]

    # --- FC weights (32 × INT8) ---
    master_write_32 $m $A_WFC [read_hex_file "$dir/fc_weights.hex"]

    # --- FC bias (4 × INT32) at fcw_addr[5]=1 region ---
    master_write_32 $m $A_WFCB [read_hex_file "$dir/fc_bias.hex"]

    puts "Loaded weights from $dir ($nwords words/oc, [llength $conv_words] conv words + bias + FC)"
}

proc load_ecg {m bytes} {
    global A_WINDOW
    set words {}
    foreach b $bytes {
        lappend words [expr {$b & 0xFF}]   ;# int8 -> unsigned byte in word[7:0]
    }
    master_write_32 $m $A_WINDOW $words
}

# ----------------------------------------------------------------------------
# Status helpers. status = {isram_free(bit2), done_latched(bit1), busy(bit0)}.
# ----------------------------------------------------------------------------
proc read_status {m} {
    global A_STAT
    return [expr {[lindex [master_read_32 $m $A_STAT 1] 0]}]
}

# Block until ALL status bits in `mask` are set
# (bit0=busy, bit1=done_latched, bit2=isram_free).
proc poll_status {m mask {what "status"}} {
    set tries 0
    while {1} {
        if {([read_status $m] & $mask) == $mask} break
        incr tries
        if {$tries > 100000} { error "Timeout waiting for $what" }
    }
}

# Pulse start for one window (kicks off Conv1 on the current input_sram contents).
proc start_inference {m} {
    global A_START
    master_write_32 $m $A_START 1
}

# Read the latched argmax class (call only after done_latched is set).
proc read_result {m} {
    global A_RES
    return [expr {[lindex [master_read_32 $m $A_RES 1] 0] & 0x3}]
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

# Extract sample s (SAMPLE_LEN bytes) from the flat byte list.
proc get_sample {ecg_all s} {
    set off [expr {$s * $::SAMPLE_LEN}]
    return [lrange $ecg_all $off [expr {$off + $::SAMPLE_LEN - 1}]]
}

# ----------------------------------------------------------------------------
# Logging: echo to the System-Console window AND write a timestamped .log file
# on the PC, so each run is kept separately (e.g. ecg_jtag_20260614_143052.log).
# The log captures EVERY sample (the Console still prints a thinned subset so it
# stays readable). Set ::LOG_ENABLE to 0 to disable file logging.
# ----------------------------------------------------------------------------
set ::LOG_ENABLE 1
set ::log_fh     ""

proc log_filename {} {
    set ts [clock format [clock seconds] -format "%Y%m%d_%H%M%S"]
    return "ecg_jtag_${ts}.log"
}

proc logputs {line} {
    puts $line
    if {$::log_fh ne ""} {
        puts $::log_fh $line
        flush $::log_fh   ;# flush per line so a mid-run channel drop still leaves a usable log
    }
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
proc main {} {
    set log_path ""
    if {$::LOG_ENABLE} {
        set log_path [log_filename]
        set ::log_fh [open $log_path w]
        logputs "# ECG JTAG run - dataset=$::ECG_FILE"
    }

    set m [open_master]

    # Runtime reconfig: if WEIGHT_DIR ships a topo.txt, push its per-layer
    # channel/nb/base into the CONFIG window first. Absent topo.txt -> the slave
    # keeps its Chapman reset default, so the legacy flow is byte-identical.
    set topo [read_topo $::WEIGHT_DIR]
    if {[llength $topo] == 4} {
        load_topology $m $topo
    } else {
        puts "No topo.txt in $::WEIGHT_DIR -> using slave reset default (Chapman 1,4,4,8)"
    }

    # Phase B01: load weights from the PC over the bus before any inference.
    load_weights $m

    set ecg_all [read_bytes $::ECG_FILE]
    set lbl_all [read_bytes $::LBL_FILE]
    set n_total [expr {[llength $ecg_all] / $::SAMPLE_LEN}]
    set n_run   $n_total
    if {$::MAX_SAMPLES > 0 && $::MAX_SAMPLES < $n_run} { set n_run $::MAX_SAMPLES }

    logputs "Total samples in file: $n_total ; running: $n_run (overlap reload)"
    set correct 0

    # ── Overlap-reload pipeline ─────────────────────────────────────────────
    # Conv1 is the only consumer of input_sram. Once the core leaves CONV1
    # (isram_free==1), the datapath reads ping_pong only, so we may load the
    # NEXT sample into input_sram while THIS inference is still computing
    # Conv2/3/4/GAP/FC. Single buffer -> enforce the driver-side contract:
    #   (1) load N+1 only after isram_free==1 (no collision with Conv1 of N),
    #   (2) start N+1 only after result N has been read (N+1 overwrites N).
    if {$n_run > 0} {
        load_ecg $m [get_sample $ecg_all 0]   ;# prime buffer with sample 0
        start_inference $m
    }
    for {set s 0} {$s < $n_run} {incr s} {
        # Overlap: as soon as Conv1(s) released input_sram, load sample s+1.
        if {$s + 1 < $n_run} {
            # isram_free==1: inference s has left Conv1, so input_sram is no
            # longer read by the datapath and may be reloaded. (Loading s+1 here
            # is safe even if s already finished: start(s+1) below happens only
            # after read_result(s), so s's class is never disturbed.)
            poll_status $m 0x4 "isram_free(s=$s)"
            load_ecg $m [get_sample $ecg_all [expr {$s + 1}]]
        }
        # Wait for inference s to finish, read its class.
        poll_status $m 0x2 "done(s=$s)"
        set pred [read_result $m]
        # Buffer now holds s+1 (loaded above) -> safe to launch it.
        if {$s + 1 < $n_run} { start_inference $m }

        set truth [expr {[lindex $lbl_all $s] & 0xFF}]
        if {$pred == $truth} { incr correct }
        set line [format "sample %4d : pred=%d truth=%d %s" \
                  $s $pred $truth [expr {$pred==$truth ? "OK" : "X"}]]
        # File: every sample. Console: thinned (first 10 + every 50th).
        if {$::log_fh ne ""} { puts $::log_fh $line; flush $::log_fh }
        if {$n_run <= 10 || ($s % 50) == 0} { puts $line }
    }
    set acc [expr {100.0 * $correct / $n_run}]
    logputs "----------------------------------------------"
    logputs [format "Accuracy: %d/%d = %.2f%%" $correct $n_run $acc]
    close_service master $m

    if {$::log_fh ne ""} {
        close $::log_fh
        puts "Results written to $log_path"
    }
}

main
