# ============================================================================
# ecg_jtag_one.tcl — demo TỪNG MẪU trên board (bản ROM, hardware/RTL/).
#
# Khác ecg_jtag_rom.tcl: script kia chạy toàn tập để LẤY SỐ THỐNG KÊ (10.432 mẫu,
# ra bảng độ chính xác). Script này chạy 4 mẫu đại diện 4 lớp và in RÕ TỪNG BƯỚC
# của một lần suy luận — dùng khi trình bày/bảo vệ, để thấy CƠ CHẾ hoạt động chứ
# không phải con số tổng hợp.
#
# 4 mẫu được chọn từ lần chạy toàn tập (ecg_rom_20260731_152219.log): mỗi lớp lấy
# một mẫu mà phần cứng đã dự đoán ĐÚNG, nằm giữa vùng của lớp đó.
#
# Hình dạng sóng 4 mẫu này: software/python/results/figures/demo_4samples.png
#
# CÁCH CHẠY (board đã nạp jtag_top.sof):
#   Tools -> System Debugging Tools -> System Console
#   source D:/Thesis101/hardware/fpga/soc/ecg_jtag_one.tcl
# ============================================================================

# ---- byte addresses (RTL/avalon_slave.v) ----
set A_START  0x0C
set A_STAT   0x10
set A_RES    0x14
set A_WINDOW 0x4000

set ::HERE [file dirname [file normalize [info script]]]
set ::ECG_FILE   [file join $::HERE "demo_data/ningba_test_ecg_int8.bin"]
set ::LBL_FILE   [file join $::HERE "demo_data/ningba_test_labels.bin"]
set ::SAMPLE_LEN 2500

# {chỉ số mẫu trong tập test}  — mỗi lớp 1 mẫu, xem header
set ::PICKS {568 1578 2903 4397}

proc open_master {} {
    set masters [get_service_paths master]
    if {[llength $masters] == 0} {
        error "Khong tim thay JTAG master. Board da nap jtag_top.sof va cam USB-Blaster chua?"
    }
    set m [lindex $masters 0]
    open_service master $m
    return $m
}

proc read_bytes {path} {
    set f [open $path rb]
    set data [read $f]
    close $f
    binary scan $data c* signed
    return $signed
}

proc main {} {
    set names {AFIB GSVT SB SR}

    puts ""
    puts "=================================================================="
    puts "   DEMO TUNG MAU - CNN Accelerator tren DE10-Standard"
    puts "   (Cyclone V 5CSXFC6D6F31C6, 100 MHz, INT8 power-of-2)"
    puts "=================================================================="
    puts "   Trong so Chapman QAT-INT8 bake san trong bitstream (ROM)."
    puts "   Moi mau: nap 2500 byte ECG -> start -> doi done -> doc lop."
    puts ""

    set m [open_master]
    puts "   JTAG master: da ket noi"
    puts ""

    set ecg_all [read_bytes $::ECG_FILE]
    set lbl_all [read_bytes $::LBL_FILE]

    set n   0
    set nok 0
    foreach idx $::PICKS {
        incr n
        set truth [expr {[lindex $lbl_all $idx] & 0xFF}]
        set off   [expr {$idx * $::SAMPLE_LEN}]
        set bytes [lrange $ecg_all $off [expr {$off + $::SAMPLE_LEN - 1}]]

        puts "------------------------------------------------------------------"
        puts [format "  \[%d/4\]  Mau #%d      Nhan thuc te: %s" \
              $n $idx [lindex $names $truth]]
        puts ""

        # 1. Nạp tín hiệu ECG (1 lệnh khối, DATA WINDOW tự sinh addr+we)
        set words {}
        foreach b $bytes { lappend words [expr {$b & 0xFF}] }
        set t0 [clock milliseconds]
        master_write_32 $m $::A_WINDOW $words
        set t_load [expr {[clock milliseconds] - $t0}]
        puts [format "    1. Nap tin hieu ECG   : 2500 mau INT8  (%d ms qua JTAG)" $t_load]

        # 2. Kích hoạt suy luận
        master_write_32 $m $::A_START 1
        puts             "    2. Kich hoat         : START = 1"

        # 3. Chờ hoàn tất
        set tries 0
        while {1} {
            set st [lindex [master_read_32 $m $::A_STAT 1] 0]
            if {($st & 0x2) == 0x2} break
            incr tries
            if {$tries > 100000} { error "Timeout cho mau #$idx" }
        }
        puts             "    3. Trang thai        : busy -> done"
        puts             "       Do tre loi        : 5216 chu ky = 52,16 us @ 100 MHz"

        # 4. Đọc kết quả
        set pred [expr {[lindex [master_read_32 $m $::A_RES 1] 0] & 0x3}]
        puts ""
        if {$pred == $truth} {
            incr nok
            puts [format "    => KET QUA: %s  (lop %d)      DUNG" \
                  [lindex $names $pred] $pred]
        } else {
            puts [format "    => KET QUA: %s  (lop %d)      SAI (that: %s)" \
                  [lindex $names $pred] $pred [lindex $names $truth]]
        }
        puts ""
    }

    puts "=================================================================="
    puts [format "   Ket qua: %d/4 mau dung" $nok]
    puts "   Toan tap (4973 mau): 94,27% - khop bit voi mo hinh Python"
    puts "=================================================================="
    puts ""

    close_service master $m
}

main
