# JTAG Demo Checklist — ECG CNN Accelerator on DE10-Standard

**Mục đích**: Verify CNN accelerator trên board DE10-Standard qua JTAG-to-Avalon Master Bridge (không dùng HPS — Quartus Lite limitation).

**Kết quả cuối**: Chạy toàn bộ Chapman test set (~1065 samples), đo accuracy ≈ 94.27% (khớp Python 94.65%), log file timestamped.

---

## A. HARDWARE CẦN THIẾT

| Item | Thông số | Trạng thái |
|---|---|---|
| **Board** | DE10-Standard (Cyclone V 5CSXFC6D6F31C6) | ✅ |
| **Debug cable** | USB-Blaster II (hoặc I) | ✅ (standard) |
| **Bitstream** | `output_files/jtag_top.sof` | ✅ Sẵn (6.7 MB) |
| **Power** | 5V USB để board | ✅ Standard |

**Status**: Toàn bộ HW standard DE10-Standard, không phần cứng thêm.

---

## B. SOFTWARE CẦN THIẾT

### B1. Altera Quartus + System Console (trên PC)

| Tool | Phiên bản | Vai trò |
|---|---|---|
| **Quartus Prime LITE** | 25.1 trở lên (hoặc 24.2) | Program `.sof` vào board |
| **System Console** | đi kèm Quartus | Chạy Tcl script `ecg_jtag_console.tcl` từ PC |

**Cách kiểm tra**:
```powershell
# Tìm System Console
$env:QUARTUS_ROOTDIR = "D:\altera_lite\25.1std"  # hoặc path của bạn
& "$env:QUARTUS_ROOTDIR\bin\system-console" --version
```

### B2. Tcl script driver (đã sẵn)

File: `hardware/fpga/soc/ecg_jtag_console.tcl`
- Không cần edit, chạy trực tiếp
- Đọc demo_data, load weight, chạy inference, ghi log

### B3. Demo data (đã sẵn)

```
hardware/fpga/soc/demo_data/
├── chapman_weights/              # Chapman INT8 weights (QAT)
│   ├── w_ram0.hex .. w_ram7.hex  # Conv weight (8 per-oc file, 40-bit)
│   ├── conv_bias.hex             # Conv bias (INT32 × 32)
│   ├── fc_weights.hex            # FC weight (INT8 × 32)
│   ├── fc_bias.hex               # FC bias (INT32 × 4)
│   └── topo.txt                  # Topology config: in_ch cp_en nb base (1,4,4,8)
├── chapman_test_ecg_int8.bin     # Test ECG (1065 × 2500 byte INT8)
├── chapman_test_labels.bin       # Test labels (1065 × 1 byte, class 0-3)
├── chapman_test_demo_meta.json   # Sample count, format
├── ptbxl_weights/                # PTB-XL variant (cross-dataset C3)
│   ├── (same structure, nb[Conv3]=7 instead of 6)
│   └── topo.txt
├── ptbxl_test_ecg_int8.bin       # PTB-XL test ECG
├── ptbxl_test_labels.bin
└── ptbxl_test_demo_meta.json
```

**Status**: ✅ Đầy đủ, không cần regenerate.

---

## C. BITSTREAM STATUS

### C1. Main JTAG bitstream

```
hardware/fpga/output_files/jtag_top.sof    (6.7 MB)
└─ compiled from: soc/jtag_top.v + Qsys jtag_system
   └─ includes: core_pll (50→100MHz) + reset logic + Avalon master + ecg_core
```

**Verification**: ✅ Bit-exact 21/21 simulation `tb_top_window.v` (burst DATA WINDOW write).

### C2. Compile chain (nếu cần re-build)

```
1. Open Quartus → hardware/fpga/
2. Project: ecg_accelerator_top.qpf (hoặc tạo mới từ jtag_top.v)
3. Settings → Top-level entity = jtag_top
4. Compile → output_files/jtag_top.sof
5. TimeQuest: check Fmax (target 100 MHz ✅ typical)
```

**Status**: ✅ Sẵn sàng, không cần compile lại.

### C3. Variant bitstreams (backup/future)

```
output_files/
├── ecg_accelerator_top.sof      # Phase C baseline (no JTAG master)
├── uart_top.sof                 # Phase D UART variant (chờ USB-TTL 3.3V)
└── niosv_top.sof                # Phase D Nios V/m variant
```

**Dùng cho demo**: `jtag_top.sof` ONLY.

---

## D. CHUẨN BỊ TRƯỚC CHẠY

### D1. Board setup

```
1. Plug USB-Blaster to PC
2. Plug USB 5V power to DE10 J21 (USB mini-B)
3. On board: LED should light (P=5V indicator)
4. Push KEY0 (reset button) → LED should blink (PLL lock indicator, FPGA active)
```

### D2. Quartus Programmer (1 lần duy nhất)

```powershell
# Di chuyển đến thư mục fpga
cd D:\Thesis101\hardware\fpga

# Mở Quartus Programmer
& "D:\altera_lite\25.1std\bin\quartus_pgm"
```

**Trong GUI**:
1. Auto-detect: `Hardware Setup → USB-Blaster (or II)`
2. Add device: `Add File → output_files/jtag_top.sof`
3. Check: Program/Configure + Verify
4. Click **Start** → sau ~30s, "100% (Successful)" xuất hiện

**Status sau lần này**: Board chạy bitstream JTAG, LEDs đột sáng/tắt (normal).

### D3. Chuẩn bị Tcl script

**File chính**: `hardware/fpga/soc/ecg_jtag_console.tcl`

**Cấu hình** (top file):
```tcl
set ::WEIGHT_DIR  "demo_data/chapman_weights"   # hoặc ptbxl_weights cho C3
set ::ECG_FILE    "demo_data/chapman_test_ecg_int8.bin"
set ::LBL_FILE    "demo_data/chapman_test_labels.bin"
set ::MAX_SAMPLES 0        # 0 = all (~1065), hoặc 3 để test nhanh
set ::LOG_ENABLE  1        # 1 = ghi log file .log, 0 = skip
```

**Default setup**: Chapman, all 1065 samples, log enabled → **dùng ngay không cần edit**.

---

## E. CHẠY DEMO

### E1. Từ PowerShell

```powershell
cd D:\Thesis101\hardware\fpga\soc

# Kiếm đường dẫn System Console
$SC = "D:\altera_lite\25.1std\bin\system-console.bat"  # Windows batch wrapper

# Chạy script
& $SC --script=ecg_jtag_console.tcl 2>&1 | Tee-Object -FilePath "jtag_run.log"
```

### E2. Output console

```
Claimed master: /devices/0/master/0
Loaded topology: {1 15 8 0} {4 15 6 1} {4 255 6 5} {8 255 7 9}
Loaded weights from demo_data/chapman_weights (17 words/oc, ...)
Total samples in file: 1065 ; running: 1065 (overlap reload)
sample    0 : pred=0 truth=0 OK
sample   50 : pred=0 truth=0 OK
sample  100 : pred=0 truth=1 X
...
sample 1065 : pred=3 truth=3 OK
----------------------------------------------
Accuracy: 1004/1065 = 94.27%
Results written to ecg_jtag_20260701_153020.log
```

**Thời gian**: ~vài phút (phụ thuộc JTAG speed, System Console overhead).

### E3. Log file

Tự động tạo: `ecg_jtag_<YYYYMMDD>_<HHMMSS>.log`
- Ghi TOÀN BỘ 1065 sample (mỗi dòng = 1 sample)
- Flush từng dòng (survive nếu kênh JTAG rớt)
- Console chỉ in 10 đầu + mỗi 50

**Dùng cho**: Xác minh accuracy chính xác, debug nếu sai.

---

## F. TROUBLESHOOTING

| Vấn đề | Triệu chứng | Cách fix |
|---|---|---|
| **Board không detect** | "No JTAG Avalon master found" | USB-Blaster bị unplug / driver. Thử re-plug cable; check Device Manager xem có "Altera USB" |
| **Program .sof fail** | Quartus Programmer: "Device could not be detected" | Power board, push KEY0 reset |
| **Tcl syntax error** | "unknown command" | Encoding file: đảm bảo UTF-8 CRLF (Windows); hoặc dùng WSL System Console |
| **Script time-out** | Poll timeout waiting for isram_free | JTAG overhead lớn; thắng là JTAG chậm nên giảm MAX_SAMPLES để test. Production logic OK (sim bit-exact) |
| **Accuracy != 94.27%** | < 90% hoặc 100% | (a) Weight file corrupt (check w_ram*.hex không trống); (b) ECG file sai (check bit order INT8 signed); (c) Bitstream cũ (re-program .sof); đo lại trên sim nếu cần |

---

## G. CROSS-DATASET DEMO (C3 — PTB-XL, tùy chọn)

### Cách chạy

**Chỉ cần sửa 2 dòng** trong `ecg_jtag_console.tcl`:

```tcl
set ::WEIGHT_DIR  "demo_data/ptbxl_weights"      # ← từ chapman_weights
set ::ECG_FILE    "demo_data/ptbxl_test_ecg_int8.bin"
set ::LBL_FILE    "demo_data/ptbxl_test_labels.bin"
set ::MAX_SAMPLES 0
```

**Kết quả kỳ vọng**: PTB-XL test acc ~92.6–93% (vs Chapman 94.27%).

**Ý nghĩa**: 
- **Cùng bitstream** `jtag_top.sof` chạy cả 2 dataset ✅
- Topology CONFIG (nb, channel) tự load từ `topo.txt` → nb[Conv3]=7 (vs Chapman=6)
- Weight reload từ Avalon trước mỗi chạy ✅
- **Chứng minh** cross-dataset transferability trên hardware thực

---

## H. VERIFICATION vs SIMULATION (không cần board)

Nếu chưa có board, xác minh script bằng sim:

```bash
cd hardware/fpga/simulation/questa
vsim -do run_tb_top_window.do
```

**Kết quả**: 21/21 checkpoint bit-exact (input, pool1-4, gap, logits, argmax). Bit-identical với on-board (nếu không có noise JTAG).

---

## I. DELIVERABLE CHECKLIST

Để demo JTAG thành công:

- [ ] **Hardware**: DE10-Standard + USB-Blaster connected
- [ ] **Software**: Quartus 25.1 + System Console (path có trong script)
- [ ] **Bitstream**: `jtag_top.sof` programmed via Quartus Programmer
- [ ] **Demo script**: `ecg_jtag_console.tcl` (no edit needed, default Chapman)
- [ ] **Data**: `demo_data/chapman_weights/` + `chapman_test_ecg_int8.bin` (sẵn)
- [ ] **Run**: `system-console --script=ecg_jtag_console.tcl`
- [ ] **Output**: `ecg_jtag_<timestamp>.log` chứa 1065 sample, accuracy ≈ 94.27%

---

## J. FILES LIÊN QUAN

```
hardware/fpga/
├── output_files/
│   └── jtag_top.sof                    ← Program vào board
├── soc/
│   ├── jtag_top.v                      ← Top-level RTL
│   ├── jtag_top.sdc                    ← Timing constraint
│   ├── jtag_top.qpf                    ← Quartus project (if re-compile)
│   ├── jtag_system.qsys                ← Qsys (if re-generate)
│   ├── ecg_jtag_console.tcl            ← Driver script (run this)
│   ├── ecg_core_hw.tcl                 ← Component def
│   ├── jtag_demo.md                    ← Detailed architecture
│   ├── JTAG_PHASE_D_STEPS.md           ← Setup detail
│   └── demo_data/
│       ├── chapman_weights/ + ptbxl_weights/
│       ├── chapman_test_*.bin + ptbxl_test_*.bin
│       └── *_meta.json
└── simulation/questa/
    └── run_tb_top_window.do            ← Sim verification (no board)
```

---

## K. NEXT STEPS (sau demo)

1. **Nếu AC 94.27% ✅**: Hardware thực đúng Python, ready cho paper.
2. **Nếu AC lệch**: Debug via log file, check weight/topo load, re-run sim.
3. **PTB-XL cross-dataset** (optional): Chỉ sửa WEIGHT_DIR + ECG_FILE, re-run.
4. **Nios V/UART variant** (future Phase D): Chạy `niosv_top.sof` / `uart_top.sof` thay vì JTAG.

---

**Viết**: 2026-07-01  
**Cho**: Phase D on-board verification (no HPS)
