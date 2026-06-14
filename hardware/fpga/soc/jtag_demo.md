# JTAG-to-Avalon On-Board Demo — ECG CNN Accelerator (Phase D)

Mô tả chi tiết luồng demo chạy core CNN trên DE10-Standard **không dùng HPS**, điều khiển
từ PC qua **JTAG-to-Avalon Master Bridge** + System Console.

> Lý do không dùng HPS: IP Cyclone V HPS bị license-gate ở Quartus **Standard**; bản **Lite**
> không có. Để chạy core đã verify trên board mà không cần HPS, ta drive Avalon-MM slave của
> nó từ PC qua JTAG TAP (USB-Blaster), tái dùng `avalon_slave.v` y nguyên.

---

## 1. Kiến trúc đường demo

```
PC (System Console, Tcl)
  │  ecg_jtag_console.tcl  (master_write_32 / master_read_32)
  ▼
USB-Blaster II  ──JTAG TAP──►  FPGA
                                 │
              ┌──────────────────┴───────────────────────────┐
              │  Qsys system: jtag_system                     │
              │   ┌─────────────────┐   Avalon-MM (32-bit)    │
              │   │ JTAG-to-Avalon  │──────────────┐          │
              │   │  Master (master_0)             │          │
              │   └─────────────────┘              ▼          │
              │   clock_bridge (100MHz) ┌────────────────────┐│
              │   reset_bridge          │ ecg_core_0         ││
              │                         │ (ecg_accelerator_  ││
              │                         │  top: avalon_slave ││
              │                         │  + ecg_core)       ││
              │                         └────────────────────┘│
              └────────────────────────────────────────────────┘
   jtag_top.v: core_pll 50→100MHz + reset glue, wrap jtag_system
```

- **Top synth**: `soc/jtag_top.v` — wrap Qsys `jtag_system` + `core_pll` (50→100 MHz) + reset sync.
- **Qsys `jtag_system`** (Platform Designer): JTAG master + clock/reset bridge + component `ecg_core`
  (thực chất bọc `ecg_accelerator_top` = `avalon_slave` + `ecg_core`).
- **Component def**: `soc/ecg_core_hw.tcl` (khai báo Avalon slave 13-bit, addressSpan 8192).

---

## 2. Hai tần số TÁCH BIỆT (hay nhầm)

| Clock | Tần số | Nguồn | Vai trò |
|---|---|---|---|
| **Core clock** | **100 MHz** | board osc 50 MHz → `core_pll` ×2 | clock của `ecg_core`; inference = 5216 cy = **52 µs** |
| **JTAG TAP clock (TCK)** | ~6–24 MHz | USB-Blaster II (riêng) | truyền bit JTAG; **độc lập** với core, SDC không ràng buộc |

> **TCK KHÔNG phải nguyên nhân demo chậm.** Bottleneck là overhead phần mềm System Console/Tcl
> (~4.5 ms mỗi lệnh `master_*`), không phải băng thông JTAG. Vì vậy tăng TCK không cứu được —
> phải **giảm số lệnh** (xem §5).

---

## 3. Avalon-MM register map (avalon_slave.v)

Địa chỉ Avalon đơn vị **WORD** (13-bit). `master_*` dùng địa chỉ **BYTE** = word × 4.

| Word addr | Byte addr | R/W | Ý nghĩa |
|---|---|---|---|
| 0x0000 | 0x0000 | W | `sram_din[7:0]` (legacy byte path) |
| 0x0001 | 0x0004 | W | `sram_wr_addr[11:0]` (legacy) |
| 0x0002 | 0x0008 | W | `sram_we[0]` (legacy) |
| 0x0003 | 0x000C | W | `start[0]` (kick inference, clear done_latched) |
| 0x0004 | 0x0010 | R | `status` = {isram_free, done_latched, busy} |
| 0x0005 | 0x0014 | R | `result[1:0]` (argmax class 0..3) |
| **0x1000..0x19C3** | **0x4000..0x6710** | W | **DATA WINDOW** — 2500 word, mỗi word = 1 byte ECG |

**DATA WINDOW** (`addr[12]==1`): khi ghi 1 word vào địa chỉ `0x1000+i`, slave tự động
`sram_din<=word[7:0]; sram_wr_addr<=i; sram_we<=1`. Tức **1 word = 1 byte SRAM**, base 0x0000.

> Window là **additive**: dải legacy 0x00–0x05 byte-identical với slave 5-bit cũ, nên
> regression `tb_top.v` (21/21 bit-exact) không bị ảnh hưởng.

---

## 4. Giao thức điều khiển (ecg_jtag_console.tcl)

4 class: AFIB(0), GSVT(1), SB(2), SR(3).

### `open_master` — claim JTAG master service
```tcl
set m [lindex [get_service_paths master] 0]
claim_service master $m ""
```

### `load_ecg` — nạp 1 sample (2500 byte) bằng MỘT block write
```tcl
# build list 2500 word (low byte = ECG int8 → unsigned), ship 1 lệnh
master_write_32 $m 0x4000 $words   ;# System Console tự tăng byte-addr +4/word
```
→ word `0x1000+i` (byte `0x4000+4i`) → SRAM index i. Slave auto din+addr+we.

### `run_inference` — start, poll done, đọc result
```tcl
master_write_32 $m 0x0C 1                 ;# START
while { !([master_read_32 $m 0x10 1] & 0x2) } { ... }   ;# poll done_latched
return [expr {[master_read_32 $m 0x14 1] & 0x3}]        ;# RESULT class
```

### `main` — quét cả test set, ghi log
- Đọc `demo_data/chapman_test_ecg_int8.bin` (+ `_labels.bin`), mỗi sample 2500 byte.
- Loop: `load_ecg` → `run_inference` → so với truth → đếm correct.
- **Log file** `ecg_jtag_<YYYYMMDD>_<HHMMSS>.log` (timestamp, mỗi lần chạy 1 file):
  ghi MỌI sample + flush từng dòng (sống sót nếu kênh JTAG rớt). Console in thưa (10 đầu + mỗi 50).
- Cuối: `Accuracy: <correct>/<n> = <acc>%`.

Config đầu file: `::ECG_FILE`, `::LBL_FILE`, `::MAX_SAMPLES` (0=all, vd 3 = sanity),
`::LOG_ENABLE` (1/0).

---

## 5. Vì sao demo trước ~10h → giờ vài phút

| | Demo trước | Hiện tại (DATA WINDOW) |
|---|---|---|
| Nạp 1 byte | 3 lệnh (din+addr+we) | — |
| Nạp 1 sample (2500 byte) | 7500 lệnh | **1 lệnh** block |
| Full set 1065 sample | ~8M lệnh × ~4.5 ms ≈ **~10 h** | ~vài nghìn lệnh ≈ **vài phút** |
| Core clock / RTL core / accuracy | 100 MHz / nguyên / 94.27% | **không đổi** |

Chỉ đổi **cách host gửi data** (gói cả sample 1 lệnh thay vì hỏi-đáp 7500 lần). Tốc độ *tính
toán* của core không đổi (vẫn 52 µs/inference). Cái rút ngắn là **thời gian chờ nạp qua dây debug**.

> ⚠️ "vài phút" là **ước lượng** từ việc giảm số lệnh — phải **đo lại khi cắm board**.
> Logic đã verify bit-exact trên sim (§7) nhưng chưa chạy HW sau khi sửa.

---

## 6. Các bước chạy demo (khi có board)

1. **Regenerate Qsys** `jtag_system` trong Platform Designer (vì address đổi 5→13-bit) → Generate HDL.
   - Đã làm + kiểm khớp: `avs_address width=13`, `addressSpan=8192`, `baseAddress=0x0000`,
     `qsys-generate succeeded`.
2. **Compile** `jtag_top` → `.sof`.
3. **Program** `.sof` vào DE10-Standard (Quartus Programmer, device 5CSXFC6D6).
4. **Chạy**:
   ```
   cd hardware/fpga/soc
   system-console --script=ecg_jtag_console.tcl
   ```
5. Đọc accuracy trên Console + file `ecg_jtag_<timestamp>.log`.

---

## 7. Verification (sim, không cần board)

Cả hai path bit-exact với Python golden (Questa):

| Testbench | Path nạp | Kết quả |
|---|---|---|
| `tb_top.v` (legacy, 13-bit widened) | byte-at-a-time (3 write/byte) | 10 PASS/0 FAIL, **21/21 bit-exact, max\|diff\|=0** |
| `tb_top_window.v` (burst) | DATA WINDOW (1 write/byte) | 10 PASS/0 FAIL, **21/21 bit-exact, max\|diff\|=0** |

→ Window path cho kết quả giống hệt byte path; map legacy không bị phá.
Run: `run_tb_top.do`, `run_tb_top_window.do` (questa).

Kết quả on-board lần demo trước (byte path, trước khi sửa): **94.27% (1004/1065)** — khớp Python 94.65%.

---

## 8. File liên quan

| File | Vai trò |
|---|---|
| `soc/jtag_top.v` | Top synth: PLL 50→100MHz + reset + wrap jtag_system |
| `soc/jtag_top.sdc` | Constraint: 50MHz osc, derive_pll, false-path KEY0 |
| `soc/jtag_system.qsys` | Qsys system (nguồn; generate dir bị .gitignore) |
| `soc/ecg_core_hw.tcl` | Component def Avalon slave 13-bit, addressSpan 8192 |
| `soc/ecg_jtag_console.tcl` | Host driver PC (load/inference/log) |
| `../../RTL/avalon_slave.v` | Bus adapter: legacy regs + DATA WINDOW |
| `../../RTL/ecg_accelerator_top.v` | Wrap avalon_slave + ecg_core (13-bit address) |

> So sánh với variant UART (thay JTAG, ship 2500 byte/sample qua serial) và variant Nios V/m
> (RISC-V soft-core on-chip) — xem `PROJECT.md` / memory Phase D.
