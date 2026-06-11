# Phase D (biến thể JTAG) — chạy core trên DE10-Standard KHÔNG cần HPS

> **Tại sao biến thể này:** Quartus Prime **Lite Edition** (bản đang cài: `25.1std SC Lite`)
> **không có IP "Arria V/Cyclone V Hard Processor System"** — nó bị license-gate sang
> Standard Edition. Nên không dựng được Qsys/HPS như [PHASE_D_STEPS.md](PHASE_D_STEPS.md).
>
> Thay vào đó: drive Avalon-MM slave của core **từ PC qua JTAG** bằng IP
> **JTAG to Avalon Master Bridge** (có sẵn trong Lite) + **System Console**. Tái dùng nguyên
> `avalon_slave + ecg_core` đã verify 21/21 bit-exact — KHÔNG đổi RTL core.

```
PC (System Console, Tcl)
   │ USB-Blaster II / JTAG
   ▼
JTAG to Avalon Master Bridge  (IP Qsys, Lite có sẵn)
   │ Avalon-MM
   ▼
ecg_core.avs  (avalon_slave + ecg_core)
   │
   ▼  result[1:0]
```

**Quy ước:** ✋ = thao tác tay GUI. 📄 = file đã có sẵn. ⚠️ = chỗ dễ sai.

---

## File đã chuẩn bị (📄)
- `jtag_top.v`        — top-level mới (PLL 50→100 + reset glue + Qsys `jtag_system`). KHÔNG có DDR3/HPS.
- `jtag_top.sdc`      — SDC: 50MHz osc + derive_pll_clocks + false_path KEY0_n.
- `ecg_core_hw.tcl`   — component Qsys bọc `ecg_accelerator_top` (đã sửa `package require qsys`).
- `ecg_jtag_console.tcl` — script System Console: nạp ECG, chạy inference, tính accuracy.

---

## 0. Tiền đề
- [x] `ecg_core.v` tách, `ecg_accelerator_top` = avalon_slave + ecg_core. 21/21 PASS.
- [x] `ecg_core_hw.tcl` đã sửa version 21.1 → `package require qsys`.
- [ ] Backup `hardware/fpga/` trước khi đổi top-level.

---

## 1. ✋ Tạo Qsys system mới
1. Quartus (project `ecg_accelerator_top.qpf` đã mở) → **Tools → Platform Designer**.
2. Save As → **`jtag_system.qsys`** vào `hardware/fpga/soc/`.

## 2. ✋ Add JTAG to Avalon Master Bridge
1. IP Catalog → gõ **`JTAG to Avalon`** → chọn **"JTAG to Avalon Master Bridge"** → Add.
   (Đây là IP `altera_jtag_avalon_master`, Lite CÓ — khác với HPS bị ẩn.)
2. Không cần chỉnh tham số → Finish.

## 3. ✋ Add Clock + Reset bridge cho conduit
> Cần nguồn clock/reset để export ra `jtag_top.v`. Dùng PLL ở top (ngoài Qsys), nên trong
> Qsys chỉ cần **clock conduit vào** + **reset conduit vào**.
1. IP Catalog → **Clock Bridge Intel FPGA IP** (`altera_clock_bridge`) → Add → tên `clk_in`.
2. IP Catalog → **Reset Bridge Intel FPGA IP** (`altera_reset_bridge`) → Add → tên `rst_in`
   - đặt **Synchronous edges = None**, **Active low = ON** (cho reset_n core).

## 4. ✋ Add component `ecg_core`
1. Tools → Options → **IP Search Path** → Add `hardware/fpga/soc/` (nếu chưa).
2. IP Catalog → **Custom/ECG → ECG CNN Accelerator** → Add.

## 5. ✋ Nối dây (Connections)
| From | To | Ý nghĩa |
|---|---|---|
| `jtag_master.master` | `ecg_core.avs` | PC ghi/đọc 6 register qua JTAG |
| `clk_in.out_clk` | `jtag_master.clk` | clock cho JTAG master |
| `clk_in.out_clk` | `ecg_core.clk` | clock core (100 MHz) |
| `clk_in.out_clk` | `rst_in.clk` | clock cho reset bridge |
| `rst_in.out_reset` | `jtag_master.clk_reset` | reset JTAG master |
| `rst_in.out_reset` | `ecg_core.reset_n` | reset core (async-low) |

> `ecg_core.reset_h` (conduit sync-high) — export riêng ra top (xem §6). Không nối trong Qsys.

## 6. ✋ Export conduit (cột Export — tên phải KHỚP `jtag_top.v`)
| Qsys signal | Export name | Port trong jtag_top.v |
|---|---|---|
| `clk_in.in_clk`       | `ecg_clk`     | `.ecg_clk_clk(core_clk)` |
| `rst_in.in_reset`     | `ecg_reset_n` | `.ecg_reset_n_reset_n(core_rst_n)` |
| `ecg_core.reset_h`    | `ecg_reset_h` | `.ecg_reset_h_reset(core_rst)` |

## 7. ✋ Address Map & Generate
1. **Address Map** tab → base của `ecg_core.avs` = **`0x0000_0000`**.
   - Script `ecg_jtag_console.tcl` dùng byte offset 0x00..0x14 từ base master → khớp base 0x0.
2. **Generate HDL** → Verilog → Generate. Sinh `soc/synthesis/jtag_system.v` + `.qip`.
3. ⚠️ Mở `soc/synthesis/jtag_system.v`, **copy port list thật** → đối chiếu §6 với `jtag_top.v`.
   Qsys hay thêm hậu tố (`_clk_clk`, `_reset_reset`, `_reset_n_reset_n`). **Sửa cho khớp.**

## 8. ✋ PLL IP
- IP Catalog (trong Quartus, KHÔNG trong Qsys) → **PLL Intel FPGA IP** → tạo **`core_pll`**:
  refclk **50 MHz**, outclk_0 = **100 MHz**, bật **`locked`**. Tên instance = `core_pll`
  (khớp `jtag_top.v`).

## 9. ✋ Đổi top-level & add file vào `.qsf`
Thêm vào `hardware/fpga/ecg_accelerator_top.qsf`:
```tcl
set_global_assignment -name TOP_LEVEL_ENTITY jtag_top
set_global_assignment -name VERILOG_FILE soc/jtag_top.v
set_global_assignment -name QSYS_FILE    soc/jtag_system.qsys
set_global_assignment -name QIP_FILE     soc/synthesis/jtag_system.qip
set_global_assignment -name SDC_FILE     soc/jtag_top.sdc
```
- ⚠️ **Bỏ** `SDC_FILE ../ecg_accelerator_top_100mhz.sdc` (Phase C dùng top khác).
- Giữ các `VERILOG_FILE` RTL + `HEX_FILE` weight (component dùng lại; weight vẫn `$readmemh`).
- PLL IP (`core_pll`) tự add `.qip` của nó khi tạo — kiểm tra `.qsf` có dòng QIP của PLL.

## 10. ✋ Pin assignment (cực ít — không có HPS/DDR3)
Chỉ cần 2 chân:
- `FPGA_CLK1_50`  → pin oscillator 50MHz (DE10-Standard pin table Terasic).
- `KEY0_n`        → pin push-button KEY0.
> JTAG dùng TAP riêng (USB-Blaster), KHÔNG cần gán pin FPGA.

## 11. ✋ Compile → .sof
1. Quartus compile → `output_files/jtag_top.sof` (hoặc `ecg_accelerator_top.sof` tùy tên project).
2. ⚠️ Timing Analyzer: core_clk 100MHz WNS ≥ 0 (Phase C đã +0.508ns slack → ổn).

## 12. ✋ Nạp board & chạy
1. **Programmer** → nạp `.sof` qua USB-Blaster II vào FPGA (volatile, đủ cho demo).
2. Mở **System Console**: Quartus → **Tools → System Debugging Tools → System Console**.
   - Hoặc shell: `system-console --script=ecg_jtag_console.tcl` (chạy từ `hardware/fpga/soc/`).
3. Script tự: mở JTAG master → nạp từng sample ECG → start → poll done → đọc result → tính accuracy.

## 13. Verify (mục tiêu Phase D)
- 3 sample đầu Chapman: result phải khớp golden (sample0→3, sample1→1, sample2→2 như tb_top).
  - Quick check: set `set ::MAX_SAMPLES 3` trong `ecg_jtag_console.tcl`.
- Full test set Chapman (1065 samples): accuracy ~**94.65%** (khớp Python/RTL sim).
  - ⚠️ Nạp qua JTAG là 3 ghi/byte × 2500 × 1065 → **chậm** (vài phút – chục phút). Chấp nhận
    được cho demo on-board; không phải đường truyền tốc độ cao.

---

## So với HPS (PHASE_D_STEPS.md) — khác gì?
| | HPS (Standard edition) | JTAG (Lite edition, file này) |
|---|---|---|
| Cần Quartus | Standard | **Lite OK** |
| IP HPS | Có | Không cần |
| DDR3 preset | Bắt buộc (dễ sai) | Không cần |
| Nạp data | HPS Linux driver C | PC System Console Tcl |
| Latency đo on-board | ARM cycle counter | Không (đã có từ sim 5216cy) |
| Tốc độ nạp | Nhanh (bridge) | Chậm (JTAG, OK cho demo) |
| Đủ cho paper C2/C4 | Có | **Có** (chứng minh core chạy thật trên silicon) |

---

## ⚠️ Điểm dễ sai
1. **§7.3 reconcile port name** — Qsys đổi tên export, phải sửa `jtag_top.v` cho khớp.
2. **Reset bridge polarity** — `rst_in` phải Active-low để khớp `ecg_core.reset_n` (async-low).
3. **Base address ≠ 0** → sửa offset trong `ecg_jtag_console.tcl` (cộng base vào A_DIN..A_RES).
4. **PLL tên instance** phải đúng `core_pll`, có port `refclk/rst/outclk_0/locked`.
