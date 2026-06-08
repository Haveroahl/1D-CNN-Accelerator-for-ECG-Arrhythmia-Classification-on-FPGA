# Phase D — Hướng dẫn thao tác tay (đẩy xuống DE10-Standard qua HPS)

> Mục tiêu: chạy core đã verify (21/21 bit-exact) trên board thật, HPS Linux nạp ECG → đọc class.
> File text đã chuẩn bị sẵn (không cần viết lại): `soc_top.v`, `soc_top.sdc`, `ecg_core_hw.tcl`,
> `../sw/hps/ecg_classify.c`. Phần GUI (Platform Designer, pin, PLL IP) **phải làm tay** — dưới đây từng bước.

**Quy ước:** ✋ = thao tác tay GUI bắt buộc. 📄 = file đã có sẵn, chỉ dùng. ⚠️ = chỗ dễ sai.

---

## 0. Tiền đề (đã xong)
- [x] `ecg_core.v` tách khỏi bus, `ecg_accelerator_top` = wrapper (avalon_slave + ecg_core). 21/21 PASS.
- [x] `.qsf` đã add `ecg_core.v`. `ecg_core_hw.tcl` đã có `ecg_core.v` trong fileset.
- [x] `soc_top.v`, `soc_top.sdc`, `ecg_classify.c` đã chuẩn bị.

> **Trước khi bắt đầu:** commit/backup `hardware/fpga/` — Phase D sẽ đổi top-level entity.

---

## 1. ✋ Mở Platform Designer, tạo system
1. Quartus → **Tools → Platform Designer**.
2. Save As → **`soc_system.qsys`** vào `hardware/fpga/soc/`.

## 2. ✋ Add HPS (Cyclone V Hard Processor System)
1. IP Catalog → **"Arria V/Cyclone V Hard Processor System"** → Add.
2. **FPGA Interfaces** tab:
   - ✅ Bật **Lightweight HPS-to-FPGA** (bridge `0xFF20_0000` — driver C dùng đúng base này).
   - ❌ Tắt heavyweight HPS-to-FPGA và FPGA-to-HPS (không cần).
   - ✅ Bật **HPS-to-FPGA reset (h2f_reset)**.
3. ⚠️ **SDRAM** tab — DDR3: **copy y nguyên preset DE10-Standard** từ GHRD golden
   (`DE10_Standard_GHRD`). **Đừng tự chế số DDR3** → sai = HPS không boot Linux.
4. **Peripheral Pins**: bật SD/MMC (boot) + UART (console).

## 3. ✋ Add component `ecg_core` (📄 `ecg_core_hw.tcl` đã sẵn)
1. Tools → Options → **IP Search Path** → add `hardware/fpga/soc/`.
2. IP Catalog → **Custom/ECG → ECG CNN Accelerator** → Add.
   - Component này bọc `ecg_accelerator_top` (chứa avalon_slave + ecg_core). Cổng Avalon = nhóm `avs`.

## 4. ✋ Nối dây (Connections)
| From | To | Ý nghĩa |
|---|---|---|
| `hps_0.h2f_lw_axi_master` | `ecg_core.avs` | HPS đọc/ghi 6 register |
| clock 100 MHz (xem §7) | `hps_0.h2f_lw_axi_clock` | clock bridge |
| clock 100 MHz | `ecg_core.clk` | clock core |
| `hps_0.h2f_reset` | `ecg_core.reset_n` | reset async-low |

> Dùng **cách (A)**: export `ecg_core.clk` ra conduit, PLL ở `soc_top` cấp 100 MHz cho cả core
> và `h2f_lw_axi_clock`. Khớp `soc_top.sdc` (derive_pll_clocks).

## 5. ✋ Export conduit (cột Export, double-click đặt tên — phải KHỚP `soc_top.v`)
| Qsys signal | Export name | Port trong soc_top.v |
|---|---|---|
| `ecg_core.clk` | `ecg_clk` | `.ecg_clk_clk(core_clk)` |
| `ecg_core.reset_n` | `ecg_reset_n` | `.ecg_reset_n_reset_n(core_rst_n)` |
| `ecg_core.reset_h` | `ecg_reset_h` | `.ecg_reset_h_reset(core_rst)` |
| `hps_0.h2f_reset` | `hps_h2f_reset` | `.hps_h2f_reset_reset_n(hps_h2f_rst_n)` |
| `hps_0.memory` | `memory` | `.memory_mem_*` |

## 6. ✋ Address map & Generate
1. **Address Map** tab → gán base của `ecg_core.avs`. Gợi ý **`0x0000_0000`**.
   - ⚠️ ⭐ **GHI LẠI SỐ NÀY.** Driver C: `BRIDGE_BASE 0xFF200000 + <base này>`.
     Nếu gán `0x0` → `ecg_classify.c` (`AVS_BASE 0x0`) đúng nguyên. Gán khác → sửa `AVS_BASE`.
2. **Generate HDL** → Verilog → Generate. Sinh `soc/synthesis/soc_system.v` + `.qip`.
3. ⚠️ Mở `soc/synthesis/soc_system.v`, **copy port list thật** → đối chiếu `soc_top.v` (§5).
   Qsys hay thêm hậu tố (`_clk_clk`, `_reset_n_reset_n`, `memory_mem_a`…). **Sửa cho khớp.**

## 7. ✋ PLL IP
- IP Catalog → **PLL Intel FPGA IP** → tạo **`core_pll`**: refclk 50 MHz, outclk_0 = 100 MHz,
  bật `locked`. Tên instance phải là `core_pll` (khớp `soc_top.v:60`).

## 8. ✋ Đổi top-level & add file vào `.qsf`
Thêm vào `hardware/fpga/ecg_accelerator_top.qsf` (hoặc tạo project Phase D riêng):
```tcl
set_global_assignment -name TOP_LEVEL_ENTITY soc_top
set_global_assignment -name VERILOG_FILE soc/soc_top.v
set_global_assignment -name QSYS_FILE    soc/soc_system.qsys
set_global_assignment -name QIP_FILE     soc/synthesis/soc_system.qip
set_global_assignment -name SDC_FILE     soc/soc_top.sdc
```
- ⚠️ **Bỏ** `SDC_FILE ../ecg_accelerator_top_100mhz.sdc` (chỉ cho Phase C).
- Giữ các `VERILOG_FILE` RTL + `HEX_FILE` weight (component dùng lại; weight vẫn `$readmemh`).
- 📄 `soc_top.sdc` đã viết sẵn: derive_pll_clocks + false_path KEY0_n, không constrain avs_*.

## 9. ✋ Pin assignment
- Gán `FPGA_CLK1_50`, `KEY0_n`, toàn bộ HPS DDR3/IO theo **DE10-Standard pin table** (Terasic).
- Nhanh: import pin `.qsf` từ GHRD golden, giữ phần HPS + clock + KEY dùng.

## 10. ✋ Compile → .sof → board
1. Quartus compile → `output_files/soc_top.sof`.
2. ⚠️ Check Timing Analyzer: core_clk (100 MHz) phải WNS ≥ 0 (Phase C đã +0.508ns slack → ổn).
3. Boot Linux trên HPS (SD image Terasic DE10-Standard). Nạp FPGA:
   - convert `.sof` → `.rbf` nhúng vào boot, **hoặc** nạp `.sof` qua JTAG sau khi Linux chạy.

## 11. 📄 Build & chạy driver (`../sw/hps/ecg_classify.c`)
Trên máy có cross-compiler ARM:
```bash
arm-linux-gnueabihf-gcc -O2 -Wall -o ecg_classify ../sw/hps/ecg_classify.c
```
Tạo file ECG nhị phân (2500 int8 bytes) — xem §12. Copy `ecg_classify` + `sampleN.bin` lên board, SSH/UART:
```bash
sudo ./ecg_classify sample0.bin     # → class = 3 (SR)  v.v.
```
- ⚠️ Driver giả định `AVS_BASE 0x0`. Nếu §6 gán base khác → sửa `#define AVS_BASE` trong `.c`.

## 12. Verify on-board (mục tiêu Phase D)
- 3 sample golden Phase C: result phải = **3 / 1 / 2** (như tb_top: sample0→3, sample1→1, sample2→2).
- Chạy full test set Chapman → accuracy ~94.65% (khớp Python/RTL sim).
- (Load PTB-XL weight runtime = **Phase B01**, cần weight RAM — chưa làm ở Phase D này.)

> **Tạo `sampleN.bin` từ hex sim có sẵn:** `ecg_sample0.hex` (1 byte hex/dòng) →
> dùng script nhỏ đọc hex, ghi raw int8. Hoặc export thẳng từ Python dataset (np.int8 → `.tofile`).

---

## Bản đồ phụ thuộc (cái gì block cái gì)
```
§1-2 HPS + DDR3 preset ──┐
§3 ecg_core component ────┼─► §4 connections ─► §5 export ─► §6 generate ─► §8 qsf ─► §10 compile ─► §11 driver
§7 PLL ──────────────────┘                                  (⚠️ reconcile port names §6.3)
```
Điểm chết người nhất: **§2.3 DDR3 preset** (HPS không boot) và **§6.3 reconcile port** (compile fail).
 