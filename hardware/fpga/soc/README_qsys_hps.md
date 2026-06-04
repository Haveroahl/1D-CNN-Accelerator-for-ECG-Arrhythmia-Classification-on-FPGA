# Đưa ECG Accelerator lên DE10-Standard qua HPS — hướng dẫn Qsys/Platform Designer

> Mục tiêu: cùng RTL đã verify (21/21 bit-exact) chạy on-board, HPS nạp ECG + đọc class.
> File trong thư mục này: `soc_top.v` (Quartus top mới), `ecg_core_hw.tcl` (component Qsys).
>
> **Cảnh báo trung thực:** Toàn bộ hướng dẫn này dựa trên kiến trúc Cyclone V SoC chuẩn +
> RTL của bạn, **chưa dựng và chạy thật trên board**. Port name Qsys sinh ra có thể lệch
> chút so với template trong `soc_top.v` — phải đối chiếu sau khi Generate (xem Bước 6).

---

## Hai con đường — chọn đúng theo mục tiêu

| Mục tiêu | Top synthesis | Cần Qsys? | HPS? |
|---|---|---|---|
| **Phase C** — Fmax + Power + Resource (không cần board chạy) | `ecg_accelerator_top` (như hiện tại) | ❌ | ❌ |
| **Phase D** — chạy thật trên board, HPS nạp data | `soc_top` (file này) | ✅ | ✅ |

**Quan trọng:** project `.qsf` hiện tại đặt `TOP_LEVEL_ENTITY = ecg_accelerator_top`. Cấu hình
đó **đúng cho Phase C** — synth core đứng một mình, `avs_*` thành chân FPGA ảo, TimeQuest +
PowerPlay vẫn ra số đúng cho lõi. Bạn **không cần Qsys để lấy Fmax/power**.

Chỉ khi sang **Phase D** (HPS điều khiển) mới cần đổi top sang `soc_top` + dựng Qsys. Đừng
trộn hai việc — làm Phase C trước với top hiện tại, lấy số cho paper, rồi mới Phase D.

---

## Phase D — các bước dựng hệ HPS (làm theo thứ tự)

### Bước 0 — Backup project
Phase D sẽ đổi top-level. Commit hoặc copy `hardware/fpga/` trước khi bắt đầu.

### Bước 1 — Mở Platform Designer
Quartus → **Tools → Platform Designer**. Lưu system mới tên **`soc_system.qsys`** vào
`hardware/fpga/soc/`.

### Bước 2 — Add HPS (Cyclone V Hard Processor System)
1. IP Catalog → tìm **"Arria V/Cyclone V Hard Processor System"** → add.
2. Tab **FPGA Interfaces**:
   - Bật **Lightweight HPS-to-FPGA interface** (đây là bridge `0xFF20_0000` mà driver C dùng).
   - Tắt HPS-to-FPGA (heavyweight) và FPGA-to-HPS nếu không cần — gọn hơn.
   - Bật **HPS-to-FPGA reset (h2f_reset)** → sẽ export thành `hps_h2f_reset`.
3. Tab **SDRAM**: cấu hình DDR3 đúng preset DE10-Standard.
   > **Lấy preset ở đâu:** mở golden top của DE10-Standard (Terasic CD-ROM /
   > GHRD `DE10_Standard_GHRD`), copy y nguyên cấu hình DDR3 PHY. Sai timing DDR3 =
   > HPS không boot Linux. Đây là phần dễ sai nhất — **đừng tự chế số DDR3**.
4. Tab **Peripheral Pins**: gán đúng theo board (SD/MMC để boot, UART để xem console).

### Bước 3 — Add `ecg_core` component
1. Đảm bảo Qsys thấy `ecg_core_hw.tcl`: Tools → Options → **IP Search Path** → add
   `hardware/fpga/soc/`. (Hoặc copy .tcl vào project root; Qsys auto-scan.)
2. IP Catalog → **Custom/ECG → ECG CNN Accelerator** → add.
3. Nếu báo lỗi đọc .tcl: kiểm tra đường dẫn RTL trong `generate_synth` (đang là `../../RTL`,
   tính từ `soc/` → đúng).

### Bước 4 — Nối dây (Connections)
| From (master/source) | To (slave/sink) | Ý nghĩa |
|---|---|---|
| `hps_0.h2f_lw_axi_master` | `ecg_core.avs` | HPS ghi/đọc 6 register qua lightweight bridge |
| clock source (xem dưới) | `hps_0.h2f_lw_axi_clock` | clock cho lightweight bridge |
| clock source | `ecg_core.clk` | 100 MHz core clock |
| `hps_0.h2f_reset` | `ecg_core.reset_n` | reset core (async-low) |

**Clock:** lightweight bridge và core nên cùng 100 MHz để tránh clock-crossing trong
interconnect. Hai cách:
- **(A) Đơn giản:** export `ecg_core.clk` ra conduit, cấp 100 MHz từ PLL ở `soc_top` (như
  template). Nhớ cũng cấp 100 MHz cho `h2f_lw_axi_clock`.
- **(B) Gọn hơn:** dùng `hps_0.h2f_user0_clock` (HPS xuất ra một clock lập trình được) làm
  nguồn cho cả hai → khỏi cần PLL FPGA. Nhưng phải cấu hình clock đó trong HPS = 100 MHz.

> Khuyên (A) cho lần đầu vì PLL FPGA dễ kiểm soát và khớp `.sdc` 100 MHz có sẵn.

### Bước 5 — Export các conduit ra top
Trong cột **Export** của Qsys, export (double-click để đặt tên):
- `ecg_core.clk`        → `ecg_clk`        (nếu dùng cách A)
- `ecg_core.reset_n`    → `ecg_reset_n`
- `ecg_core.reset_h`    → `ecg_reset_h`    (conduit sync reset)
- `hps_0.h2f_reset`     → `hps_h2f_reset`
- `hps_0.memory`        → `memory`         (DDR3)
- `hps_0` HPS IO conduit → theo board

### Bước 6 — Assign base address & Generate
1. **Address Map** tab: gán base address của `ecg_core.avs`. Ví dụ `0x0000_0000` trong
   không gian lightweight bridge.
   > **Đây là số quyết định địa chỉ trong driver C.** Địa chỉ HPS thấy =
   > `0xFF20_0000 (lwh2f base) + <base bạn gán đây>`. Nếu gán `0x0`, thì
   > `BRIDGE_BASE = 0xFF200000` trong `ecg_classify.c` là đúng. **Ghi lại con số này.**
2. **Generate HDL** → Verilog → Generate. Qsys sinh ra `soc/synthesis/soc_system.v`.
3. **Mở `soc_system.v`, copy port list thật** → đối chiếu với `soc_top.v` template. Sửa
   tên port cho khớp (Qsys hay thêm hậu tố kiểu `_clk_clk`, `_reset_n_reset_n`,
   `memory_mem_a`…). **Bước này bắt buộc — template chỉ là khung.**

### Bước 7 — PLL IP
IP Catalog → **PLL Intel FPGA IP** → tạo `core_pll`: refclk 50 MHz, outclk_0 = 100 MHz,
bật `locked`. Khớp tên instance `core_pll` trong `soc_top.v` (hoặc đổi `soc_top.v` cho khớp).

### Bước 8 — Đổi top-level & add file vào Quartus
Trong `.qsf` (hoặc Project Navigator):
```tcl
set_global_assignment -name TOP_LEVEL_ENTITY soc_top
set_global_assignment -name VERILOG_FILE soc/soc_top.v
set_global_assignment -name QSYS_FILE    soc/soc_system.qsys
# (Qsys-generated .qip cũng cần add — Generate sẽ tạo soc/synthesis/soc_system.qip)
set_global_assignment -name QIP_FILE soc/synthesis/soc_system.qip
```
Giữ nguyên các `VERILOG_FILE` RTL + `HEX_FILE` weight đã có (chúng được component dùng lại).

### Bước 9 — Pin assignment
Gán chân cho `FPGA_CLK1_50`, `KEY0_n`, và toàn bộ HPS DDR3/IO theo **DE10-Standard pin
table** (Terasic). Cách nhanh: import `.qsf` pin-assignment từ GHRD golden top rồi chỉ giữ
phần HPS + clock + key bạn dùng.

### Bước 10 — Compile → .sof + boot
1. Quartus compile → `output_files/soc_top.sof`.
2. Tạo SD card image: U-Boot + kernel + rootfs (dùng image Terasic DE10-Standard) +
   nhúng `soc_top.rbf` (convert từ .sof) để FPGA được nạp lúc boot, hoặc nạp .sof qua JTAG
   sau khi Linux đã chạy.
3. SSH/UART vào Linux trên HPS → chạy `ecg_classify` (xem `sw/hps/`).

---

## SDC cho Phase D

`.sdc` hiện tại (`ecg_accelerator_top_100mhz.sdc`) constrain `clk` và các `avs_*` port như
**top-level ports**. Sang Phase D những thứ đó không còn là top port nữa (chúng nằm trong
Qsys). Cần:
- **Bỏ** `set_input_delay/set_output_delay` trên `avs_*` (Qsys interconnect tự lo, và
  HPS-FPGA timing do Qsys-generated SDC quản).
- **Đổi** `create_clock` từ `[get_ports clk]` sang clock do PLL/Qsys sinh (ví dụ
  `core_pll|outclk_0`), hoặc dùng `derive_pll_clocks`.
- Qsys-generated `soc_system.sdc` (trong .qip) tự xử timing HPS bridge — **đừng constrain lại tay.**

→ Tạo một SDC riêng cho Phase D (`soc_top.sdc`), đừng tái dùng nguyên file Phase C.

---

## Tóm tắt quyết định kiến trúc (trả lời câu hỏi gốc)

- **KHÔNG** bê `ecg_accelerator_top` (kèm avalon_slave) làm top synthesis **cho on-board HPS**.
- **GIỮ** `avalon_slave` bên trong core — nó là address-decoder hợp lệ, đã verify; chỉ cần
  khai báo nhóm `avs_*` là Avalon-MM slave interface để Qsys nối vào HPS (`ecg_core_hw.tcl`).
- Top synthesis on-board = **`soc_top`** bọc Qsys (HPS + interconnect + ecg_core) + PLL + reset glue.
- **Cho Phase C (chỉ lấy số Fmax/power):** giữ nguyên top hiện tại, **không cần Qsys** — nhanh hơn.
```
