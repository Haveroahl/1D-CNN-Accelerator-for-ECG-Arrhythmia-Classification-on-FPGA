# Phase D add-on — Accuracy on 7-segment displays (demo)

> Bổ sung cho [PHASE_D_STEPS.md](PHASE_D_STEPS.md). Mục tiêu: chạy cả test set qua
> accelerator (HPS driver `ecg_demo.c`), đếm dự đoán đúng, **hiện accuracy % lên
> HEX2/HEX1/HEX0**. HPS ghi số 0..100 qua 1 PIO; RTL `seg7_acc.v` tách BCD + decode.

**Quy ước:** ✋ = thao tác tay Platform Designer/Quartus GUI. 📄 = file đã có sẵn.

---

## Tổng quan luồng dữ liệu
```
HPS ──avs (đã có)──► ecg_core           (nạp ECG, đọc result)
HPS ──PIO 7-bit────► seg7_pio ──acc_pct──► seg7_acc.v ──► HEX2/HEX1/HEX0
       (ghi 0..100)                         (BCD + 7-seg decode, active-low)
```
- Driver `ecg_demo.c`: ghi accuracy hiện thời/cuối ra `SEG7_PIO_BASE`.
- `seg7_acc.v`: 7-bit binary → trăm/chục/đơn vị → 3 HEX (active-low {g,f,e,d,c,b,a}).

## File đã chuẩn bị (📄)
- `seg7_acc.v` — decoder 7-seg, 3 digit (0..100). Đã thêm instance vào `soc_top.v`.
- `soc_top.v` — đã thêm cổng `HEX0/1/2` + instance `u_seg7` + cổng Qsys `.seg7_pio_export(acc_pct)`.
- `../sw/hps/ecg_demo.c` — driver batch, ghi PIO qua `SEG7_PIO_BASE`.

---

## 1. ✋ Thêm PIO vào Platform Designer
1. Mở `soc_system.qsys` (đã tạo ở PHASE_D_STEPS §1-2).
2. IP Catalog → **PIO (Parallel I/O)** → Add.
   - **Width = 7 bits**, **Direction = Output**.
   - Đặt tên instance: **`seg7_pio`**.
3. Connections:
   - `seg7_pio.s1` (Avalon-MM slave) ◄── `hps_0.h2f_lw_axi_master`
   - `seg7_pio.clk` ◄── clock 100 MHz (cùng nguồn `ecg_core.clk`)
   - `seg7_pio.reset` ◄── `hps_0.h2f_reset` (hoặc cùng reset core)
4. Export cổng output: cột **Export** của `seg7_pio.external_connection` → đặt tên **`seg7_pio`**.
   - ⚠️ Qsys sinh tên đầy đủ kiểu `seg7_pio_external_connection_export`. Sau Generate,
     đối chiếu với `.seg7_pio_export(acc_pct)` trong `soc_top.v` và **sửa cho khớp**.

## 2. ✋ Address Map cho PIO
1. Tab **Address Map** → gán base cho `seg7_pio.s1` trong dải của `h2f_lw_axi_master`.
2. ⭐ **GHI LẠI offset này.** Driver: `#define SEG7_PIO_BASE (AVS_BASE + <offset>)`.
   - Mặc định trong `ecg_demo.c` là `AVS_BASE + 0x20`. Nếu gán khác → sửa define đó.
   - ⚠️ Đừng để chồng dải địa chỉ với `ecg_core.avs` (6 word = 0x00..0x14).

## 3. ✋ Generate HDL
1. **Generate HDL → Verilog → Generate**.
2. Mở `soc/synthesis/soc_system.v`, copy tên port PIO thật → đối chiếu `soc_top.v`
   (`.seg7_pio_export(...)`). Sửa tên cho khớp (PHASE_D_STEPS §6.3).

## 4. ✋ Thêm file vào `.qsf`
```tcl
set_global_assignment -name VERILOG_FILE soc/seg7_acc.v
```
(các dòng `soc_top.v`, `soc_system.qsys`, `.qip` đã có ở PHASE_D_STEPS §8.)

## 5. ✋ Pin assignment cho HEX (DE10-Standard)
- Gán `HEX0[6:0]`, `HEX1[6:0]`, `HEX2[6:0]` theo **DE10-Standard pin table** (Terasic).
- Nhanh: copy phần HEX0/1/2 từ `.qsf` của GHRD golden DE10-Standard.
- ⚠️ HEX trên DE10-Standard **active-low** — khớp `seg7_acc.v` (segment on = 0). Không invert thêm.

## 6. Build & chạy driver (📄 `ecg_demo.c`)
```bash
arm-linux-gnueabihf-gcc -O2 -Wall -o ecg_demo ../sw/hps/ecg_demo.c
# copy ecg_demo + demo_data/*.bin lên board
sudo ./ecg_demo chapman_test_ecg_int8.bin chapman_test_labels.bin   # HEX → ~"094"
sudo ./ecg_demo ptbxl_test_ecg_int8.bin   ptbxl_test_labels.bin     # HEX → ~"077"
```
- File `.bin` sinh từ `software/python/export_test_demo.py` → đã có ở `soc/demo_data/`.
- HEX cập nhật accuracy hiện thời mỗi 64 sample, giữ giá trị cuối khi xong.
- Accuracy đầy đủ + confusion matrix in ra UART/console.

---

## Điểm dễ sai
| Chỗ | Triệu chứng | Cách đúng |
|---|---|---|
| `SEG7_PIO_BASE` ≠ offset Qsys | HEX không đổi / số bậy | Khớp define với Address Map §2 |
| Tên export PIO sai | Compile fail (port mismatch) | Đối chiếu `soc_system.v` §3 |
| PIO set Input thay vì Output | HEX tối / không nhận ghi | PIO **Output**, width 7 |
| Quên gán pin HEX | HEX tối | §5 pin table |
| Invert active-low 2 lần | Segment ngược | `seg7_acc.v` đã active-low, pin thẳng |
