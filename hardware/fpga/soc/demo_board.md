# demo_board.md — Checklist chạy on-board DE10-Standard (Phase D)

> Hai variant test trong ngày: **JTAG-to-Avalon** (full accuracy ~94.65%) và
> **Nios V/m** (demo 3 sample golden). Mọi `.sof` + firmware ĐÃ build sẵn —
> không cần compile lại. Chỉ nạp board và chạy.

---

## 0. Tình trạng artifact (đã verify, không build lại)

| Thứ | Đường dẫn | Trạng thái |
|---|---|---|
| JTAG bitstream | `output_files/jtag_top.sof` | ✅ sẵn |
| Nios V bitstream (firmware nhúng) | `output_files/niosv_top.sof` | ✅ sẵn |
| JTAG test data | `soc/demo_data/chapman_test_ecg_int8.bin` + `chapman_test_labels.bin` (1065 sample) | ✅ sẵn |
| JTAG driver | `soc/ecg_jtag_console.tcl` | ✅ sẵn |
| Nios firmware | nhúng trong `niosv_top.sof` (main.c→app.elf→onchip hex, timestamps nhất quán) | ✅ sẵn |

---

## 1. Chuẩn bị phần cứng / môi trường (làm TRƯỚC)

- [ ] Board **DE10-Standard** + cáp **USB-Blaster II** + adapter nguồn.
- [ ] Cắm board, bật nguồn, cắm USB-Blaster vào PC.
- [ ] Driver USB-Blaster II nhận board — verify bằng:
  ```
  jtagconfig
  ```
  Phải liệt kê được TAP (vd `1) USB-BlasterII ... 5CSXFC6D6`). Nếu trống → cài lại driver.
- [ ] Quartus Programmer mở được và thấy board.
- [ ] `system-console` gọi được (JTAG variant) — nằm ở
  `D:\altera_lite\25.1std\quartus\sysconsole\bin\`.
- [ ] `juart-terminal` gọi được (Nios variant) — trong niosv toolchain / PATH.

---

## 2. Variant A — JTAG-to-Avalon (full accuracy)

**Mục tiêu:** quét cả 1065 sample Chapman trên silicon → in accuracy ~94.65%.

1. **Nạp bitstream**
   - Quartus → **Programmer** → Add File → `output_files/jtag_top.sof` → **Start**.
   - (Volatile — mất khi tắt nguồn, đủ cho demo.)

2. **Quick-check 3 sample trước** (sanity, ~giây)
   - Mở `soc/ecg_jtag_console.tcl`, đặt tạm:
     ```tcl
     set ::MAX_SAMPLES 3
     ```
   - Chạy (⚠️ PHẢI từ thư mục `hardware/fpga/soc/` — script đọc `demo_data/` bằng path tương đối):
     ```
     cd hardware/fpga/soc
     system-console --script=ecg_jtag_console.tcl
     ```
   - Kỳ vọng: sample0→pred=3, sample1→pred=1, sample2→pred=2 (khớp golden tb_top).

3. **Full test set**
   - Đặt lại `set ::MAX_SAMPLES 0` (0 = tất cả).
   - Chạy lại lệnh trên.
   - ⚠️ Nạp qua JTAG là 3 ghi/byte × 2500 × 1065 → **chậm vài phút–chục phút**. Bình thường.
   - Kỳ vọng dòng cuối: `Accuracy: ~1008/1065 = ~94.6%`.

---

## 3. Variant B — Nios V/m (demo SoC, 3 sample)

**Mục tiêu:** chứng minh CPU RISC-V on-chip điều khiển accelerator standalone.

1. **Nạp bitstream** (firmware Nios đã nhúng sẵn)
   - Programmer → `output_files/niosv_top.sof` → **Start**.

2. **Đọc kết quả qua JTAG UART**
   ```
   juart-terminal
   ```
   - Kỳ vọng in ra:
     ```
     === ECG CNN accelerator on Nios V/m (Phase D) ===
     ECG_BASE = 0x000a0040
     sample 0 : pred=3 golden=3 OK
     sample 1 : pred=1 golden=1 OK
     sample 2 : pred=2 golden=2 OK
     Result: 3/3 match golden
     ```

3. **(Tùy chọn) Nạp nóng firmware mới mà KHÔNG re-program FPGA**
   - Chỉ cần nếu sửa `main.c` rồi build lại — không cần cho demo mặc định:
     ```
     niosv-download -g sw/niosv/app/build/app.elf
     juart-terminal
     ```

---

## 4. Điểm dễ sai (đọc trước khi bực)

1. **`system-console` chạy SAI thư mục** → "file not found demo_data/...". Phải `cd hardware/fpga/soc` trước.
2. **`jtagconfig` trống** → driver USB-Blaster II chưa cài / cáp lỏng → board không nhận.
3. **JTAG full set "đứng im"** → không treo, chỉ chậm (JTAG byte-by-byte). Đợi, hoặc giảm `MAX_SAMPLES` để xác nhận đường chạy đúng trước.
4. **Nios không in gì** → kiểm tra `juart-terminal` đã attach đúng cable; thử nhấn KEY0 reset board.
5. **Hai variant KHÔNG nạp đồng thời** → mỗi lần chỉ 1 `.sof` trong FPGA. Nạp lại để đổi variant.

---

## 5. Phân vai (để báo cáo / trả lời GV)

| Variant | Trả lời câu hỏi | Bằng chứng |
|---|---|---|
| **JTAG** | "Core chạy đúng accuracy trên silicon thật?" | Full 1065 sample → ~94.65% (host-driven) |
| **Nios V** | "CPU nhúng điều khiển accelerator standalone được?" | 3 sample golden 3/3 (datapath CPU→Avalon→core) |

> Full-test-set accuracy đầy đủ vốn đã được chứng minh bit-exact bằng RTL sim (21/21
> checkpoint khớp Python). On-board chỉ tái xác nhận trên silicon — không phải để đo lại metric.
