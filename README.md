# 1D-CNN Accelerator for ECG Arrhythmia Classification on FPGA

Bộ tăng tốc CNN 1 chiều lượng tử hoá INT8 phân loại rối loạn nhịp tim từ tín hiệu ECG,
triển khai trên Intel Cyclone V (DE10-Standard). Đồng thiết kế thuật toán–phần cứng:
huấn luyện và lượng tử hoá bằng PyTorch, lõi IP viết bằng Verilog, xác thực **bit-exact**
giữa mô hình Python và RTL.

**Khoá luận tốt nghiệp — Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM.**

---

## Tổng quan

| Hạng mục | Giá trị |
|---|---|
| Thiết bị | Intel Cyclone V `5CSXFC6D6F31C6` (DE10-Standard) |
| Độ chính xác | **94,27 %** (Chapman-Ningbo, INT8 khớp-bit) · **94,65 %** (Chapman) |
| Độ trễ suy luận | 52,16 µs (5 216 chu kỳ @ 100 MHz), tất định |
| Thông lượng | ~19 200 suy luận/giây |
| Tần số tối đa | 108,86 MHz |
| Tài nguyên | 2 148 ALM (5 %) · 3 158 thanh ghi · 28 DSP (25 %) · 20 M10K (4 %) |
| Công suất | 536,08 mW → **27,96 µJ/suy luận** ¹ |
| Tham số mô hình | 640 (sau tỉa kênh, từ 1 244) |

¹ Đo bằng PowerPlay với VCD cửa sổ suy luận. Độ tin cậy **thấp**: tệp VCD chỉ phủ
2,5 % nút tổ hợp (giới hạn của mô phỏng mức RTL), nên phần tổ hợp dùng ước lượng mặc
định của công cụ. Xem `PAPER_DATA.md`.

---

## Kiến trúc mô hình

```
Đầu vào 2500 mẫu INT8  (5 giây @ 500 Hz, chuyển đạo II)
  → Conv1 (1→4,  K=5, pad=2)         → MaxPool /5 → 500×4
  → Conv2 (4→4,  K=5, pad=2)         → MaxPool /5 → 100×4
  → Conv3 (4→8,  K=5, pad=2)         → MaxPool /5 →  20×8
  → Conv4 (8→8,  K=5, pad=2, ReLU)   → MaxPool /5 →   4×8
  → GAP → FC (8→4) → Argmax → lớp (0–3)
```

Bốn lớp: AFIB (rung nhĩ) · GSVT (nhịp nhanh trên thất) · SB (nhịp chậm xoang) · SR (nhịp xoang).

ReLU **chỉ** đặt sau Conv4 — Conv1–3 giữ nguyên giá trị âm để bảo toàn đặc trưng hình
thái sóng ECG.

## Lượng tử hoá — power-of-2, làm tròn nửa lên

```
hệ số tỉ lệ:  2^nb, chọn theo  nb = floor(log2(127 / |x|max))
tái tỉ lệ:    out = clamp( (acc + 2^(nb-1)) >> nb , -127, 127 )
```

Việc dùng luỹ thừa của 2 khiến phép tái tỉ lệ chỉ còn **dịch bit + cộng — không tốn DSP**,
khác với tỉ lệ thực cần một bộ nhân cho mỗi lần tái tỉ lệ.

Điểm khác biệt so với công trình trước (Liu 2023 cũng dùng power-of-2 nhưng cắt cụt sàn):

- **Làm tròn nửa lên** thay vì cắt cụt → **+0,38 %** độ chính xác, không tốn thêm DSP.
- **Khảo sát định lượng** power-of-2 vs tỉ lệ thực vs cắt cụt, kèm kiểm định 5-fold.
- **Xác thực khớp-bit** với RTL — 21 điểm kiểm tra mỗi mẫu.

Tham số mỗi lớp: `nb = 8/7/6/7/0`, `w_shift = 6/7/6/7/7`, `input_shift = 2`.

## Kiến trúc phần cứng

```
Avalon-MM → Input SRAM (2500×8b)
                  ↓
      ┌───────────────────────────┐
      │  Conv-Pool Engine          │   8 khối CP song song
      │  K=5, pad=2, stride=1      │   MaxPool K=5, stride=5
      └───────────┬───────────────┘
                  │ Ping-Pong SRAM (giữa các lớp)
                  ↓
      ┌───────────────────────────┐
      │  GAP / FC / Argmax         │
      └───────────┬───────────────┘
                  ↓  result[1:0]
```

Trọng số nạp sẵn vào ROM bằng `$readmemh`; cấu hình mạng cố định trong `cnn_controller.v`.

## Xác thực

| Mức | Kết quả |
|---|---|
| Đơn vị (`tb_cp_block`) | 23/23 đạt |
| Tích hợp lớp (`tb_layer`) | 8/8 đạt |
| Toàn hệ khớp-bit (`tb_top`) | **21/21 đạt** — 21 điểm kiểm tra × 3 mẫu |
| Khớp-bit Chapman-Ningbo | 7/7 đạt, sai khác tuyệt đối lớn nhất = 0 |
| Chạy trên mạch (JTAG-to-Avalon) | 94,27 % — 1 004/1 065 mẫu |

21 điểm kiểm tra gồm: đầu vào, 4 đầu ra pooling, GAP, 4 logit FC và argmax — mỗi điểm
so sánh từng bit giữa mô phỏng Python và mô phỏng RTL.

---

## Cấu trúc thư mục

```
software/python/        Pipeline PyTorch
  model/                Định nghĩa mô hình
  train.py              Huấn luyện float32
  prune_finetune.py     Tỉa kênh có cấu trúc + tinh chỉnh
  quantization/         QAT INT8 (power-of-2, tỉ lệ thực, cắt cụt)
  generate_golden.py    Sinh tệp .mem tham chiếu cho RTL
  export_weights_int8.py

hardware/
  RTL/                  Verilog — bản chính (ROM)
  testbench/            Testbench
  fpga/                 Dự án Quartus, mô phỏng Questa, phần mềm nhúng
  docs/                 Tài liệu thiết kế chi tiết
  System_Design.md      Tài liệu kiến trúc

paper/                  Bản thảo khoá luận, hình vẽ, tài liệu tham khảo
```

## Tái lập kết quả

Yêu cầu: Python 3.10+ với PyTorch · Quartus Prime Lite 25.1 · Questa/ModelSim.

```powershell
# 1. Huấn luyện, tỉa kênh, lượng tử hoá
cd software/python
python train.py
python prune_finetune.py --checkpoint .\results\best_model.pth
python quantization\qat_int8.py --checkpoint .\results\best_model_pruned.pth `
       --output_dir .\results\qat_int8

# 2. Xuất trọng số + sinh tham chiếu khớp-bit
python export_weights_int8.py --checkpoint .\results\qat_int8\model_qat_int8.pth `
       --output_dir .\results\weights_qat_int8
python generate_golden.py

# 3. Mô phỏng RTL (Questa)
cd ..\..\hardware\fpga\simulation\questa
vsim -c -do run_tb_rtl_rom.do

# 4. Tổng hợp (Quartus)
#    Mở hardware/fpga/ecg_accelerator_top.qsf → Compile Design
```

> **Lưu ý:** `$readmemh` đọc tệp `.hex` theo thư mục làm việc của trình mô phỏng
> (`fpga/simulation/questa/`), không phải `hardware/RTL/`. Sau mỗi lần xuất lại trọng số
> phải đồng bộ các tệp `.hex` sang thư mục mô phỏng.

Bộ dữ liệu không kèm trong repo (dung lượng lớn). Chapman-Shaoxing và Ningbo tải từ
PhysioNet; đặt vào `data/` theo mô tả trong `software/README.md`.

---

## Tài liệu

- [`hardware/System_Design.md`](hardware/System_Design.md) — kiến trúc, đường dữ liệu, phân tích timing
- [`hardware/docs/`](hardware/docs/) — pipeline CP, FSM điều khiển, GAP/FC, giao diện bộ nhớ
- [`PAPER_DATA.md`](PAPER_DATA.md) — số liệu chốt dùng cho bài viết
- [`SOTA_TABLE.md`](SOTA_TABLE.md) — so sánh với các công trình liên quan

## Tác giả

Lê Đức — Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM.
