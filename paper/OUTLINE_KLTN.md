# OUTLINE KHOÁ LUẬN TỐT NGHIỆP
## Thiết kế CNN Accelerator cho phân loại rối loạn nhịp tim ECG trên FPGA Intel Cyclone V

> Cập nhật 2026-07-27: cấu trúc mới 5 chương — gộp thiết kế SW + HW vào Chương 3,
> tách kết quả/đánh giá ra Chương 4. Trọng tâm phần cứng.

**Trang đầu (số La Mã):** Bìa · Lời cảm ơn · Lời cam kết · Tóm tắt · Abstract ·
Mục lục · Danh sách chữ viết tắt · Danh sách hình · Danh sách bảng.

---

## CHƯƠNG 1: GIỚI THIỆU
- **1.1. Bối cảnh và giới thiệu đề tài** — rối loạn nhịp & đột tử tim; nhu cầu monitoring
  liên tục wearable/edge; định hướng lightweight INT8 accelerator trên FPGA.
- **1.2. Các nghiên cứu về CNN ứng dụng phân loại nhịp tim** — tổng quan CNN 1D/2D cho ECG,
  các accelerator FPGA tiền nhiệm (tham chiếu Liu 2023), khoảng trống nghiên cứu.
- **1.3. Công cụ và board sử dụng** — PyTorch, Quartus 25.1 / Questa, board DE10-Standard
  (Cyclone V 5CSXFC6D6F31C6), giao tiếp JTAG-to-Avalon.
- **1.4. Cấu trúc khoá luận**

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT
- **2.1. Tín hiệu điện tim và tập dữ liệu nhịp tim**
  - 2.1.1. Tín hiệu điện tim và ý nghĩa lâm sàng — sóng P-QRS-T, 4 nhóm nhịp AFIB/GSVT/SB/SR
  - 2.1.2. Tập dữ liệu huấn luyện Chapman và kiểm tra chéo Georgia — mapping 4-class, split patient-indep 70/15/15
- **2.2. Mạng thần kinh tích chập nhẹ 1 chiều**
  - 2.2.1. Convolution Layer — tích chập 1D, kernel, padding, stride
  - 2.2.2. Pooling Layer — MaxPool, GAP
  - 2.2.3. Hàm kích hoạt — ReLU (chỉ sau Conv4)
  - 2.2.4. Fully-Connected Layer — FC + argmax
  - 2.2.5. Mô hình đề xuất — cấu trúc (4,4,8,8), K=5 — Hình 2.x
- **2.3. Huấn luyện mô hình**
  - 2.3.1. Cơ chế học của mạng tích chập — lan truyền tiến/ngược
  - 2.3.2. Hàm mất mát Cross-Entropy
  - 2.3.3. Thuật toán tối ưu Adam
  - 2.3.4. Cắt tỉa, Taylor/L1 — tỉa kênh có cấu trúc
- **2.4. Lượng tử hoá**
  - 2.4.1. Phương pháp lượng tử QAT — fake-quant, STE
  - 2.4.2. Lượng tử hoá luỹ thừa mũ 2 — nb/w_shift, round-half-up, rescale 0 DSP

## CHƯƠNG 3: THIẾT KẾ VÀ TRIỂN KHAI MẠNG CNN ⭐ TRỌNG TÂM
- **3.1. Quy trình xử lý dữ liệu và huấn luyện mô hình**
  *(Hình: pipeline software tổng quát)*
  - 3.1.1. Xử lý dữ liệu — lead II, resample, z-score, cửa sổ 2500 mẫu
  - 3.1.2. Huấn luyện mô hình — float32 baseline Chapman 94.65%
  - 3.1.3. Lượng tử hoá trọng số — QAT power-of-2 round-half-up, nb per-layer, bias 2^nb
  - 3.1.4. Trích xuất trọng số — flat_weights.hex (580 INT8), layout PE-major, bias INT32 LE
- **3.2. Thiết kế hệ thống mạng CNN**
  *(Hình: diagram tổng lõi CNN — Input SRAM → CP-Engine → Ping-Pong SRAM → GAP/FC/Argmax)*
  - 3.2.1. Thiết kế khối convolution-pool unit — pipeline 5 tầng (S1 MAC → adder tree →
    accumulate → bias+rescale → clamp/ReLU → pool); quy ước Conv4 chuẩn tham chiếu
  - 3.2.2. Thiết kế khối engine unit — 8 PE song song theo kênh, SRW window, padding
    front/back, prefetch, delay-chain (a_d5), weight MUX theo tầng
  - 3.2.3. Thiết kế khối gap-fc-argmax unit — GAP floor division (sum>>2),
    FC nb=0 raw logits, argmax
  - 3.2.4. Thiết kế khối điều khiển và các khối phụ trợ — FSM IDLE→LOAD→CONV1-4→GAP_FC→DONE,
    Input SRAM, Ping-Pong SRAM, weight/bias ROM ($readmemh, topology Chapman hard-code)
  - 3.2.5. Luồng dữ liệu của hệ thống mạng CNN — timing per-layer, heartbeat pool_write,
    overlap reload input N+1
- **3.3. Tích hợp giao tiếp và điều khiển hệ thống**
  - 3.3.1. Tích hợp giao tiếp Avalon-MM Wrapper với lõi CNN — bus adapter (avalon_slave),
    nạp input ECG / đọc result, thanh ghi start/busy/done, memory map
  - 3.3.2. Tích hợp hệ thống với IP JTAG-to-Avalon giao tiếp thông qua PC — jtag_top.v,
    System Console driver

## CHƯƠNG 4: KẾT QUẢ THỰC HIỆN VÀ ĐÁNH GIÁ
- **4.1. Kết quả huấn luyện mô hình**
  - 4.1.1. Kết quả từ tập data huấn luyện — Chapman 94.65%/F1 0.9396, 5-fold (std 0.4–0.9%),
    CM + ROC (macro-AUC 0.967), ablation lượng tử Table 4 (power-of-2 ≈ general, −4 DSP18)
  - 4.1.2. Kết quả từ tập data kiểm tra chéo — Georgia (12-lead balanced): zero-shot /
    linear-probe / full-FT, CM + AUC; phân rã C2==C6 → quant drop 0% (all drop = distribution shift)
- **4.2. Kết quả mô phỏng chức năng**
  - 4.2.1. Kết quả độ chính xác của hệ thống — bit-exact 21/21 checkpoint, max|diff|=0 LSB
  - 4.2.2. Kết quả kiểm tra toàn bộ data test và data kiểm tra chéo — chạy hết test set
    Chapman + Georgia qua RTL, khớp Python; khớp-bit trên 2 bộ trọng số (Chapman 21/21,
    Chapman-Ningbo 7/7 max|diff|=0)
- **4.3. Kết quả thực nghiệm trên board FPGA** — DE10-Standard JTAG: 94.27% (1004/1065)
  khớp Python 94.65%; biến thể Nios V/m & UART
- **4.4. Đánh giá tài nguyên và hiệu năng**
  - 4.4.1. Đánh giá tài nguyên — bản ROM **2120 ALM / 3158 Reg / 28 DSP / 20 M10K**
  - 4.4.2. Đánh giá hiệu năng — latency 52.16 µs / 5216 cycle, throughput ~19,200 inf/s,
    **Fmax 108.46 MHz**; công suất DE10 805.53 mW → **42.02 µJ/inf** (kèm caveat
    confidence Low: Quartus Lite không có SDF cho Cyclone V)
- **4.5. So sánh với các nghiên cứu khác** — bảng SoTA ECG-FPGA (Liu 2023 …), Pareto area↔latency

## CHƯƠNG 5: KẾT LUẬN
- Tổng kết đóng góp (C1 lượng tử power-of-2 round-half-up + ablation; C2 bit-exact;
  C3 cross-dataset Georgia; C4 kiến trúc 8-PE ROM cố định)
- Hạn chế (single-lead, topology cố định, weight bake ROM) và hướng phát triển

**TÀI LIỆU THAM KHẢO** (≥15) · **PHỤ LỤC** (port module, memory map đầy đủ,
hyperparams, CM bổ sung).

---

## Điểm cần chốt trước khi viết số vào bài (theo PAPER_DATA.md §8)
- 🔴 Accuracy: dùng **94.65%** nhất quán toàn bài (bản mới FC-bias, khớp golden RTL).
  Table 4 chú thích "A2=94.37 bản trước FC-bias".
- 🔴 Fmax: bản ROM 8-PE dùng **108.46 MHz** (compile 2026-07-07); board jtag_top +2.202ns@100MHz. KHÔNG dùng 104.85 (số cũ trước ROM build), KHÔNG dùng 137.6 (internal path).
- 🔴 **Phạm vi khóa luận = bản RTL ROM trên DE10-Standard.** KHÔNG đề cập: biến thể SIMD-20,
  cơ chế weight-load qua Avalon (RTL_weight/), port DE0-Nano/Cyclone IV, Elastic-Pareto.
  Năng lượng chỉ dùng số DE10 (805.53 mW / 42.02 µJ) kèm caveat confidence Low.
- 🟢 Cross-dataset Georgia: N chính thức = 5459 (chốt 2026-07-29). Số ningba INT8 zero-shot: acc 0,9300 / F1-macro 0,9151 / AUC 0,9580; float32 0,9291 / 0,9142 / 0,9813. Nguồn: results/georgia/EVAL_TABLES.md.
- 🟠 On-board 1004/1065: hiện chỉ trong memory — cần tìm log/chạy lại để cite được.
