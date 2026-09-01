87# Cơ sở lý thuyết — Kiến trúc CNN và Kiến trúc Phần cứng Tăng tốc

> **Phạm vi.** Chương này trình bày cơ sở lý thuyết ở **hai tầng** và lý do lựa chọn ở mỗi tầng:
> 1. **Tầng mô hình (model/network):** các họ kiến trúc CNN dùng cho phân loại ECG → vì sao chọn **1D-CNN nông (4 lớp)**.
> 2. **Tầng phần cứng (accelerator/dataflow):** các kiểu kiến trúc tăng tốc CNN trên FPGA → vì sao chọn **time-multiplexed 8-PE channel-parallel**.
>
> Hai tầng độc lập: cùng một mô hình CNN có thể ánh xạ lên nhiều kiểu dataflow khác nhau. Việc tách bạch giúp lập luận lựa chọn rõ ràng và phòng thủ được trước phản biện.
>
> **Nguyên tắc số liệu:** mọi con số trích trong chương này lấy từ kết quả đo thật của đề tài (`PAPER_DATA.md`, các báo cáo synthesis/power) hoặc từ tài liệu tham khảo có trích dẫn. Các kết luận "chọn X vì Y" đều neo vào số đo, không phải phán đoán định tính.

---

## 1. Tầng mô hình — Các họ kiến trúc CNN cho phân loại ECG

### 1.1. Bối cảnh bài toán

Tín hiệu ECG là chuỗi thời gian 1 chiều (1D time-series). Bài toán phân loại rối loạn nhịp (arrhythmia) yêu cầu mô hình học được **hình thái sóng (morphology)** — dạng phức bộ QRS, sóng P, sóng T — và đôi khi cả **nhịp độ (rhythm/rate)** giữa các phức bộ. Có hai cách biểu diễn đầu vào dẫn tới hai họ kiến trúc CNN khác nhau:

- **Biểu diễn 1D (raw waveform):** giữ tín hiệu ở dạng chuỗi mẫu → dùng **Conv1D**.
- **Biểu diễn 2D (beat-image / time–frequency):** biến đổi đoạn nhịp thành ảnh (ví dụ spectrogram, scalogram, hoặc xếp các nhịp thành ma trận) → dùng **Conv2D**.

### 1.2. Các họ kiến trúc CNN tiêu biểu

**(a) 2D-CNN trên "beat-image".** Chuyển mỗi nhịp tim thành ảnh rồi áp dụng CNN ảnh tiêu chuẩn (thường trên bộ dữ liệu MIT-BIH, 5 lớp AAMI). Đạt độ chính xác rất cao (97–99%) nhờ tận dụng được kho kiến trúc ảnh trưởng thành, nhưng **chi phí tính toán lớn**: ma trận 2D + kernel 2D làm số phép nhân-cộng (MAC) tăng theo bậc hai, dẫn tới độ trễ mức mili-giây và công suất hàng trăm mW đến vài W. Không phù hợp thiết bị đeo chạy liên tục.

**(b) 1D-CNN sâu / ResNet-1D.** Conv1D nhiều lớp, có residual connection, chạy thẳng trên waveform. Cân bằng tốt giữa độ chính xác và chi phí so với 2D, nhưng số tham số và MAC vẫn lớn (hàng chục nghìn → hàng triệu tham số) nếu sâu/rộng.

**(c) CNN–RNN/LSTM hybrid.** CNN trích đặc trưng hình thái + RNN/LSTM mô hình hóa phụ thuộc thời gian dài. Độ chính xác cao trên các tác vụ cần ngữ cảnh dài, nhưng **LSTM rất tốn phần cứng**: phép nhân ma trận hồi tiếp + hàm phi tuyến (sigmoid/tanh) khó lượng tử hóa và khó ánh xạ lên FPGA nhỏ, độ trễ phụ thuộc độ dài chuỗi.

**(d) Depthwise-separable CNN (MobileNet-style).** Tách convolution thành depthwise + pointwise để giảm MAC. Hiệu quả ở **mô hình lớn** (hàng trăm kênh), nơi phần pointwise chi phối.

**(e) Attention / Transformer cho ECG.** Mới và mạnh về độ chính xác, nhưng chi phí self-attention (bậc hai theo độ dài chuỗi) và độ phức tạp lượng tử hóa khiến chưa thực tiễn cho lớp thiết bị đeo siêu nhỏ ở thời điểm hiện tại.

### 1.3. Bảng so sánh các họ mô hình (định hướng wearable)

| Họ kiến trúc | Độ chính xác điển hình | Chi phí MAC / tham số | Khả năng lượng tử hóa INT8 | Phù hợp wearable FPGA nhỏ |
|---|---|---|---|---|
| 2D-CNN (beat-image) | Rất cao (97–99%) | Cao (kernel 2D) | Tốt | ✗ (latency ms, công suất cao) |
| ResNet-1D sâu | Cao | Trung bình–cao | Tốt | △ (cần cắt tỉa mạnh) |
| CNN + LSTM | Cao | Cao + phi tuyến hồi tiếp | Khó (sigmoid/tanh) | ✗ |
| Depthwise-separable | Cao **chỉ khi mô hình lớn** | Thấp ở mô hình lớn | **Kém ở mô hình nhỏ** (xem 1.5) | ✗ với tiny-CNN |
| Attention/Transformer | Rất cao | Rất cao (bậc hai) | Khó | ✗ (chưa thực tiễn) |
| **1D-CNN nông (đề tài)** | **94.65%** | **Rất thấp (112k MAC, 640 tham số)** | **Rất tốt (INT8 == float)** | **✓** |
 
### 1.4. Lý do chọn 1D-CNN nông 4 lớp

1. **Biểu diễn tự nhiên + chi phí thấp.** Conv1D xử lý trực tiếp waveform, không cần tiền xử lý thành ảnh; số MAC chỉ ~112k (so với 2D-CNN cao hơn nhiều bậc), cho độ trễ **52,16 µs/inference** — phù hợp giám sát liên tục.
2. **Tham số tối thiểu là một tính năng, không phải hạn chế.** Mô hình sau cắt tỉa chỉ **640 tham số / 580 trọng số INT8**, vừa vặn lưu trên-chip, giảm năng lượng truy xuất bộ nhớ — yếu tố quyết định cho thiết bị đeo.
3. **Lượng tử hóa INT8 không mất độ chính xác.** Với sơ đồ power-of-2 của đề tài, INT8 đạt **đúng bằng** float (94,65%, bit-exact), khác với các họ DW-separable nơi INT8 power-of-2 sụt 3,6–9,5 điểm phần trăm (xem 1.5).
4. **So sánh trực tiếp được với tài liệu.** Cùng bài toán Chapman 4 lớp, cùng cấu trúc 4 conv + 4 pool như Liu *et al.* 2023 — cho phép đối chứng công bằng (đề tài 94,65% > Liu INT8 92,95%).

### 1.5. Các hướng mô hình đã thử và loại bỏ (ablation âm — củng cố lựa chọn)

Đề tài đã prototype và **đo thật** các hướng mô hình thay thế, kết luận đều loại bỏ. Việc trình bày các ablation âm này làm lập luận chọn 1D-CNN nông vững hơn:

- **Depthwise-separable CNN — NO-GO.** Ở mô hình nhỏ (≤16 kênh), tách DW không tiết kiệm đáng kể mà làm INT8 power-of-2 sụt 3,6–9,5 pp; phải nới tới (8,16,16,16) mới chạm 94,27% float nhưng khi đó trọng số ~1,5× và MAC ~2× baseline → mất sạch điểm bán "ít trọng số". Per-channel power-of-2 còn tệ hơn (44%) vì phá tính cộng-được giữa kênh ở pointwise conv.
- **Lượng tử hóa INT4 — NO-GO.** Mất ~19% độ chính xác kể cả với general-scale (trần 75,6%); lớp AFIB sụp đổ. INT8 là điểm ngọt.
- **Chuyển đổi sang SNN (Spiking NN) — NO-GO.** Chuyển ANN→SNN ngây thơ chỉ đạt ~58% (do maxpool + giá trị có dấu + feature-map nhỏ); muốn dùng SNN phải huấn luyện lại từ đầu.

> **Kết luận tầng mô hình:** 1D-CNN nông 4 lớp + cắt tỉa cấu trúc là lựa chọn Pareto-tối ưu cho wearable: đủ chính xác (94,65%), tham số tối thiểu, lượng tử hóa INT8 không mất mát, và ánh xạ phần cứng đơn giản.

---

## 2. Tầng phần cứng — Các kiến trúc tăng tốc CNN trên FPGA

### 2.1. Khái niệm dataflow

**Dataflow** xác định cách các vòng lặp lồng nhau của một lớp CNN (theo kênh vào, kênh ra, vị trí không gian) được **ánh xạ lên các phần tử tính toán (PE)** và bộ nhớ trên-chip. Cùng một mô hình có thể chạy trên nhiều dataflow; lựa chọn dataflow quyết định trực tiếp tới tài nguyên (ALM, DSP, register), độ trễ và năng lượng. Theo phân loại của Sze *et al.* [35], các dataflow chính là:

- **Weight-stationary:** mỗi PE giữ cố định trọng số, activation chảy qua.
- **Output-stationary:** mỗi PE giữ cố định một output, tích lũy partial-sum tại chỗ.
- **Input-stationary:** giữ cố định activation đầu vào.

### 2.2. Hai chiến lược ánh xạ tài nguyên

Quan trọng hơn cả với FPGA nhỏ là **mức độ chia sẻ phần cứng** giữa các lớp:

Theo phân loại toolflow của Venieris *et al.* [42], accelerator CNN trên FPGA chia thành hai họ chính — *streaming architecture* (mỗi lớp phần cứng riêng) và *single-computation-engine* (một engine dùng chung) — đây là khung lập luận để đối chứng lựa chọn:

**(a) Fully-mapped / streaming (spatial / unrolled).** Mỗi lớp có **phần cứng riêng**, không chia sẻ; toàn mạng trải phẳng trên silicon. Cho thông lượng cao nhất (có thể pipeline nhiều ảnh) nhưng **tốn tài nguyên khổng lồ**. Đây là kiến trúc của Liu *et al.* 2023 [18]: vì mỗi phép nhân 8-bit được ánh xạ phần lớn vào soft-logic, họ dùng tới **ALM 51%, Register 86%** của thiết bị. (Streaming per-layer-tailored cũng là triết lý của các khung tự động hoá ánh xạ CNN như FINN/fpgaConvNet [28].)

**(b) Time-multiplexed (folded / single-engine).** Một engine tính toán dùng chung được **tái sử dụng tuần tự** qua các lớp/vị trí. Một bộ PE giới hạn được time-multiplex: tính xong một output thì PE được giao output kế tiếp. Đánh đổi: **độ trễ tăng** (II = latency, không pipeline được nhiều ảnh) để **giảm mạnh tài nguyên**. Đây là họ single-computation-engine (DPU/FINN/fpgaConvNet [28]) và là lựa chọn của đề tài.

**(c) Systolic array.** Mảng PE 2D truyền dữ liệu theo nhịp, phổ biến cho GEMM/conv lớn (TPU-style). Hiệu quả ở mô hình lớn và batch lớn; với tiny-CNN 1D (kênh ≤8, 4 lớp) thì mảng systolic bị **lãng phí PE** và overhead điều khiển không tương xứng.

### 2.3. Bảng so sánh các kiến trúc accelerator

| Kiến trúc | Tài nguyên | Độ trễ / thông lượng | Phù hợp tiny-1D-CNN wearable | Ví dụ |
|---|---|---|---|---|
| Fully-mapped (spatial) | **Rất cao** (ALM 51%, Reg 86%) | Thấp nhất / cao nhất | △ (thừa tài nguyên) | Liu 2023 |
| **Time-multiplexed (đề tài)** | **Rất thấp** (ALM 5%, Reg ~4,8k) | 52,16 µs, đủ dùng | **✓** | DPU/FINN/fpgaConvNet |
| Systolic array | Trung bình–cao | Cao ở batch lớn | ✗ (lãng phí PE ở model nhỏ) | TPU-style |

> **Số đối chứng then chốt (đề tài vs Liu 2023, cùng Cyclone V, cùng bài toán):** DSP 28 vs 44; ALM **5% vs 51%** (ít hơn ~10×); Register **~4 800 vs ~72 000** (ít hơn ~15×). Đề tài chậm hơn (52 µs vs 66 µs ở 100 vs 50 MHz) nhưng **footprint nhỏ hơn áp đảo** — đúng định hướng wearable.

### 2.4. Lý do chọn time-multiplexed 8-PE channel-parallel

1. **Footprint quyết định tính khả thi wearable.** Tiny-1D-CNN không cần thông lượng cao (ECG ~500 Hz, một inference mỗi vài giây là dư). Đổi độ trễ (vẫn chỉ 52 µs) lấy giảm tài nguyên ~10–15× là đánh đổi đúng. Time-multiplexed cho ALM 5% vs 51% của fully-mapped.
2. **8-PE khớp tự nhiên với chiều kênh.** Lớp lớn nhất (Conv4) có 8 kênh ra → 8 PE channel-parallel xử lý trọn một vị trí mỗi lần lặp, các lớp nhỏ hơn là tập con. Không lãng phí PE như systolic array.
3. **Lượng tử hóa power-of-2 → 0 DSP cho rescale.** Bước rescale chỉ là dịch bit + cộng (round-half-up), loại bỏ multiplier mà general-scale INT8 cần. Điều này gắn trực tiếp với năng lượng trên Cyclone V, nơi khối DSP chiếm phần lớn dynamic power.
4. **Trọng số nạp sẵn trong ROM (`$readmemh`).** Toàn bộ 580 hệ số INT8 của Conv cùng bias được bake vào bitstream, nên lõi không cần cổng bus ghi trọng số — loại bỏ logic giải mã địa chỉ ghi và thanh ghi đệm, giữ tài nguyên ở mức tối thiểu (2.120 ALM, 5 % device). Đánh đổi: đổi bộ trọng số đòi hỏi tổng hợp lại bitstream, chấp nhận được vì mô hình đã cố định sau khi huấn luyện.

### 2.5. Vì sao chọn song song theo kênh thay vì theo vị trí

Bên trong khối tính toán, có hai hướng song song hóa: theo **kênh** (mỗi PE lo một kênh đầu ra) và theo **vị trí** (nhiều PE cùng tính các điểm ra liền kề trên một kênh). Đề tài chọn hướng theo kênh với 8 PE, vì:

- Mô hình sau tỉa có số kênh ra nhỏ và cố định ở bội số của 2 (4-4-8-8) → ánh xạ trực tiếp một PE cho một kênh, bộ điều khiển giữ được dạng máy trạng thái phẳng.
- Độ trễ đạt được (52,16 µs) đã nhanh hơn yêu cầu của bài toán đo ECG liên tục (~1 suy luận mỗi nhịp tim) khoảng bốn bậc độ lớn → rút ngắn thêm độ trễ không mang lại giá trị sử dụng, trong khi số bộ nhân và độ phức tạp điều khiển sẽ tăng đáng kể.
- Ràng buộc thực của bài toán là **tài nguyên và năng lượng**, không phải thông lượng → tối ưu theo hướng dấu chân nhỏ là lựa chọn đúng mục tiêu.

---

## 3. Kết luận chương cơ sở lý thuyết

| Quyết định | Lựa chọn | Căn cứ định lượng |
|---|---|---|
| **Tầng mô hình** | 1D-CNN nông 4 lớp (4-4-8-8) + cắt tỉa | 94,65% / 640 tham số / 112k MAC; INT8 == float; DW/INT4/SNN đều loại bỏ bằng đo thật |
| **Tầng dataflow** | Time-multiplexed 8-PE channel-parallel | ALM 5% vs 51% (Liu); 52 µs đủ dùng (nhanh hơn yêu cầu 4 bậc độ lớn); 8-PE khớp đúng số kênh 4-4-8-8 |
| **Tầng lượng tử hóa** | Power-of-2 QAT round-half-up | 0 DSP rescale; +0,38% vs floor; bit-exact 21 checkpoint |

Cả ba quyết định hội tụ về một mục tiêu: **một IP core ECG đã được kiểm chứng bit-exact, dấu chân tài nguyên tối thiểu, phù hợp giám sát đeo single-lead** — đó là lý do lựa chọn kiến trúc này thay vì các họ mô hình và kiến trúc tăng tốc khác.

---

## Nguồn tham khảo (số [n] khớp `REFERENCES.md`)

Mỗi quyết định kiến trúc trong chương này neo vào tài liệu sau (dùng cho lập luận "vì sao chọn", không phải cơ chế hoạt động — cơ chế nằm ở chương `CO_SO_LY_THUYET_HW`):

- **[18]** Liu *et al.* 2023, *FPGA accelerator for ECG analysis*, Frontiers in Physiology, DOI 10.3389/fphys.2023.1079503 — đối thủ trực tiếp fully-mapped (cùng Chapman, cùng Cyclone V) → dẫn chứng đối chứng tài nguyên (ALM 51% vs 5%).
- **[26]** Sze *et al.*, "Efficient Processing of Deep Neural Networks," *Proc. IEEE* 2017, DOI 10.1109/JPROC.2017.2761740 — phân loại dataflow (weight/output/input-stationary) → §2.1.
- **[28]** Venieris *et al.*, "Toolflows for Mapping CNNs on FPGAs: A Survey," *ACM CSUR* 2018, DOI 10.1145/3186332 — khung phân loại streaming vs single-computation-engine → §2.2.

> **Lưu ý nội bộ (xóa trước khi nộp):** số liệu đề tài lấy từ `PAPER_DATA.md` — accuracy 94,65 (Chapman, bản FC-bias dùng cho golden RTL; không dùng 94,37 cũ) và 94,27 (Chapman-Ningbo INT8 khớp-bit); **Fmax 108,46 MHz** (bản ROM, compile 2026-07-07; KHÔNG dùng 104,85 cũ, KHÔNG dùng 137,6 internal path); ALM 2.120 / Reg 3.158 / DSP 28. Liu: DSP 39%/ALM 51%/Reg 86%/66µs@50MHz/66mW, INT8 acc 92,95%. Ablation âm DW/INT4/SNN từ các memory tương ứng. **Phạm vi khóa luận = bản RTL ROM trên DE10; KHÔNG đề cập SIMD-20, weight-load qua Avalon, hay port DE0-Nano.**
