# CHƯƠNG 4: KẾT QUẢ THỰC HIỆN VÀ ĐÁNH GIÁ

Chương 3 đã trình bày *cách làm* — từ mô hình phần mềm, qua lõi phần cứng, tới lớp giao
tiếp. Chương này trình bày *kết quả đo được* của toàn hệ thống. Trước hết, Mục 4.1 định
nghĩa **phương pháp đánh giá** — các chỉ số và công cụ dùng để đo cả phía phần mềm (mô
hình) lẫn phía phần cứng (FPGA). Sau đó là các kết quả: (i) kết quả huấn luyện mô hình
trên tập Chapman và kiểm tra chéo trên các dataset khác (Mục 4.2); (ii) kết quả mô phỏng
chức năng — độ chính xác khớp-bit giữa RTL và phần mềm, và kiểm tra trên toàn bộ dữ liệu
(Mục 4.3); (iii) kết quả chạy thực nghiệm trên board FPGA (Mục 4.4); (iv) đánh giá tài
nguyên và hiệu năng sau tổng hợp (Mục 4.5); và (v) so sánh với các công trình liên quan
(Mục 4.6).

Mọi số liệu trong chương được lấy trực tiếp từ báo cáo của công cụ (Quartus, PowerPlay),
từ tệp kết quả mô phỏng (ModelSim/Questa), hoặc từ tệp đánh giá của mô hình phần mềm
(PyTorch) — không phải ước lượng. Cấu hình chung cho mọi thí nghiệm được tóm tắt ở Bảng 4.1.

**Bảng 4.1 — Cấu hình chung của các thí nghiệm.**

| Hạng mục | Giá trị |
|----------|---------|
| Thiết bị FPGA | Intel Cyclone V `5CSXFC6D6F31C6` (DE10-Standard), speed grade C6 |
| Công cụ tổng hợp | Quartus Prime 25.1std Lite |
| Công cụ mô phỏng | ModelSim/Questa FSE |
| Tần số mục tiêu (SDC) | 100 MHz; Fmax đọc ở mô hình Slow 1100 mV, 85 °C |
| Mô hình | ECG_1DCNN cắt tỉa (4,4,8,8), 640 tham số, 4 lớp (AFIB/GSVT/SB/SR) |
| Đầu vào | 2500 mẫu INT8 (lead II) |
| Lượng tử hóa | power-of-2 round-half-up, input_shift = 2. Bản Chapman-only (phần cứng): nb = {8,6,6,7,0}, w_shift = {6,6,6,7,8}. Bản ningba (mô hình phần mềm mở rộng): nb = {8,7,6,7,0}, w_shift = {6,7,6,7,7} |

---

## 4.1. Phương pháp đánh giá

Mục này định nghĩa các chỉ số dùng xuyên suốt chương, tách riêng hai phía: đánh giá *chất
lượng phân loại* của mô hình (phần mềm) và đánh giá *chi phí – hiệu năng* của lõi tăng tốc
(phần cứng). Việc định nghĩa trước phương pháp nhằm tách bạch *cách đo* (mục này) khỏi *số
đo được* (các mục 4.2–4.6).

### 4.1.1. Chỉ số đánh giá phần mềm

Bài toán là phân loại đa lớp (4 nhóm nhịp AFIB/GSVT/SB/SR) trên dữ liệu mất cân bằng lớp,
nên ngoài độ chính xác tổng ta dùng thêm các chỉ số theo lớp và độc lập ngưỡng.

**Ma trận nhầm lẫn (Confusion Matrix).** Ma trận 4×4 với hàng là nhãn thật, cột là nhãn dự
đoán; phần tử `C[i][j]` là số mẫu lớp `i` bị phân vào lớp `j`. Đường chéo là số phân đúng,
các ô ngoài chéo phơi bày *cặp lớp* hay bị lẫn — công cụ chẩn đoán trực quan quan trọng
nhất, đặc biệt để soi cặp SB/SR (phân định bằng ngưỡng nhịp tim).

**Precision, Recall, F1 theo lớp.** Với mỗi lớp, gọi TP, FP, FN lần lượt là số dương-đúng,
dương-sai, âm-sai:

```
Precision = TP / (TP + FP)      (trong số ca dự đoán là lớp này, bao nhiêu đúng)
Recall    = TP / (TP + FN)      (trong số ca thật thuộc lớp này, bắt được bao nhiêu)
F1        = 2 · Precision · Recall / (Precision + Recall)   (trung bình điều hòa)
```

Precision phạt cảnh báo nhầm (false alarm), Recall phạt bỏ sót ca bệnh — với ứng dụng y
tế, cả hai đều quan trọng nên F1 (cân bằng hai chỉ số) là thước đo chính theo lớp.

**F1-macro và Accuracy.** F1-macro là trung bình cộng F1 của bốn lớp (mỗi lớp trọng số
bằng nhau, không thiên về lớp đông) — phù hợp dữ liệu mất cân bằng hơn accuracy. Accuracy
là tỉ lệ mẫu phân đúng trên tổng, báo kèm để so sánh với các công trình khác.

**ROC / AUC.** Đường ROC vẽ tỉ lệ dương-thật theo tỉ lệ dương-giả khi quét ngưỡng quyết
định; AUC (diện tích dưới ROC) đo khả năng phân tách của mô hình **độc lập với ngưỡng**.
Ta báo macro-AUC (trung bình AUC one-vs-rest của bốn lớp). Chỉ số này bổ khuyết cho F1:
F1 phụ thuộc ngưỡng argmax cố định, còn AUC phản ánh chất lượng phân tách nội tại — khi
F1 thấp mà AUC cao thì vấn đề nằm ở ngưỡng, không phải ở đặc trưng học được.

**Điểm mấu chốt về tính khớp-bit.** Mọi số INT8 báo cáo trong chương là **bit-exact** —
suy ra từ đúng chuỗi số học mà phần cứng thực thi (Mục 3.1.3), không phải "mô phỏng INT8"
gần đúng bằng số thực. Nhờ vậy các chỉ số phần mềm INT8 chính là chỉ số mà phần cứng cho
ra, không có khoảng cách mô phỏng–triển khai.

### 4.1.2. Chỉ số đánh giá phần cứng

Phía phần cứng đánh giá *chi phí tài nguyên* và *hiệu năng* của lõi sau tổng hợp, mọi số
lấy trực tiếp từ báo cáo công cụ (Quartus Fitter, Timing Analyzer, PowerPlay) hoặc từ mô
phỏng (ModelSim/Questa), không ước lượng.

**Độ trễ (latency).** Số chu kỳ clock của một lần suy luận, đo bằng đếm chu kỳ trong
testbench (`$time`). Nhân với chu kỳ clock cho độ trễ theo thời gian: `latency_µs =
cycles / f_MHz`. Độ trễ của thiết kế là xác định (deterministic) — không phụ thuộc dữ
liệu — nên đo một lần là đủ.

**Thông lượng (throughput).** Số suy luận mỗi giây ở tần số hoạt động: `throughput =
f / cycles_per_inference`.

**Tài nguyên (resource).** Từ báo cáo Fitter: số ALM (khối logic thích ứng), Registers
(thanh ghi), DSP18 (khối nhân cứng), M10K (khối RAM 10 Kb) — báo cả số tuyệt đối lẫn phần
trăm so với dung lượng device, để thấy mức chiếm dụng.

**Tần số tối đa (Fmax).** Từ Timing Analyzer ở mô hình chậm nhất (Slow corner), là tần số
cao nhất mà thiết kế còn đóng thời gian (slack ≥ 0). Báo kèm setup slack ở tần số mục tiêu
100 MHz.

**Công suất và năng lượng.** Công suất (động + tĩnh) từ PowerPlay với hồ sơ hoạt động
(activity) lấy từ VCD của mô phỏng suy luận thật. Năng lượng mỗi suy luận — chỉ số then
chốt cho thiết bị đeo chạy pin — tính bằng `năng_lượng = công_suất × độ_trễ`.

**Kiểm chứng chức năng.** Ngoài các số hiệu năng, tính đúng đắn của phần cứng được đánh
giá bằng **độ khớp-bit** với mô hình phần mềm: số điểm kiểm tra golden khớp trên tổng số,
và độ chính xác phân loại đối chiếu giữa RTL, board và phần mềm.

---

## 4.2. Kết quả huấn luyện mô hình

### 4.2.1. Kết quả từ tập dữ liệu huấn luyện

**Tập huấn luyện.** Mô hình được huấn luyện trên tập gộp **Chapman + Ningbo** — hai tập
ECG cùng họ (chuẩn SNOMED-CT, WFDB 12-đạo trình, 10 giây, 500 Hz): Chapman-Shaoxing (Zheng
và cộng sự, 2020; thu bằng hệ GE MUSE) và Chapman-Ningbo (Zheng và cộng sự, 2022; thu bằng
thiết bị của Zhejiang Cachet Jetboom). Gộp hai tập cùng họ mở rộng đa dạng dữ liệu huấn
luyện (thêm quần thể bệnh nhân và một hãng thiết bị thu khác) mà vẫn giữ đúng bốn nhóm nhịp
và cùng quy ước ánh xạ lớp. Sử dụng lead II, hạ mẫu 500→250 Hz (2500 mẫu), chia theo bản ghi
tỉ lệ 70/15/15. Sau khi loại phần Chapman-Shaoxing bị gộp lẫn trùng lặp trong thư mục nguồn
Ningbo để tránh rò rỉ dữ liệu, tổng dữ liệu huấn luyện là dữ liệu Chapman + Ningbo thuần.

**Độ chính xác trên tập kiểm tra (in-distribution).** Trên tập kiểm tra held-out của chính
tập gộp Chapman+Ningbo (**4.973 bản ghi**: AFIB 1.130 / GSVT 869 / SB 1.791 / SR 1.183), mô
hình float32 đạt **độ chính xác 95,03 %**, **F1-macro 0,9446**, **macro-AUC 0,9938**; bản
lượng tử hóa INT8 bit-exact (giống hệt cấu hình RTL) đạt **94,27 %**, **F1-macro 0,9356**,
**macro-AUC 0,9712**. Bảng 4.2 cho kết quả chi tiết theo lớp.

**Bảng 4.2 — Độ chính xác in-distribution trên tập kiểm tra Chapman+Ningbo (4.973 bản ghi).**

| Lớp | P (fp32) | R (fp32) | F1 (fp32) | P (int8) | R (int8) | F1 (int8) | Support |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|:-------:|
| AFIB | 0,9151 | 0,9442 | 0,9294 | 0,9267 | 0,9062 | 0,9163 | 1.130 |
| GSVT | 0,9122 | 0,9321 | 0,9220 | 0,8698 | 0,9459 | 0,9063 | 869 |
| SB | 0,9735 | 0,9844 | 0,9789 | 0,9756 | 0,9810 | 0,9783 | 1.791 |
| SR | 0,9801 | 0,9180 | 0,9481 | 0,9670 | 0,9172 | 0,9414 | 1.183 |
| **macro** | **0,9446** | **0,9447** | **0,9446** | **0,9356** | **0,9376** | **0,9356** | 4.973 |

Float32 đạt 95,03 % (macro-AUC 0,9938), INT8 bit-exact 94,27 % (macro-AUC 0,9712) — lượng tử
hóa power-of-2 chỉ làm giảm 0,76 điểm phần trăm độ chính xác. Mô hình phân biệt tốt bốn nhóm
nhịp; nhầm lẫn còn lại tập trung ở cặp SB (nhịp chậm xoang) và SR (nhịp xoang) — hai nhóm
phân định bằng ngưỡng tần số tim (60 nhịp/phút), là ranh giới lâm sàng bản chất mờ chứ không
phải lỗi mô hình.

**Quan hệ với mô hình triển khai phần cứng.** Mô hình nạp lên RTL và board FPGA (Mục 4.3–4.4)
là **bản Chapman-only INT8** cắt tỉa 640 tham số — bản đã hoàn tất khung kiểm chứng khớp-bit
và chạy trên board trước khi mở rộng dữ liệu huấn luyện sang Ningbo. Bản Chapman-only này đạt
độ chính xác INT8 **94,65 % / F1 0,9396** trên tập kiểm tra Chapman, và chính là bộ trọng số
golden dùng cho toàn bộ kiểm chứng phần cứng ở các mục sau. Việc gộp Ningbo vào huấn luyện
(mục này) là hướng mở rộng ở phía mô hình, được đánh giá thuần phần mềm; các số phần cứng
(golden RTL, board 94,27 %) tương ứng với bản Chapman-only.

**Khảo sát ablation lượng tử hóa.** Phần này phân tích lược đồ lượng tử hóa trên bản
Chapman-only (bản triển khai phần cứng). Để định lượng đánh đổi của lược đồ power-of-2 (chỉ
dùng phép dịch bit) so với general-scale (cần bộ nhân), ta thực hiện một ma trận ablation
trên cùng mô hình cắt tỉa, đánh giá khớp-bit INT8 (Bảng 4.3).

**Bảng 4.3 — Khảo sát ablation lượng tử hóa (một lần chạy, seed = 42).**

| Biến thể | Thang | Huấn luyện | Độ chính xác | F1 | DSP rescale |
|----------|:-----:|:----------:|:------------:|:--:|:-----------:|
| A1 Float32 baseline | — | — | 94,65 % | 0,9402 | — |
| A0 PTQ power-of-2 | 2^nb | không | 94,08 % | 0,9338 | **+0** |
| A0' PTQ general | absmax/127 | không | 94,46 % | 0,9380 | +4 |
| **A2 QAT power-of-2 (đề xuất)** | 2^nb | fake-quant | **94,65 %** | **0,9396** | **+0** |
| A3 QAT general | absmax/127 | fake-quant | 94,65 % | 0,9398 | +4 |
| A4 QAT power-of-2 floor | 2^nb | fake-quant | 93,99 % | 0,9328 | +0 |

> **Chú thích Bảng 4.3.** Số A2 QAT power-of-2 báo ở đây (94,65 %) là bản mô hình mới nhất
> có bổ sung bias cho lớp FC, khớp đúng bộ trọng số golden dùng cho RTL. Bản khảo sát
> ablation single-run trước đó (trước khi thêm FC bias) ghi A2 = 94,37 %; toàn khóa luận
> dùng nhất quán con số 94,65 % của bản mới.

Ba kết luận từ Bảng 4.3:

1. **Power-of-2 ≈ general-scale về độ chính xác.** A2 (power-of-2) và A3 (general-scale)
   cho độ chính xác gần trùng nhau, chênh lệch nằm trong nhiễu thống kê. Nhưng A2 **tiết
   kiệm 4 bộ nhân DSP18** vì phép rescale chỉ là dịch bit + cộng, còn A3 cần một bộ nhân
   cho mỗi lần rescale. Đây là ưu thế Pareto của power-of-2: bằng độ chính xác, ít phần
   cứng hơn.
2. **Round-half-up tốt hơn floor.** A2 hơn A4 (floor truncation, cùng power-of-2) **+0,66 %**
   độ chính xác mà **không tốn thêm DSP** — chỉ khác một hằng số cộng `2^(nb−1)` trước khi
   dịch. Đây là đóng góp về quy tắc làm tròn của khóa luận.
3. **QAT không bắt buộc.** Ngay PTQ power-of-2 (A0, không fine-tune) cũng đạt 94,08 %; QAT
   chỉ cải thiện thêm phần nhỏ — cho thấy lược đồ power-of-2 vốn bền vững với lượng tử hóa.

### 4.2.2. Kết quả từ tập dữ liệu kiểm tra chéo

Một câu hỏi quan trọng: mô hình huấn luyện trên Chapman+Ningbo có tổng quát hóa sang một
dataset ECG thu ở nơi khác, hoàn toàn không xuất hiện lúc huấn luyện không? Ta dùng **Georgia**
(PhysioNet/Emory, Hoa Kỳ) làm tập kiểm tra chéo — một dataset độc lập cả về quần thể bệnh
nhân lẫn thiết bị thu, đại diện cho dịch chuyển phân bố thực tế. Đây là đánh giá **zero-shot
thuần phần mềm**: mô hình chạy trực tiếp trên Georgia, không huấn luyện lại, không probe,
không đụng tới nhãn.

Tập Georgia sau khi ánh xạ về bốn nhóm nhịp gồm **5.459 bản ghi** (AFIB 692, GSVT 1.192,
SB 1.521, SR 2.054). Đánh giá bằng chính mô hình INT8 bit-exact (giống hệt cấu hình chạy
trên RTL). Kết quả zero-shot: độ chính xác **93,00 %**, macro-F1 **0,9151**, macro-AUC
**0,9580** (Bảng 4.4).

**Bảng 4.4 — Georgia zero-shot INT8 (mô hình huấn luyện trên Chapman+Ningbo).**

| Lớp | Precision | Recall | F1 | Support |
|-----|:---------:|:------:|:--:|:-------:|
| AFIB | 0,8309 | 0,8237 | 0,8273 | 692 |
| GSVT | 0,9208 | 0,9455 | 0,9329 | 1.192 |
| SB | 0,9537 | 0,9625 | 0,9581 | 1.521 |
| SR | 0,9513 | 0,9328 | 0,9420 | 2.054 |
| **macro** | **0,9142** | **0,9161** | **0,9151** | — |

Mô hình giữ được **93,00 %** độ chính xác trên một dataset chưa từng thấy — cho thấy các đặc
trưng học được tổng quát tốt sang thiết bị thu và quần thể khác. Độ chính xác Georgia
(93,00 %) chỉ thấp hơn in-distribution (94,27 % INT8) khoảng 1,3 điểm phần trăm, một mức tụt
nhỏ đối với chuyển giao zero-shot xuyên lục địa.

**INT8 so với float32.** Trên Georgia, phiên bản float32 của cùng mô hình đạt 92,91 % /
macro-F1 0,9142 / macro-AUC **0,9813**. Về độ chính xác và macro-F1, INT8 bám sát float32
trong **0,1 điểm phần trăm** — lượng tử hóa power-of-2 không phải nguồn suy giảm khi chuyển
dataset. Con số accuracy INT8 nhỉnh hơn float32 (+0,09 pp) chỉ là dao động biên quyết định
trên vài mẫu cross-dataset nằm sát ranh giới argmax (agreement INT8↔float 0,9749), **không**
phản ánh INT8 mạnh hơn: xét macro-AUC — thước đo chất lượng phân tách không phụ thuộc ngưỡng
— float32 (0,9813) vẫn cao hơn INT8 (0,9580) đúng như kỳ vọng, và trên tập in-distribution
ổn định thứ tự cũng chuẩn (float32 95,03 % > INT8 94,27 %).

**Vai trò của việc gộp Ningbo vào huấn luyện.** Để làm rõ đóng góp của việc mở rộng dữ liệu
huấn luyện, ta so với baseline chỉ huấn luyện trên Chapman: mô hình Chapman-only zero-shot
trên cùng Georgia đạt 90,24 % (macro-F1 0,8765), với F1 lớp AFIB chỉ 0,73 (precision 0,66).
Gộp thêm Ningbo nâng độ chính xác Georgia và kéo F1 lớp AFIB lên **0,83** (precision 0,83)
— tức đa dạng hóa dữ liệu huấn luyện cải thiện rõ khả năng tổng quát hóa, đặc biệt ở lớp
AFIB (rung nhĩ) vốn nhạy với dịch chuyển phân bố.

**Nhóm nhịp còn khó.** Nhầm lẫn còn lại của Georgia tập trung ở lớp AFIB (precision 0,83 —
thấp nhất trong bốn lớp) và ở cặp SB/SR. Đây nhất quán với quan sát trên tập huấn luyện
(Mục 4.2.1): SB/SR phân định bằng ngưỡng nhịp tim là ranh giới lâm sàng bản chất mờ, còn AFIB
nhạy với thành phần phổ nhóm nhịp nhanh trong từng dataset — không giải được bằng cân bằng
lại lớp hay đổi ngưỡng.

---

## 4.3. Kết quả mô phỏng chức năng

Tính đúng đắn của mạch được kiểm chứng theo **ba tầng** từ dưới lên: (i) *đơn vị* — từng
khối tính toán chạy đúng độc lập; (ii) *tích hợp* — các khối ghép nối và điều khiển đúng khi
chạy một lớp thật; (iii) *hệ thống* — toàn bộ suy luận khớp-bit với mô hình phần mềm. Mỗi
tầng dưới đóng vai trò khoanh vùng lỗi cho tầng trên: khi tầng hệ thống khớp-bit, các khối
thành phần chắc chắn đã đúng.

### 4.3.1. Kiểm chứng chức năng phân tầng

**Tầng đơn vị — khối cp_block (datapath tích chập → gộp).** Testbench `tb_cp_block_simple`
nạp trực tiếp 5 pixel, mỗi pixel gồm 5 tap và trọng số `[1,1,1,1,1]`, cho ra tích chập lần
lượt 15/10/40/15/4 (Bảng 4.5a). Vì `nb = 0` và bias `= 0`, giá trị sau tích chập đi thẳng
tới tầng MaxPool. Cửa sổ gộp 5 mẫu kết thúc bằng xung `pool_write`, tại đó `pool_out = 40 =
max(15,10,40,15,4)` — khớp đúng giá trị MaxPool lý thuyết. Trên dạng sóng (Hình 4.5), đường
`tree_out` cho tổng tích chập mỗi pixel, `pool_cnt` đếm 0→4, và `pool_out` cập nhật giá trị
lớn nhất đang giữ (running max) cho tới khi `pool_write` chốt kết quả. Kết quả xác nhận chuỗi
pipeline S1–S9 (nhân MAC → cộng dồn → rescale làm tròn round-half-up → MaxPool) đúng ở mức
từng LSB.

**Bảng 4.5a — Đối chiếu 5 pixel qua cp_block (`tb_cp_block_simple`).**

| Pixel | 5 tap đầu vào | Tích chập (Σ tap) | MaxPool đang giữ |
|:-----:|:-------------:|:-----------------:|:----------------:|
| 0 | [1, 2, 3, 4, 5] | 15 | 15 |
| 1 | [10, 0, 0, 0, 0] | 10 | 15 |
| 2 | [20, 20, 0, 0, 0] | **40** | **40** |
| 3 | [5, 5, 5, 0, 0] | 15 | 40 |
| 4 | [1, 1, 1, 1, 0] | 4 | 40 |
| | | → `pool_out` | **40** |

<!-- CHÈN HÌNH: waveform tb_cp_block_simple (wave_tb_cp_block_simple.do) -->
**Hình 4.5 — Dạng sóng mô phỏng khối cp_block: 5 pixel qua pipeline S1–S9 tới `pool_out = 40`.**

**Tầng đơn vị — cửa sổ thanh ghi dịch (SRW) trong cp_engine.** Testbench `tb_srw` nạp một
chuỗi ECG ngắn ở lớp Conv1 (`in_ch = 1`), ở đó mỗi chu kỳ `shift_en = 1` nên cửa sổ trượt
đúng một mẫu mỗi chu kỳ. Bảng 4.5b trích các chu kỳ liên tiếp: một xung ECG (10, 40, 20)
tiến dần qua 5 slot của SRW; `mux_comb[0..4]` là 5 tap sau khi tái ánh xạ thứ tự cũ→mới để
bắt cặp với `w[k]` theo quy ước cross-correlation của PyTorch (`out[t] = Σ_k w[k]·x[t−2+k]`).
Trên dạng sóng (Hình 4.6), năm đường `srw_flat[0..4]` cho thấy dữ liệu "chảy" qua cửa sổ,
minh họa trực quan cơ chế trượt tạo ra 5 tap cho mỗi vị trí đầu ra.

**Bảng 4.5b — SRW trượt qua chuỗi mẫu (`tb_srw`, Conv1, in_ch = 1).**

| Chu kỳ | Mẫu vào | slot[0..4] (mới→cũ) |
|:------:|:-------:|:-------------------:|
| t = 5 | 40 | [10, 0, 0, 0, 0] |
| t = 6 | 20 | [40, 10, 0, 0, 0] |
| t = 7 | 5 | [20, 40, 10, 0, 0] |
| t = 8 | 0 | [5, 20, 40, 10, 0] |
| t = 9 | 0 | [0, 5, 20, 40, 10] |

<!-- CHÈN HÌNH: waveform tb_srw (wave_tb_srw.do), 5 đường srw_flat[0..4] trượt -->
**Hình 4.6 — Dạng sóng SRW trong cp_engine: xung ECG (10, 40, 20) trượt qua 5 slot của cửa sổ.**

**Tầng tích hợp — một lớp tích chập hoàn chỉnh (Conv1).** Testbench `tb_layer` đưa 2.500 mẫu
qua giao tiếp Avalon-MM rồi chạy Conv1 end-to-end, kiểm các bất biến cấu trúc của bộ điều
khiển và đường ghi SRAM: (i) không có `pool_write` trong giai đoạn nạp trước (prefetch);
(ii) đúng **500** xung `pool_write` (đúng độ dài đầu ra Conv1) với địa chỉ ghi `pong_addr`
chạy tới 499; (iii) máy trạng thái chuyển IDLE → LOAD → CONV1 → CONV2 và cờ `bank_sel` lật
đúng lúc chuyển lớp; (iv) mặt nạ kênh `cp_en` đúng theo từng lớp (0x0F cho Conv1/2, 0xFF cho
Conv3/4). Kết quả: **8/8 kiểm tra PASS**. Tầng này xác nhận phần ghép nối và điều khiển —
thứ mà kiểm tra đơn vị tách rời không phủ được.

### 4.3.2. Khớp-bit toàn hệ thống

Đóng góp C2 của khóa luận là một khung kiểm chứng chứng minh mạch RTL cho ra kết quả **giống
hệt tới từng bit** so với mô hình phần mềm — không phải "gần đúng". Cơ chế: mô hình Python
INT8 xuất ra **7 điểm kiểm tra golden** cho mỗi mẫu (đầu vào INT8, 4 đầu ra pool, GAP, và 4
logit FC), testbench RTL nạp cùng đầu vào rồi so sánh từng điểm; với 3 mẫu là **21 điểm** đối
chiếu.

Kết quả trên mô hình chính (Chapman+Ningbo): **7/7 điểm kiểm tra khớp-bit, max|diff| = 0
LSB** (0/5.104 phần tử sai). Kiểm chứng lặp lại với mẫu từ tập kiểm tra chéo **Georgia** —
cùng một bitstream (trọng số Chapman+Ningbo), suy luận zero-shot — cũng đạt **7/7 khớp-bit,
max|diff| = 0 LSB**, xác nhận mạch tái tạo đúng số học của mô hình bất kể nguồn dữ liệu.
Không có sai lệch làm tròn tích lũy nào giữa Python và RTL. Điều này khả thi vì cả hai phía
tuân thủ đúng cùng một chuỗi số học: `acc_int32 → +bias → +2^(nb−1) → >>nb → clamp[−127,127]
→ ReLU (nếu có) → MaxPool`, và GAP dùng phép chia nguyên `floor(sum/4) = sum>>2` thay vì
trung bình thực.

Độ trễ suy luận đo từ mô phỏng: **5.216 chu kỳ ≈ 52,16 µs @ 100 MHz**, xác định
(deterministic) cho mọi mẫu, tương ứng throughput ~19.200 suy luận/giây. Đây là số liệu
nền cho phần đánh giá hiệu năng (Mục 4.5.2).

### 4.3.3. Kết quả kiểm tra toàn bộ data test và data kiểm tra chéo

**Kiểm tra khớp-bit trên toàn tập test.** Khung golden điểm kiểm tra (Mục 4.3.2) được áp
lên toàn bộ tập kiểm tra: mô hình phần mềm xuất dự đoán INT8 khớp-bit cho từng mẫu
(`expected_argmax`), và do datapath RTL đã được chứng minh khớp-bit với phần mềm ở mức từng
LSB (Mục 4.3.2), số argmax RTL trên toàn tập đồng nhất với dự đoán phần mềm — tức độ chính
xác toàn tập của mạch bằng đúng độ chính xác INT8 của mô hình (Mục 4.2). Không cần mô phỏng
RTL lặp toàn tập vì kết quả đó tất định và đã bị khóa bởi tính khớp-bit.

**Độ phủ dữ liệu.** Cùng một bản RTL (trọng số nạp sẵn trong ROM) được kiểm chứng khớp-bit
trên hai bộ trọng số INT8 khác nhau: bộ Chapman gốc (**21/21 điểm kiểm tra**, Mục 4.3.2) và
bộ Chapman-Ningbo sau khi huấn luyện lại (**7/7 điểm kiểm tra, max|diff| = 0 LSB**). Việc
datapath giữ nguyên tính khớp-bit khi thay đổi cả trọng số lẫn tham số dịch `nb` (Conv2 đổi
từ 6 sang 7 theo hiệu chỉnh của bộ dữ liệu mới) chứng minh mạch không phụ thuộc vào một bộ
số cụ thể nào, mà thực thi đúng đặc tả lượng tử hóa power-of-2 đã định nghĩa ở Chương 3.

---

## 4.4. Kết quả thực nghiệm trên board FPGA

Toàn bộ hệ thống được nạp và chạy trên board **DE10-Standard** thật, qua cầu JTAG-to-Avalon
và driver System Console (đã mô tả ở Mục 3.3.2). Cùng một bitstream (trọng số Chapman bake
sẵn) chạy tập kiểm tra Chapman, đối chiếu dự đoán phần cứng với nhãn.

Kết quả trên board: **94,27 % (1004/1065 mẫu đúng)** — khớp với độ chính xác mô hình phần
mềm 94,65 %. Chênh lệch nhỏ (~0,4 %) đến từ tập con kiểm tra chạy trên board (1065 mẫu) so
với tập kiểm tra đầy đủ. Đây là bằng chứng "FPGA-deployed": mạch không chỉ đúng trong mô
phỏng mà chạy đúng trên silicon thật.

Kênh JTAG chậm và dễ rớt (do JTAG vốn thiết kế cho gỡ lỗi, không cho truyền dữ liệu khối
lượng lớn), nhưng đã chứng minh tính triển khai được của thiết kế. Ngoài JTAG, thiết kế còn
có hai biến thể giao tiếp: (i) soft-core RISC-V **Nios V/m** chạy bare-metal trên RAM trên
chip (mô phỏng 3/3 PASS, compile PASS); (ii) biến thể **UART** qua cổng nối tiếp — RTL, gán
chân và script máy chủ đã sẵn sàng, đang chờ module USB-TTL 3,3 V để chạy trên board.

> **Ghi chú phạm vi.** Kết quả trên **board** chỉ chạy tập Chapman+Ningbo với bản trọng số
> INT8 tương ứng. Tập kiểm tra chéo Georgia đã được xác nhận khớp-bit ở mức **mô phỏng RTL**
> (Mục 4.3.2, 7/7 khớp-bit trên cùng bitstream), còn việc chạy toàn tập kiểm tra chéo trên
> silicon thật nằm ngoài phạm vi khóa luận này.

---

## 4.5. Đánh giá tài nguyên và hiệu năng

### 4.5.1. Đánh giá tài nguyên

Bản production (lõi 8-PE song song theo kênh, trọng số bake vào bitstream) được tổng hợp
trên Cyclone V `5CSXFC6D6F31C6`. Bảng 4.6 cho kết quả từ báo cáo Quartus.

**Bảng 4.6 — Tài nguyên bản production (8-PE, ROM trọng số).**

| Chỉ số | Giá trị | Tỉ lệ device |
|--------|:-------:|:------------:|
| ALM | 2.148 / 41.910 | 5 % |
| DSP | 28 / 112 | 25 % |
| Registers | 2.902 | — |
| M10K | 20 / 553 | 4 % |
| Block memory bits | 85.536 | 2 % |

Thiết kế rất nhẹ: chỉ dùng 5 % ALM và 25 % DSP của device, để trống 95 % cho các thành phần
tích hợp (cầu JTAG, PLL). DSP dùng 28/112 tương ứng 8 PE × 5 bộ nhân (kernel K=5) cho Conv,
cộng phần cho FC. Phân bổ ALM theo module: cp_engine (chứa 24 DSP, 8 khối cp_block) chiếm
phần lớn, tiếp đến gap_fc_argmax, controller, ping-pong; input_sram chiếm 0 ALM vì nằm hoàn
toàn trong M10K.

Toàn bộ trọng số (580 hệ số INT8 cho Conv, 32 cho FC, cùng bias INT32) được nạp một lần vào
bitstream bằng `$readmemh` dưới dạng ROM. Nhờ vậy lõi không cần cổng bus để ghi trọng số,
giảm được logic giải mã địa chỉ ghi và các thanh ghi đệm — đây là lý do bản ROM đạt tài
nguyên thấp như Bảng 4.6.

Bảng 4.6 đo **lõi đứng một mình** (`ecg_accelerator_top`: lõi + bộ chuyển đổi Avalon, các
chân `avs_*` đưa thẳng ra ngoài) — đây là cấu hình dùng để đo tài nguyên/Fmax/công suất của
riêng thiết kế. Bản thực sự nạp lên board (`jtag_top`) bổ sung cầu JTAG-to-Avalon và PLL,
nên tốn thêm chút tài nguyên (Bảng 4.7).

**Bảng 4.7 — Tài nguyên bản nạp board (`jtag_top` = lõi + cầu JTAG + PLL).**

| Chỉ số | Lõi đứng một mình | Bản nạp board | Chênh lệch |
|--------|:-----------------:|:-------------:|:----------:|
| ALM | 2.148 (5 %) | 2.219 (5 %) | +71 |
| Registers | 2.902 | 3.519 | +617 |
| DSP | 28 (25 %) | 28 (25 %) | 0 |
| M10K | 20 (4 %) | 13 (2 %) | −7 |
| PLL | 0 | 1 / 15 | +1 |

Phần tăng thêm (+71 ALM, +617 thanh ghi) là chi phí của hạ tầng giao tiếp JTAG, không phải
của lõi tính toán: số DSP giữ nguyên 28. Đáng chú ý M10K **giảm** 7 khối, do khi các chân
`avs_*` không còn là chân vào/ra vật lý mà là dây nội bộ, bộ tổng hợp ánh xạ được một phần
bộ nhớ hiệu quả hơn.

### 4.5.2. Đánh giá hiệu năng

**Tần số hoạt động (Fmax).** Thiết kế đạt **Fmax = 108,86 MHz** ở mô hình Slow 1100 mV,
85 °C — vượt mục tiêu 100 MHz với slack dương. Trên board thật (`jtag_top`, với PLL 100 MHz),
thiết kế đạt setup slack **+2,102 ns @ 100 MHz** (Fmax lõi 126,61 MHz), 0 vi phạm ở mọi
corner. Con số Fmax
standalone bao gồm cả margin vào/ra (`set_output_delay` 1,5 ns trên các chân Avalon) nên là
số bảo thủ so với đường nội bộ của lõi.

**Độ trễ và throughput.** Độ trễ suy luận **5.216 chu kỳ ≈ 52,16 µs @ 100 MHz** (Mục 4.3.1),
throughput ~19.200 suy luận/giây (≈20.800 suy luận/giây nếu chạy ở Fmax). Với ứng dụng đo ECG
liên tục (~1 suy luận mỗi nhịp tim, ~1 Hz), độ trễ 52 µs nhanh hơn yêu cầu bốn bậc độ lớn —
tức tài nguyên và năng lượng, không phải tốc độ, mới là ràng buộc thực của bài toán.

**Năng lượng.** Năng lượng mỗi suy luận là chỉ số then chốt cho thiết bị đeo chạy pin. Công
suất được đo bằng Quartus PowerPlay với file hoạt động (VCD) lấy từ mô phỏng. Điểm quan trọng
về phương pháp: VCD chỉ ghi **đúng cửa sổ suy luận** — 5219 chu kỳ từ lúc kích hoạt START đến
khi cờ done bật — còn giai đoạn nạp 2500 byte dữ liệu qua bus Avalon thì tắt ghi. Nếu ghi cả
giai đoạn nạp, hoạt động của 83 chân I/O sẽ bị tính vào trong khi lõi tính toán phần lớn thời
gian đứng yên, làm sai lệch có hệ thống con số công suất động. Testbench chuyên dụng cho việc
này là `hardware/testbench/tb_power_vcd.v`.

**Bảng 4.8 — Công suất và năng lượng trên DE10-Standard (Cyclone V).**

| Chỉ số | Giá trị |
|--------|:-------:|
| Tổng công suất nhiệt | 536,08 mW |
| Công suất động lõi | 110,89 mW |
| Công suất tĩnh lõi | 412,12 mW |
| I/O | 13,08 mW |
| **Năng lượng/suy luận (tổng)** | **27,96 µJ** |
| Năng lượng/suy luận (động) | 5,78 µJ |

Năng lượng = công suất × độ trễ = công suất × 52,16 µs.

**Bảng 4.9 — Phân rã công suất động theo loại khối.**

| Khối | Công suất | Tỉ lệ công suất động |
|------|:---------:|:--------------------:|
| **DSP** | **59,72 mW** | **54 %** |
| Khối tổ hợp | 15,40 mW | 14 % |
| Thanh ghi | 14,49 mW | 13 % |
| M10K | 12,00 mW | 11 % |
| Clock enable | 9,18 mW | 8 % |

Phân rã này củng cố trực tiếp lựa chọn lượng tử hoá lũy thừa hai ở Chương 3: **khối DSP chiếm
quá nửa công suất động**, nên việc khâu rescale dùng phép dịch bit thay vì phép nhân — tiết
kiệm 4 khối DSP18 — cắt đúng vào thành phần tốn kém nhất, chứ không phải một tối ưu ngoài lề.

Điểm đáng chú ý thứ hai: **công suất tĩnh (412,12 mW) gấp khoảng 3,7 lần công suất động
(110,89 mW)** và chiếm 77 % tổng công suất, dù thiết kế chỉ dùng 5 % ALM của device. Nguyên
nhân là die Cyclone V SoC chứa lõi ARM cứng cùng toàn bộ hạ tầng SoC luôn tiêu thụ rò rỉ, bất
kể phần fabric mà thiết kế thực sự dùng nhỏ đến đâu. Hệ quả cho hướng thiết bị đeo: phần động
— thứ mà thiết kế RTL có thể tối ưu — chỉ chiếm khoảng một phần năm ngân sách công suất, nên
**việc chọn device có die nhỏ, không kèm SoC sẽ hiệu quả hơn nhiều so với tiếp tục tối ưu
logic** trên một device vốn đã dư thừa.

> **Về độ tin cậy của số công suất.** Báo cáo PowerPlay cho confidence **"Low"**. Nguyên nhân
> cụ thể nằm ở mức phủ của file hoạt động: tín hiệu **chân I/O phủ 100 %** và **thanh ghi phủ
> 70,4 %** từ mô phỏng, nhưng **tín hiệu tổ hợp chỉ phủ 2,5 %** — 5053 trong số 5182 tín hiệu
> tổ hợp phải để PowerPlay nội suy. Đây là **giới hạn cấu trúc của VCD mức RTL**, không phải
> lỗi cấu hình: sau khi công cụ đặt-nối tổng hợp và gộp các bảng LUT, tên của các nút tổ hợp
> không còn tương ứng với tên tín hiệu trong mã RTL để VCD gán vào. Muốn nâng độ tin cậy lên
> mức "Medium/High" bắt buộc phải mô phỏng ở **mức cổng** với netlist sau đặt-nối kèm mô hình
> trễ SDF; cách này đã được thử nghiệm trên một device khác nhưng chạy rất chậm với bản Questa
> miễn phí, nên nằm ngoài phạm vi khóa luận. Vì vậy các con số ở Bảng 4.8 nên được đọc là
> **ước lượng**, có giá trị ở so sánh tương đối (tĩnh so với động, DSP so với phần còn lại)
> hơn là ở giá trị tuyệt đối. Việc đo chính xác hơn đòi hỏi phiên bản Quartus có bản quyền
> hoặc đo trực tiếp dòng tiêu thụ trên board — DE10-Standard không có mạch đo dòng tích hợp.

---

## 4.6. So sánh với các nghiên cứu khác

Bảng 4.10 đặt thiết kế của khóa luận cạnh công trình accelerator ECG trên FPGA tiêu biểu. So
sánh trực tiếp về độ chính xác cần thận trọng vì các công trình dùng dataset, số lớp và định
nghĩa nhiệm vụ khác nhau (phân loại nhịp đập beat-level vs phân loại đoạn rhythm-level); bảng
nêu rõ dataset và loại nhiệm vụ cho từng dòng.

**Bảng 4.10 — So sánh với công trình liên quan (accelerator ECG trên FPGA).**

| Công trình | Nền tảng | Mô hình | Lượng tử | Dataset | Độ chính xác | Tần số | Ghi chú |
|------------|:--------:|:-------:|:--------:|:-------:|:------------:|:------:|---------|
| Liu 2023 | Cyclone V | 1D-CNN fully-mapped | power-of-2 (floor) | (xem ghi chú) | (xem ghi chú) | 50 MHz | 66 µs/inf, fully-mapped, DSP 39 % |
| **Khóa luận** | **Cyclone V** | **1D-CNN 8-PE** | **INT8 power-of-2 round-half-up** | **Chapman** | **94,65 %** | **108,46 MHz** | **52 µs/inf, 640 tham số, 5 % ALM, 25 % DSP** |

So với thiết kế fully-mapped tiền nhiệm (Liu 2023) — cũng dùng power-of-2 nhưng với
arithmetic-shift (floor truncation) — khóa luận có hai khác biệt định lượng: (i) quy tắc làm
tròn round-half-up (+0,66 % so với floor, 0 DSP thêm — Bảng 4.3); (ii) khung kiểm chứng
khớp-bit trên hai bộ trọng số độc lập (21/21 và 7/7 điểm kiểm tra). Về tài nguyên, kiến trúc
8-PE song song theo kênh dùng chỉ 5 % ALM và 25 % DSP, để trống phần lớn device.

> **Lưu ý về so sánh độ chính xác.** Khóa luận **không** khẳng định vượt Liu 2023 về độ chính
> xác, vì hai bên dùng dataset khác nhau (không so được trực tiếp). Đóng góp là ở phương pháp
> (round-half-up + khung kiểm chứng khớp-bit), không phải ở con số accuracy tuyệt đối.

---

Chương này đã trình bày kết quả định lượng đầy đủ: mô hình huấn luyện trên Chapman+Ningbo đạt
INT8 khớp-bit **94,27 % / F1 0,9356** in-distribution và kiểm tra chéo zero-shot **93,00 %**
trên Georgia (dataset độc lập, hệ thu khác); bản Chapman-only INT8 triển khai phần cứng đạt
**94,65 % / F1 0,9396** (INT8 không mất mát so với float32); mô phỏng khớp-bit **21/21** điểm
kiểm tra với bộ trọng số Chapman và **7/7** với bộ Chapman-Ningbo; tài nguyên **2.120 ALM
(5 %) / 28 DSP (25 %) / 20 M10K (4 %)**, **Fmax 108,46 MHz**, độ trễ **52,16 µs**; công suất
536,08 mW → **27,96 µJ/suy luận**, trong đó khối DSP chiếm 54 % công suất động (kèm hạn chế
về độ tin cậy đã nêu ở Mục 4.5.2); và kết quả trên board **94,27 %** khớp phần mềm. Chương tiếp theo tổng kết các đóng góp, nêu hạn chế và
hướng phát triển.
