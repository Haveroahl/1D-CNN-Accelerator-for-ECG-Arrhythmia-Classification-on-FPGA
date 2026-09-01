# CHƯƠNG 1: GIỚI THIỆU

## 1.1. Bối cảnh và lý do chọn đề tài

### Bối cảnh thực hiện

Rối loạn nhịp tim (arrhythmia) là một trong những nguyên nhân hàng đầu gây đột tử do
tim và các biến chứng tim mạch nghiêm trọng trên phạm vi toàn cầu. Một đặc điểm gây
khó khăn đáng kể cho công tác chẩn đoán là nhiều dạng rối loạn nhịp — điển hình như
rung nhĩ (atrial fibrillation) — có tính chất kịch phát (paroxysmal), xuất hiện không
liên tục và có thể không được ghi nhận trong một lần đo điện tâm đồ (ECG) đơn lẻ tại
cơ sở y tế. Trong khi đó, phương pháp phân tích ECG truyền thống dựa trên đánh giá
trực quan của bác sĩ chuyên khoa vừa đòi hỏi nguồn lực chuyên môn, vừa không đáp ứng
được yêu cầu giám sát liên tục trong thời gian dài. Thực tế này đặt ra nhu cầu cấp
thiết đối với các hệ thống giám sát tim mạch tự động, có khả năng vận hành liên tục
trên các thiết bị đeo (wearable) hoặc thiết bị biên (edge computing), nhằm phát hiện
sớm các bất thường nhịp tim ngay tại nơi thu nhận tín hiệu mà không phụ thuộc vào hạ
tầng tính toán tập trung.

Trong bối cảnh đó, các phương pháp học sâu — đặc biệt là mạng nơ-ron tích chập (CNN)
— đã được chứng minh có khả năng phân loại rối loạn nhịp tim từ tín hiệu ECG với độ
chính xác tiệm cận, thậm chí ở một số nghiên cứu vượt qua mức đánh giá của bác sĩ
chuyên khoa tim mạch [4]. Tuy nhiên, phần lớn các mô hình đạt hiệu năng cao nói trên
có độ phức tạp tính toán và số lượng tham số lớn, dẫn đến hai giới hạn khi triển khai
trên thiết bị biên: nền tảng vi điều khiển (MCU) thường không đáp ứng đủ thông lượng
xử lý thời gian thực, trong khi nền tảng GPU lại vượt quá ràng buộc về công suất tiêu
thụ và kích thước vật lý cho phép của một thiết bị đeo. FPGA, với đặc tính tiêu thụ
điện năng thấp, khả năng xử lý song song có thể cấu hình phù hợp với quy mô của một
mạng CNN nhỏ, độ trễ xử lý xác định (deterministic latency), và khả năng tùy biến
kiến trúc phần cứng theo đúng cấu trúc của mô hình đích, do đó được xem là một nền
tảng triển khai phù hợp cho lớp bài toán này. Đây là bối cảnh mà đề tài khóa luận
được thực hiện.

### Lý do chọn đề tài

Đề tài hướng tới việc xây dựng một mô hình **CNN một chiều (1D-CNN) nhẹ** thực hiện
phân loại **4 nhãn nhịp tim** — AFIB (rung nhĩ), GSVT (nhịp nhanh trên thất), SB
(nhịp chậm xoang) và SR (nhịp xoang bình thường) — dựa trên tập dữ liệu lâm sàng
**Chapman–Shaoxing** (thu tại Shaoxing People's Hospital, 10.646 bệnh nhân, ECG
12 chuyển đạo 10 giây, 500 Hz) [1]. Mô hình sau đó được **kiểm tra chéo tập dữ liệu
(cross-dataset)** trên hai nguồn độc lập nhằm đánh giá khả năng tổng quát hóa
(generalization): tập **Ningbo** cùng họ Chapman nhưng thu bằng thiết bị của hãng khác
(near-transfer), và tập **Georgia (Emory G12EC, PhysioNet/CinC 2020)** [2] khác cả
quần thể bệnh nhân lẫn cách phối hợp bệnh (far-transfer). Cả hai được áp dụng zero-shot
— dùng trực tiếp mô hình đã huấn luyện trên Chapman mà không huấn luyện lại.

Sau khi có mô hình đạt độ chính xác cao (94,65% trên Chapman, bit-exact giữa mô phỏng
INT8 và phần cứng), đề tài **thiết kế một lõi tăng tốc CNN 1D trên FPGA** cho mô hình
này theo hướng **tối ưu tài nguyên phần cứng**: lượng tử hóa INT8 theo lũy thừa của 2
(power-of-2) để phép co giãn (rescale) chỉ cần dịch bit và cộng — **không tốn DSP** —
thay cho phép nhân số thực của lượng tử hóa tổng quát. Lõi CNN được **tích hợp với
module JTAG-to-Avalon** thông qua giao tiếp **Avalon-MM** để nạp trọng số/dữ liệu và
đọc kết quả, cho phép chạy trực tiếp trên bo mạch FPGA. Toàn bộ thiết kế được **kiểm
chứng hai cấp**: (i) trên **mô phỏng** RTL với 21 điểm kiểm tra bit-exact so với mô
hình Python vàng (golden), và (ii) trên **phần cứng thực tế** bo mạch DE10-Standard.

Lý do chọn đề tài này gồm ba khía cạnh:

1. **Tính ứng dụng lâm sàng.** Phát hiện sớm rối loạn nhịp bằng thiết bị đeo có ý
   nghĩa cứu sống trực tiếp; một lõi phần cứng nhỏ, ít điện năng là điều kiện tiên
   quyết để đưa mô hình vào thiết bị thực.

2. **Khoảng trống kỹ thuật.** Phần lớn công trình ECG-FPGA hiện có: (a) dùng dịch bit
   theo lũy thừa 2 nhưng làm tròn cắt (floor truncation) [18]; (b) báo độ chính xác
   ở mức mô phỏng INT8 nhưng không chứng minh khớp bit-exact với RTL; và (c) chỉ đánh
   giá trên một tập dữ liệu, không trả lời được câu hỏi mô hình có tổng quát hóa hay
   không. Khóa luận này lần lượt giải quyết cả ba điểm.

3. **Tính khả thi và kiểm chứng được.** Mô hình nhỏ (640 tham số) đủ để triển khai
   trọn vẹn và verify chặt chẽ trong phạm vi một khóa luận, đồng thời vẫn giữ độ chính
   xác cạnh tranh với các mô hình lớn hơn nhiều.

---

## 1.2. Động lực và mục tiêu của đề tài

### Động lực

Từ bối cảnh nêu trên, động lực trực tiếp của đề tài là khoảng cách giữa **mô hình học
sâu đạt độ chính xác cao** và **ràng buộc triển khai trên thiết bị biên đeo được**. Một
mô hình dù chính xác đến đâu nhưng cần hàng triệu tham số, hàng chục milijoule mỗi lần
suy luận và một GPU để chạy thì không thể tích hợp vào một thiết bị giám sát nhịp tim
liên tục. Ngược lại, nếu thu gọn mô hình quá mức để chạy trên MCU thì lại đánh đổi độ
chính xác lâm sàng. FPGA cho phép thoát khỏi thế lưỡng nan này: thiết kế một lõi phần
cứng chuyên dụng ánh xạ đúng cấu trúc của một CNN 1D nhỏ, đạt độ trễ xác định ở mức vài
chục micro-giây với công suất thấp.

Động lực thứ hai đến từ khoảng trống phương pháp trong các công trình ECG-FPGA hiện có
(mục 1.1): làm tròn cắt gây mất độ chính xác không cần thiết; số liệu INT8 báo cáo ở
mức mô phỏng nhưng không được chứng minh khớp bit-exact với phần cứng thực thi; và mô
hình chỉ được đánh giá trên một tập dữ liệu duy nhất nên không trả lời được câu hỏi về
khả năng tổng quát hóa. Đề tài lấy chính ba khoảng trống này làm định hướng đóng góp.

### Mục tiêu

Mục tiêu tổng quát của đề tài là **thiết kế, kiểm chứng và triển khai một lõi tăng tốc
CNN 1D nhẹ trên FPGA Intel Cyclone V để phân loại bốn nhóm rối loạn nhịp tim từ tín
hiệu ECG**, theo hướng tối ưu tài nguyên phần cứng và bảo đảm khớp bit-exact giữa mô
hình phần mềm và phần cứng.

Mục tiêu tổng quát được cụ thể hóa thành các mục tiêu thành phần:

1. Xây dựng một mô hình 1D-CNN nhẹ (640 tham số) phân loại 4 nhóm nhịp đạt độ chính
   xác cạnh tranh trên dữ liệu Chapman/Ningbo.
2. Áp dụng lượng tử hóa INT8 power-of-2 nhận biết lượng tử (QAT) sao cho mô hình INT8
   giữ được độ chính xác của mô hình số thực và phù hợp thực thi phần cứng không cần
   DSP cho khâu rescale.
3. Thiết kế lõi tăng tốc CNN 1D bằng Verilog, đạt độ trễ xác định và đóng được timing
   trên Cyclone V.
4. Thiết lập khung kiểm chứng bit-exact giữa mô hình Python và RTL, đồng thời đánh giá
   khả năng tổng quát hóa của mô hình trên một tập dữ liệu độc lập.
5. Triển khai và kiểm thử lõi trên bo mạch FPGA thực tế (DE10-Standard).

---

## 1.3. Phương pháp thực hiện đề tài

Đề tài được thực hiện theo một quy trình khép kín từ mô hình phần mềm tới phần cứng
trên board, gồm các bước chính:

1. **Thiết kế và huấn luyện mô hình.** Xây dựng mô hình 1D-CNN nhẹ theo hướng
   layer-by-layer bằng Python/PyTorch; huấn luyện học có giám sát trên nhãn nhịp tim của
   tập Chapman/Ningbo với thuật toán tối ưu Adam [20], hàm mất mát Cross-Entropy và cơ
   chế lan truyền ngược (back-propagation).

2. **Nén mô hình.** Tỉa kênh (structured channel pruning) đưa số kênh về power-of-2
   (4,4,8,8) để phù hợp phần cứng, sau đó tinh chỉnh lại (fine-tune) để phục hồi độ
   chính xác.

3. **Lượng tử hóa QAT power-of-2.** Huấn luyện nhận biết lượng tử với fake-quantize và
   straight-through estimator, chọn hệ số dịch theo lũy thừa của 2, làm tròn round-half-up;
   xuất trọng số/độ lệch dưới dạng số nguyên INT8/INT32.

4. **Kiểm chứng khả năng tổng quát hóa.** Đối chiếu zero-shot mô hình đã lượng tử hóa
   trên hai tập độc lập — Ningbo (near-transfer) và Georgia (far-transfer) — để đo mức
   tổng quát hóa sang thiết bị thu và quần thể bệnh nhân khác.

5. **Sinh dữ liệu vàng (golden reference).** Dùng chính mô hình Python INT8 để xuất 21
   điểm kiểm tra trung gian (input, 4 đầu ra pool, GAP, logits FC) làm chuẩn so sánh
   bit-exact với phần cứng.

6. **Thiết kế lõi phần cứng.** Hiện thực IP core CNN 1D bằng Verilog (pipeline CP-block
   5 tầng, CP-Engine 8 PE, khối điều khiển FSM, khối GAP/FC/Argmax) và xây lớp giao tiếp
   Avalon-MM wrapper.

7. **Tích hợp và mô phỏng.** Tích hợp lõi CNN với IP JTAG-to-Avalon của Intel qua
   Platform Designer; mô phỏng RTL bằng ModelSim/Questa, đối chiếu bit-exact với golden.

8. **Tổng hợp và đánh giá.** Tổng hợp bằng Quartus Prime (tài nguyên, Fmax), ước lượng
   công suất/năng lượng bằng PowerPlay với VCD từ mô phỏng.

9. **Triển khai trên board.** Nạp bitstream lên DE10-Standard, dùng JTAG-to-Avalon +
   System Console nạp trọng số/ECG và đọc kết quả, kiểm thử độ chính xác thực tế.

---

## 1.4. Đóng góp của đề tài

Các đóng góp chính của khóa luận gồm:

1. **Phương pháp lượng tử hóa power-of-2 với làm tròn round-half-up.** Thay phép làm
   tròn cắt (floor truncation) thường gặp trong công trình ECG-FPGA đối chứng [18] bằng
   round-half-up `(acc + 2^(nb-1)) >> nb`, đo được cải thiện độ chính xác ở chi phí
   phần cứng bằng không (chỉ thêm một hằng số cộng vào chuỗi dịch bit sẵn có). Đóng góp
   được củng cố bằng khảo sát ablation định lượng power-of-2 so với lượng tử hóa
   general-scale, cho thấy power-of-2 tiết kiệm DSP mà độ chính xác tương đương.

2. **Khung kiểm chứng bit-exact phần mềm ↔ phần cứng.** Xây dựng quy trình đối chiếu
   21 điểm kiểm tra trung gian giữa mô hình Python QAT và mô phỏng RTL, khớp bit-exact
   trên toàn bộ chuỗi tính toán. Số liệu INT8 báo cáo trong khóa luận do đó là con số
   phần cứng thực sự tạo ra, không phải xấp xỉ mô phỏng.

3. **Nghiên cứu chuyển giao đa-dataset (cross-dataset).** Đánh giá định lượng khả năng
   tổng quát hóa của mô hình INT8 zero-shot trên hai tập độc lập ở hai đầu phổ dịch
   chuyển phân bố: Ningbo (near-transfer, giữ độ chính xác ~92,6%) và Georgia
   (far-transfer). Phân tách phần suy giảm do lượng tử hóa và phần do dịch chuyển phân
   bố (distribution shift), cho thấy lượng tử hóa power-of-2 hầu như không gây thêm mất
   mát so với mô hình số thực.

4. **Lõi IP tăng tốc CNN 1D hoàn chỉnh trên Cyclone V.** Kiến trúc 8-PE channel-parallel
   với pipeline CP-block 5 tầng, trọng số nạp sẵn trong ROM, độ trễ xác định 52,16
   µs/suy luận (~19.200 suy luận/giây) ở 100 MHz; đã tổng hợp thật trên Quartus và đo
   tài nguyên (2.120 ALM, 5 % device), Fmax (108,46 MHz) cùng công suất/năng lượng.

5. **Triển khai và kiểm chứng trên phần cứng thực tế.** Toàn bộ luồng chạy trực tiếp trên
   bo mạch DE10-Standard qua cầu JTAG-to-Avalon, đạt độ chính xác trên board **94,27 %**
   — khớp với mô hình phần mềm, xác nhận thiết kế không chỉ đúng trong mô phỏng mà còn
   hoạt động đúng trên silicon thật.

---

## 1.5. Cấu trúc khóa luận

Khóa luận được tổ chức thành năm chương:

- **Chương 1 — Giới thiệu.** Trình bày bối cảnh, lý do chọn đề tài, động lực và mục
  tiêu, phương pháp thực hiện, đóng góp và cấu trúc khóa luận.

- **Chương 2 — Cơ sở lý thuyết.** Trình bày kiến thức nền về tín hiệu ECG và bốn nhóm
  nhịp; mạng CNN một chiều (tích chập 1D, MaxPool, GAP, FC); nén mô hình (tỉa kênh,
  lượng tử hóa INT8); và nền tảng thiết kế số trên FPGA (Cyclone V, Avalon-MM,
  pipeline).

- **Chương 3 — Mô hình CNN, lượng tử hóa và chuyển giao đa-dataset.** Trình bày kiến
  trúc mô hình đề xuất, quy trình tỉa kênh và QAT power-of-2, khảo sát ablation lượng
  tử hóa, và nghiên cứu chuyển giao cross-check.

- **Chương 4 — Thiết kế phần cứng accelerator.** Trình bày chi tiết kiến trúc lõi
  tăng tốc CNN: tổ chức bộ nhớ, pipeline CP-block 5 tầng, CP-Engine 8 PE, khối điều
  khiển FSM, khối GAP/FC/Argmax và cơ chế nạp trọng số qua Avalon.

- **Chương 5 — Kiểm định, tổng hợp, triển khai và đánh giá.** Trình bày khung kiểm
  chứng bit-exact, kết quả tổng hợp Quartus (tài nguyên, Fmax), đo năng lượng, triển
  khai trên DE10-Standard và so sánh với công trình liên quan.
