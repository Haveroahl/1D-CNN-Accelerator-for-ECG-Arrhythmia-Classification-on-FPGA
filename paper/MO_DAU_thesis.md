# CHƯƠNG 1. MỞ ĐẦU

## 1.1. Đặt vấn đề

Rối loạn nhịp tim (cardiac arrhythmia) là một trong những nguyên nhân hàng đầu gây
đột tử do tim trên thế giới. Nhiều dạng rối loạn nhịp — đặc biệt là rung nhĩ (atrial
fibrillation) — diễn ra **không liên tục và không triệu chứng**, nên thường bị bỏ sót
trong các lần đo điện tim (ECG) ngắn tại phòng khám. Bằng chứng lâm sàng cho thấy việc
theo dõi ECG **liên tục, kéo dài** bằng thiết bị đeo (wearable) làm tăng đáng kể tỉ lệ
phát hiện sớm, từ đó mở ra cơ hội can thiệp kịp thời và giảm nguy cơ tử vong.

Điều này đặt ra một bài toán kỹ thuật cụ thể: **phân loại rối loạn nhịp tim từ tín hiệu
ECG ngay trên thiết bị đeo, theo thời gian thực, trong giới hạn năng lượng rất chặt của
một thiết bị chạy pin.** Đây chính là bài toán mà luận văn này hướng tới giải quyết.

Các mạng nơ-ron tích chập một chiều (1D-CNN) hiện đạt độ chính xác cao trong phân loại
ECG và đã được cộng đồng nghiên cứu chấp nhận rộng rãi. Tuy nhiên, độ chính xác cao mới
chỉ là một nửa của bài toán. Nửa còn lại — và là phần thường bị bỏ ngỏ — là **làm thế nào
để triển khai mô hình đó một cách hiệu quả trên một nền tảng phần cứng phù hợp với ràng
buộc của thiết bị đeo**. Một mô hình chính xác nhưng tiêu tốn hàng watt hoặc cần GPU để
chạy thì không thể đưa vào thiết bị đeo. Khoảng cách giữa "mô hình chạy được trên máy
tính" và "mô hình chạy được trên thiết bị thật" chính là không gian mà luận văn này tập
trung khai thác, thông qua hướng tiếp cận **đồng thiết kế phần cứng – phần mềm
(hardware–software co-design)**.

## 1.2. Định hướng tiếp cận: Đồng thiết kế phần cứng – phần mềm

Mục tiêu trọng tâm của luận văn là **thiết kế và hiện thực một lõi (core) tăng tốc CNN
trên FPGA cho bài toán phân loại 4 nhóm nhịp tim, dựa trên một mô hình CNN nhẹ**. Luận
văn **không** đặt mục tiêu đề xuất một kiến trúc mạng nơ-ron mới hay một thuật toán phân
loại ECG vượt trội về độ chính xác; mô hình CNN nhẹ đóng vai trò **tiền đề** để đưa lên
một nền tảng phần cứng triển khai được thực tế, với năng lượng đủ thấp cho thiết bị đeo.

Vì vậy, công việc được phân bổ theo hướng **phần cứng là trọng tâm (~70%), phần mềm là
hỗ trợ (~30%)**. Quan hệ giữa hai phần không phải là hai khối tách rời ghép lại, mà là
một vòng đồng thiết kế: **các quyết định ở phần mềm được đưa ra nhằm phục vụ trực tiếp
cho hiệu quả của phần cứng**, và ngược lại ràng buộc phần cứng định hình cách lượng tử
hóa và tỉa mô hình. Cụ thể:

- **Phần mềm (30%)** — huấn luyện, tỉa kênh (channel pruning) và lượng tử hóa mô hình
  CNN về INT8 bằng PyTorch. Điểm mấu chốt là lựa chọn **lượng tử hóa theo lũy thừa của 2
  (power-of-two QAT)**: hệ số rescale là một phép dịch bit thay vì một phép nhân số thực.
  Đây không phải lựa chọn ngẫu nhiên — nó được đưa ra **vì mục tiêu phần cứng**: phép dịch
  bit loại bỏ hoàn toàn nhu cầu dùng khối nhân (DSP) cho bước rescale, qua đó giảm tài
  nguyên và công suất động trên FPGA.

- **Phần cứng (70%)** — thiết kế lõi IP accelerator bằng Verilog, target Intel Cyclone V,
  gồm khối tích chập song song, bộ nhớ on-chip, máy trạng thái điều khiển và giao tiếp
  bus. Toàn bộ đường tính toán được thiết kế để **khớp bit-exact** với mô hình phần mềm.

Sợi chỉ xuyên suốt nối hai phần là chuỗi nhân-quả phục vụ mục tiêu ứng dụng:

> **power-of-two QAT (phần mềm) → 0 DSP cho rescale (phần cứng) → ít công suất động →
> năng lượng/lần suy luận thấp → phù hợp thiết bị đeo theo dõi liên tục.**

Chính chuỗi này là minh chứng cho việc đây là một bài toán **co-design thực sự**, chứ
không phải huấn luyện một mô hình rồi đem "thả" lên phần cứng.

Mô hình CNN nhẹ được sử dụng là một kiến trúc đã được kiểm chứng đạt độ chính xác cao
trong y văn cho phân loại nhịp tim; luận văn kế thừa nó làm tiền đề và tập trung vào việc
hiện thực hiệu quả trên phần cứng. Câu hỏi về cơ sở y sinh của kiến trúc mạng vì vậy thuộc
tầng thuật toán đã được giải quyết, nằm ngoài phạm vi đóng góp của luận văn này.

## 1.3. Mục tiêu và phạm vi

**Mục tiêu tổng quát:** thiết kế, hiện thực và kiểm chứng một lõi tăng tốc CNN trên FPGA
cho bài toán phân loại rối loạn nhịp tim từ ECG, đạt độ chính xác tương đương mô hình phần
mềm và có năng lượng đủ thấp cho hướng ứng dụng thiết bị đeo.

**Mục tiêu cụ thể:**

1. Huấn luyện, tỉa và lượng tử hóa INT8 (power-of-two QAT) mô hình 1D-CNN phân loại 4
   nhóm nhịp (AFIB / GSVT / SB / SR) trên tập dữ liệu Chapman, giữ độ chính xác cao.
2. Thiết kế lõi IP accelerator bằng Verilog cho Intel Cyclone V, với đường tính toán
   **khớp bit-exact** với mô hình phần mềm.
3. Xây dựng quy trình **kiểm chứng bit-exact** giữa mô hình phần mềm và mô phỏng RTL làm
   bằng chứng cho tính đúng đắn của thiết kế.
4. Tổng hợp (synthesis) thực tế trên Quartus và đo các chỉ số phần cứng thật: tài nguyên,
   tần số tối đa, công suất và **năng lượng trên mỗi lần suy luận**.
5. Triển khai và kiểm chứng trên **board FPGA thật (DE10-Standard)**, đối chiếu độ chính
   xác trên phần cứng với phần mềm.

**Phạm vi:** luận văn giới hạn ở ECG đơn đạo trình (single-lead), mô hình kích thước nhỏ
phù hợp thiết bị đeo, và phân loại 4 nhóm nhịp. Các vấn đề như đa đạo trình, xử lý luồng
liên tục thời gian thực và mở rộng số lớp được xem là hướng phát triển.

## 1.4. Đóng góp của luận văn

1. **Lõi IP accelerator CNN trên FPGA** cho phân loại ECG, hoàn chỉnh từ thiết kế RTL tới
   triển khai trên board thật, với độ trễ xác định và tài nguyên nhỏ.
2. **Quy trình lượng tử hóa power-of-two với rescale làm tròn (round-half-up)** được đối
   chiếu định lượng với lượng tử hóa tỉ lệ tổng quát, cho thấy trade-off tài nguyên/độ
   chính xác — phục vụ trực tiếp mục tiêu tiết kiệm DSP và năng lượng.
3. **Khung kiểm chứng bit-exact phần mềm ↔ phần cứng** làm bằng chứng tin cậy cho thiết
   kế, thay cho cách báo cáo "mô phỏng INT8" thường thấy vốn dễ lệch với RTL.
4. **Số liệu phần cứng đo thật** (tài nguyên, tần số, công suất, năng lượng/lần suy luận)
   và **kết quả chạy trên board FPGA thật**, đưa kết quả vượt khỏi mức mô phỏng.

## 1.5. Bố cục luận văn

- **Chương 1 — Mở đầu:** đặt vấn đề, định hướng co-design, mục tiêu và đóng góp.
- **Chương 2 — Cơ sở lý thuyết và công trình liên quan:** ECG và rối loạn nhịp, 1D-CNN,
  lượng tử hóa INT8, tăng tốc CNN trên FPGA.
- **Chương 3 — Mô hình và phương pháp lượng tử hóa:** kiến trúc mạng, power-of-two QAT,
  rescale round-half-up, khung kiểm chứng bit-exact.
- **Chương 4 — Thiết kế phần cứng:** kiến trúc accelerator, datapath, điều khiển, bộ nhớ,
  giao tiếp bus.
- **Chương 5 — Hiện thực và kiểm chứng:** tổng hợp Quartus, kết quả bit-exact, triển khai
  trên board.
- **Chương 6 — Kết quả và bàn luận:** độ chính xác, tài nguyên, tần số, năng lượng, đánh
  giá xuyên tập dữ liệu, so sánh với công trình liên quan.
- **Chương 7 — Kết luận và hướng phát triển.**
