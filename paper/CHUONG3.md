# CHƯƠNG 3: THIẾT KẾ VÀ TRIỂN KHAI MẠNG CNN

Chương này trình bày toàn bộ quá trình hiện thực hóa hệ thống, đi từ mô hình phần mềm
tới lõi phần cứng và cuối cùng là khối tích hợp giao tiếp. Nội dung chia làm ba phần
tương ứng ba giai đoạn của quy trình thiết kế: (i) quy trình xử lý dữ liệu và huấn luyện
mô hình bằng PyTorch, kết thúc bằng việc lượng tử hóa và trích xuất bộ trọng số INT8
(Mục 3.1); (ii) thiết kế kiến trúc phần cứng lõi tăng tốc CNN bằng Verilog, gồm khối
tích chập–gộp, khối engine song song, khối phân loại đầu ra và khối điều khiển (Mục 3.2);
(iii) tích hợp lõi CNN với hệ thống bus Avalon-MM và cầu nối JTAG-to-Avalon để giao tiếp
với máy tính chủ (Mục 3.3). Toàn bộ số liệu định lượng — độ chính xác, khảo sát lượng tử
hóa, tài nguyên, năng lượng — được đặt ở Chương 4 nhằm tách bạch phần *cách làm* (Chương
3) khỏi phần *kết quả đo được* (Chương 4).

Quan điểm thiết kế xuyên suốt chương là **đồng thiết kế phần cứng–phần mềm
(hardware–software co-design)**: mọi quyết định ở phía phần mềm (số kênh lũy thừa hai,
thang lượng tử lũy thừa mũ hai, quy tắc làm tròn, bố cục trọng số) đều được đưa ra với
ràng buộc hiện thực phần cứng trong đầu, sao cho mô hình phần mềm và mạch phần cứng khớp
nhau tới từng bit. Nhờ vậy, kết quả suy luận trên FPGA không phải là một phép xấp xỉ của
mô hình phần mềm mà là bản sao chính xác về mặt số học của nó — nền tảng cho khung kiểm
chứng khớp-bit ở Chương 4.

---

## 3.1. Quy trình xử lý dữ liệu và huấn luyện mô hình

Giai đoạn phần mềm có mục tiêu tạo ra một bộ trọng số INT8 nhỏ gọn, đủ chính xác và có
định dạng phù hợp để nạp trực tiếp vào phần cứng. Quy trình gồm bốn bước tuần tự, thể
hiện ở Hình 3.1:

```
   Dữ liệu thô Chapman (12-lead, 500 Hz, XML)
                  │
                  ▼
   ┌───────────────────────────────────┐
   │ 3.1.1  Xử lý dữ liệu               │
   │  Lead II · 500→250 Hz · Z-score   │
   │  ánh xạ 4-class · split 70/15/15  │
   └────────────────┬──────────────────┘
                    ▼
   ┌───────────────────────────────────┐
   │ 3.1.2  Huấn luyện + cắt tỉa kênh   │
   │  float32 (1244) → prune → 640      │
   │  fine-tune 2 pha                   │
   └────────────────┬──────────────────┘
                    ▼
   ┌───────────────────────────────────┐
   │ 3.1.3  Lượng tử hóa INT8           │
   │  power-of-2 QAT · round-half-up    │
   │  nb, w_shift, input_shift          │
   └────────────────┬──────────────────┘
                    ▼
   ┌───────────────────────────────────┐
   │ 3.1.4  Trích xuất trọng số         │
   │  flat_weights.hex (580 INT8)       │
   │  pack 40-bit · bias INT32 LE       │
   └────────────────┬──────────────────┘
                    ▼
         Nạp vào phần cứng (Mục 3.2, 3.3)
```

**Hình 3.1 — Quy trình phần mềm bốn bước, từ dữ liệu thô đến bộ trọng số sẵn sàng nạp phần cứng.**

### 3.1.1. Xử lý dữ liệu

Tập dữ liệu huấn luyện và đánh giá chính là **Chapman-Shaoxing** [1], gồm bản ghi ECG 12
chuyển đạo, thời lượng 10 giây, tần số lấy mẫu 500 Hz, thu nhận trên hệ thống GE MUSE ECG
của bệnh viện Shaoxing People's Hospital. Bộ dữ liệu chứa 10.646 bản ghi từ 10.646 bệnh
nhân, mỗi bản ghi được gán nhãn nhịp theo mã SNOMED-CT ở cấp độ bản ghi.

**Ánh xạ nhãn.** Các mã nhịp chi tiết của bộ dữ liệu được gộp về bốn nhóm nhịp đích đã
định nghĩa ở Mục 2.1.2. Quy tắc gộp dựa trên bản chất điện sinh lý của từng nhóm, cho ở
Bảng 3.1.

**Bảng 3.1 — Quy tắc ánh xạ mã nhịp SNOMED-CT về bốn nhóm nhịp đích.**

| Nhóm đích | Mã nhịp gốc được gộp | Bản chất |
|-----------|----------------------|----------|
| AFIB | AFib, AFlutter | Rung nhĩ / cuồng nhĩ — mất sóng P, RR không đều |
| GSVT | ST, SVT, AT, AVNRT, AVRT, SAAWR | Nhịp nhanh trên thất — HR cao, nhiều cơ chế |
| SB | SBrad | Nhịp chậm xoang — HR thấp, hình thái xoang |
| SR | SR, Sinus Irregularity | Nhịp xoang bình thường / xoang không đều nhẹ |

Quy tắc ánh xạ này được giữ nhất quán cho mọi tập dữ liệu kiểm tra chéo ở Chương 4 (Ningbo,
PTB-XL, Georgia) để bảo đảm nhãn đồng nhất giữa các nguồn — điều kiện tiên quyết cho một
nghiên cứu chuyển giao đa-dataset có ý nghĩa.

**Tiền xử lý tín hiệu.** Từ bản ghi 12 chuyển đạo, chỉ trích xuất **chuyển đạo chi II
(Lead II)** làm đầu vào đơn kênh, theo lập luận đã trình bày ở Mục 2.1.1. Tín hiệu Lead II
được xử lý qua ba bước, tóm tắt ở Bảng 3.2:

**Bảng 3.2 — Các bước tiền xử lý tín hiệu và vai trò.**

| Bước | Thao tác | Vào → Ra | Vai trò |
|------|----------|----------|---------|
| 1 | Lấy mẫu lại 500 → 250 Hz | 5000 → 2500 mẫu | Giảm nửa tính toán/bộ nhớ; vẫn thỏa Nyquist (dải ECG ≤ 40 Hz) |
| 2 | Chuẩn hóa Z-score theo bản ghi | 2500 → 2500 mẫu (float) | Xóa khác biệt gain giữa thiết bị; đồng nhất thang biên độ |
| 3 | Định dạng cửa sổ cố định | 2500 mẫu | Một mẫu đầu vào (input sample) cho mô hình |

Bước lấy mẫu lại đưa cửa sổ 10 giây từ 5000 mẫu còn **2500 mẫu**, vừa giảm khối lượng
tính toán và bộ nhớ phần cứng (Input SRAM chỉ cần 2500 × 8-bit), vừa vẫn bảo toàn dải tần
hữu ích của ECG. Bước chuẩn hóa Z-score đặc biệt quan trọng đối với lượng tử hóa: nó loại
bỏ khác biệt về độ khuếch đại giữa các thiết bị thu (ví dụ Chapman thu bằng GE MUSE với độ
phân giải 4,88 µV/LSB, khác với thiết bị của tập Ningbo) và đưa mọi bản ghi về cùng thang
biên độ, nhờ đó một hệ số `input_shift` cố định (Mục 3.1.3) áp dụng đúng cho mọi mẫu.

**Phân chia tập dữ liệu.** Dữ liệu được chia theo tỉ lệ **70/15/15** (huấn luyện / kiểm
định / kiểm tra) theo nguyên tắc **độc lập bệnh nhân (patient-independent)** — các bản ghi
của cùng một bệnh nhân không xuất hiện đồng thời ở hai tập khác nhau. Nguyên tắc này ngăn
hiện tượng rò rỉ thông tin (data leakage) khiến độ chính xác báo cáo bị thổi phồng, và
phản ánh đúng kịch bản triển khai thực tế nơi mô hình gặp bệnh nhân chưa từng thấy. Do
Chapman gán một `patient_id` cho mỗi bản ghi, việc chia độc lập bệnh nhân được thực hiện
trực tiếp trên trường này.

### 3.1.2. Huấn luyện mô hình

**Kiến trúc mạng đề xuất.** Mô hình là một mạng CNN 1D gồm bốn tầng tích chập nối tiếp,
mỗi tầng theo sau bởi một tầng gộp cực đại, rồi kết thúc bằng gộp trung bình toàn cục và
một tầng kết nối đầy đủ. Cấu hình chi tiết và sự biến đổi kích thước tensor qua từng tầng
cho ở Bảng 3.3.

**Bảng 3.3 — Cấu hình các tầng và biến đổi kích thước tensor của mô hình ECG-1DCNN (bản đã cắt tỉa).**

| Tầng | Kênh vào | Kênh ra | Kernel | Đệm | Gộp | ReLU | Kích thước vào | Kích thước ra | Tham số |
|------|:--------:|:-------:|:------:|:---:|:---:|:----:|:--------------:|:-------------:|:-------:|
| Conv1 | 1 | 4 | 5 | 2 | /5 | Không | 1 × 2500 | 4 × 500 | 24 |
| Conv2 | 4 | 4 | 5 | 2 | /5 | Không | 4 × 500 | 4 × 100 | 84 |
| Conv3 | 4 | 8 | 5 | 2 | /5 | Không | 4 × 100 | 8 × 20 | 168 |
| Conv4 | 8 | 8 | 5 | 2 | /5 | **Có** | 8 × 20 | 8 × 4 | 328 |
| GAP | 8 | 8 | — | — | /4 | — | 8 × 4 | 8 × 1 | 0 |
| FC | 8 | 4 | — | — | — | — | 8 | 4 | 36 |
| **Tổng** | | | | | | | | | **640** |

Số tham số mỗi tầng tính theo `(kênh_vào × kênh_ra × K) + kênh_ra` cho tầng tích chập, và
`(vào × ra) + ra` cho FC. Ví dụ Conv4: `8 × 8 × 5 + 8 = 328`. Tổng cộng 640 tham số.

Ba đặc điểm thiết kế cần lưu ý, tất cả đều xuất phát từ ràng buộc đồng thiết kế phần cứng:

- **ReLU chỉ đặt sau Conv4.** Ba tầng Conv1–Conv3 không dùng hàm kích hoạt phi tuyến sau
  tích chập. Lựa chọn này nhằm **bảo toàn các đặc trưng âm** của tín hiệu ECG (ví dụ sóng
  S, đoạn ST chênh xuống, sóng Q sâu) — vốn mang thông tin chẩn đoán và sẽ bị ReLU cắt bỏ
  nếu áp dụng sớm. Đặc biệt, việc giữ giá trị âm ở các tầng đầu là lý do INT8 (dải
  [−127, 127]) là điểm ngọt: hạ xuống INT4 sẽ không đủ dải động cho các kích hoạt âm biên
  độ lớn này, làm sập độ chính xác (chi tiết định lượng ở Chương 4). Chỉ tại tầng cuối,
  trước khi gộp và phân loại, ReLU mới được dùng để tạo phi tuyến và bảo đảm đầu vào GAP
  không âm — điều kiện để hiện thực GAP bằng phép chia số nguyên trên phần cứng (Mục 3.2.3).
- **Số kênh đầu ra là lũy thừa của 2** (4, 4, 8, 8). Ràng buộc này giúp ánh xạ trực tiếp
  lên tám phần tử xử lý (PE) song song của phần cứng mà không lãng phí tài nguyên hay cần
  logic đệm lẻ. Kênh ra tối đa bằng 8 cũng đúng bằng số cp_block của engine (Mục 3.2.2).
- **Kernel K=5, đệm (padding) 2 ở mọi tầng**, sải bước (stride) 1 cho tích chập và gộp
  cực đại cửa sổ 5 sải bước 5. Kernel lẻ có đệm đối xứng giữ nguyên độ dài chuỗi sau tích
  chập, còn gộp /5 thu gọn chuỗi đúng 5 lần mỗi tầng (2500 → 500 → 100 → 20 → 4).

Mô hình được huấn luyện ở độ chính xác float32 bằng hàm mất mát cross-entropy và bộ tối
ưu Adam, theo lý thuyết đã trình bày ở Mục 2.3. Các siêu tham số huấn luyện cho ở Bảng 3.4.

**Bảng 3.4 — Siêu tham số huấn luyện mô hình cơ sở float32.**

| Siêu tham số | Giá trị |
|--------------|---------|
| Hàm mất mát | Cross-entropy |
| Bộ tối ưu | Adam |
| Kích thước batch | theo cấu hình mặc định pipeline |
| Hàm kích hoạt | ReLU (chỉ Conv4) |
| Khởi tạo trọng số | mặc định PyTorch (Kaiming/uniform) |
| Chỉ tiêu chọn mô hình | độ chính xác trên tập kiểm định |

**Cắt tỉa kênh có cấu trúc.** Mô hình float32 gốc có số kênh lớn hơn và tổng cộng **1244
tham số**. Để giảm kích thước phục vụ triển khai phần cứng, áp dụng **cắt tỉa kênh có cấu
trúc (structured channel pruning)** — loại bỏ trọn vẹn các kênh (bộ lọc) kém quan trọng
thay vì đặt lẻ các trọng số về 0 như cắt tỉa phi cấu trúc. Chỉ cắt tỉa có cấu trúc mới thu
nhỏ được kích thước tensor thật, do đó mới giảm được tài nguyên phần cứng.

Tiêu chí quan trọng của mỗi bộ lọc là **chuẩn L1** của nó [13] (tổng trị tuyệt đối các
trọng số), kết hợp tham chiếu độ quan trọng bậc nhất Taylor [12] (xấp xỉ mức tăng mất mát
khi loại bộ lọc bằng khai triển Taylor bậc nhất quanh trọng số hiện tại). Các bộ lọc có
chuẩn L1 nhỏ nhất — đóng góp yếu nhất vào đặc trưng đầu ra — bị loại bỏ cùng toàn bộ kênh
liên quan. Bảng 3.5 tổng hợp số kênh trước và sau cắt tỉa.

**Bảng 3.5 — Số kênh mỗi tầng trước và sau cắt tỉa có cấu trúc.**

| Tầng | Kênh ra gốc | Kênh ra sau tỉa | Hành động |
|------|:-----------:|:---------------:|-----------|
| Conv1 | 4 | 4 | Giữ nguyên |
| Conv2 | 8 | 4 | Loại 4 bộ lọc yếu nhất |
| Conv3 | 8 | 8 | Giữ nguyên |
| Conv4 | 16 | 8 | Loại 8 bộ lọc yếu nhất |
| FC (đầu vào) | 16 | 8 | Co theo Conv4 |

Sau cắt tỉa, mô hình còn **640 tham số** — giảm 48,6% so với bản gốc. Do cắt tỉa làm giảm
độ chính xác tạm thời, mô hình được **tinh chỉnh lại (fine-tune) theo hai pha**: pha 1 gồm
30 epoch với tốc độ học 1e-3 để phục hồi nhanh sau cú sốc cắt tỉa, pha 2 gồm 20 epoch với
tốc độ học 1e-4 để tinh chỉnh mịn. Lịch hai pha (tốc độ học lớn rồi nhỏ) giúp mô hình vừa
thoát khỏi điểm suy giảm do cắt tỉa vừa hội tụ ổn định. Kết quả là mô hình đã cắt tỉa
`best_model_pruned.pth` với số kênh (4, 4, 8, 8) — chính là cấu hình dùng cho toàn bộ
phần cứng.

### 3.1.3. Lượng tử hóa trọng số

Mô hình float32 đã cắt tỉa được lượng tử hóa về INT8 theo phương pháp **lượng tử hóa nhận
biết huấn luyện với thang lũy thừa mũ hai (power-of-2 QAT)**, theo lý thuyết ở Mục 2.4.
Đây là phương pháp then chốt để loại bỏ hoàn toàn phép nhân trong khâu tái tỉ lệ (rescale)
trên phần cứng. Mục này trình bày *phương pháp*; các số liệu so sánh định lượng giữa
power-of-2 và general-scale, cũng như giữa làm tròn round-half-up và cắt bỏ (floor), được
báo cáo ở Chương 4.

**Chọn hệ số dịch bit.** Mỗi tensor (trọng số từng tầng, kích hoạt từng tầng) được gán
một hệ số dịch bit là số nguyên, chọn theo giá trị tuyệt đối lớn nhất của tensor:

```
shift_bits = floor( log2( 127 / abs_max ) )
```

Ý nghĩa: `2^shift_bits` là lũy thừa hai lớn nhất sao cho giá trị lớn nhất của tensor sau
khi nhân thang vẫn không vượt 127 (giới hạn INT8). Nhờ hệ số là số nguyên, phép nhân với
thang tỉ lệ `2^shift` trở thành phép **dịch bit** thuần túy trên phần cứng, không cần bộ
nhân — đây là toàn bộ động lực của phương pháp.

**Quy trình lượng tử hóa INT8.** Đường ống số học được thực hiện đồng nhất giữa mô hình
Python và RTL, theo bốn bước:

```
(1) Lượng tử đầu vào:   x_int8 = clamp( round(x_float · 2^input_shift), -127, 127 )
(2) Lượng tử trọng số:  w_int8 = clamp( round(w_float · 2^w_shift),     -127, 127 )
(3) Tích chập + bias:   acc_int32 = Σ (x_int8 · w_int8) + bias_scaled
(4) Tái tỉ lệ:          out_int8  = clamp( round_half_up( acc_int32 / 2^nb ), -127, 127 )
```

Các tham số lượng tử theo tầng — kết quả từ hiệu chỉnh (calibration) trên tập huấn luyện —
cho ở Bảng 3.6.

**Bảng 3.6 — Tham số lượng tử hóa power-of-2 theo tầng.**

| Tầng | `w_shift` (thang trọng số) | `nb` (bit tái tỉ lệ) |
|------|:-------------------------:|:--------------------:|
| Conv1 | 6 | 8 |
| Conv2 | 6 | 6 |
| Conv3 | 6 | 6 |
| Conv4 | 7 | 7 |
| FC | 8 | 0 |

Ngoài ra, **`input_shift = 2`** là hệ số dịch bit áp cho tín hiệu ECG đầu vào khi chuyển
sang INT8. Bias được nhân thang trước theo `bias_scaled = round(b_float · 2^nb)` và lưu
dưới dạng INT32 little-endian, để có thể cộng thẳng vào bộ tích lũy 32-bit trước khi dịch.
FC có `nb = 0`, nghĩa là không tái tỉ lệ ở tầng cuối — các logit INT32 thô đi thẳng vào
argmax (argmax bất biến với thang nên điều này không ảnh hưởng kết quả phân lớp).

**Làm tròn round-half-up.** Điểm khác biệt so với các thiết kế dịch-bit trước đây (điển
hình là thiết kế fully-mapped của Liu [7], vốn dùng cắt bỏ phần thập phân — floor
truncation) là phép tái tỉ lệ dùng **làm tròn tới số nguyên gần nhất (round-half-up)**,
hiện thực bằng cách cộng nửa đơn vị của bit bị dịch trước khi dịch:

```
round_half_up(acc) = ( acc + 2^(nb-1) ) >> nb
```

So với floor (`acc >> nb`, luôn làm tròn xuống, tạo sai lệch âm hệ thống), round-half-up
phân bố sai số làm tròn đối xứng quanh 0, nhờ đó giảm độ chệch tích lũy qua bốn tầng. Phép
cộng `2^(nb-1)` này không tốn bộ nhân, chỉ là một hằng số cộng vào — do đó giữ nguyên ưu
điểm "0 DSP cho tái tỉ lệ" của power-of-2, đồng thời cải thiện độ chính xác so với floor
(số liệu ở Chương 4). Trên phần cứng, hằng số làm tròn này còn được gộp thẳng vào số hạng
khởi tạo của bộ tích lũy (Mục 3.2.1) để không nằm trên đường tới hạn.

Lượng tử hóa dùng phương pháp QAT (huấn luyện với lượng tử hóa giả — fake-quant, lan
truyền ngược qua bộ ước lượng thẳng STE ở Mục 2.4). Kết quả là mô hình INT8
`model_qat_int8.pth`. Cần nhấn mạnh rằng đường ống INT8 nói trên được thiết kế **khớp bit
chính xác (bit-exact)** với phần cứng: cùng thứ tự phép tính, cùng quy tắc làm tròn, cùng
ngưỡng bão hòa — đây là nền tảng cho khung kiểm chứng 21 điểm ở Chương 4.

### 3.1.4. Trích xuất trọng số

Bước cuối của giai đoạn phần mềm là chuyển bộ trọng số INT8 và bias INT32 thành tệp nhị
phân theo đúng bố cục mà phần cứng nạp vào. Tệp `flat_weights.hex` chứa **580 giá trị INT8**
thực dùng, **không có dòng chú thích** (để hàm `$readmemh` của Verilog đọc từ byte đầu tiên
mà không lệch địa chỉ).

**Bố cục trọng số tích chập.** Mỗi tầng lưu theo thứ tự "PE-major" (kênh ra ở ngoài cùng):

```
[trọng số INT8: kênh ra (PE) → kênh vào → tap]  rồi  [bias INT32 little-endian]
```

Cách sắp xếp này cho phép mỗi khối xử lý phần cứng (mỗi cp_block phụ trách một kênh ra)
đọc đúng bộ 5 trọng số của kênh mà nó phụ trách. Năm trọng số của một cặp (kênh ra, kênh
vào) được **đóng gói thành một từ 40-bit** (5 × 8-bit), khớp với cách khối engine đọc
trọng số theo từng từ (Mục 3.2.2). Ví dụ, từ 40-bit đầu tiên của Conv1 (kênh ra 0, kênh
vào duy nhất) trong tệp thật là `FE361D3F41`, tương ứng năm trọng số INT8
[−2, 54, 29, 63, 65] theo thứ tự byte — chính là năm hệ số nhân với cửa sổ 5 mẫu ở Mục 3.2.1.
Bias mỗi kênh là INT32, lưu little-endian.

**Bố cục trọng số FC.** Trọng số tầng kết nối đầy đủ lưu theo **kênh ra làm hàng** (4 hàng
× 8 cột), phục vụ vòng nhân-cộng tuần tự của khối GAP/FC/Argmax (Mục 3.2.3).

Đến đây, giai đoạn phần mềm hoàn tất: đầu ra là mô hình INT8 khớp bit với phần cứng và bộ
trọng số đã định dạng sẵn để nạp. Các mục tiếp theo trình bày phần cứng tiêu thụ các dữ
liệu này.

---

## 3.2. Thiết kế hệ thống mạng CNN

Lõi tăng tốc CNN được thiết kế theo kiến trúc **một engine tính toán dùng chung, chia sẻ
thời gian (time-multiplexed single-engine)**: thay vì trải toàn bộ bốn tầng ra phần cứng
riêng biệt (fully-mapped), một khối engine 8 phần tử xử lý được tái sử dụng tuần tự cho cả
bốn tầng tích chập, dưới sự điều phối của một máy trạng thái hữu hạn (FSM). Lựa chọn này
phù hợp với mô hình rất nhỏ (640 tham số): trải toàn bộ sẽ lãng phí tài nguyên vào các
tầng nhỏ (Conv1 chỉ 1 kênh vào), trong khi tái sử dụng engine giữ diện tích thấp mà vẫn
đạt độ trễ đủ nhanh cho ứng dụng theo dõi liên tục.

Sơ đồ luồng dữ liệu tổng thể ở Hình 3.2. Máy tính chủ ghi cửa sổ ECG vào Input SRAM; engine
đọc, tính bốn tầng tích chập tuần tự (Conv1 đọc từ Input SRAM, Conv2–4 đọc từ Ping-Pong
SRAM), rồi khối GAP/FC/Argmax cho ra lớp dự đoán.

```
Host ──Avalon-MM──► Input SRAM (2500×8b, cố định)
                          │ (chỉ Conv1 đọc từ đây)
                          ▼
              ┌────────────────────────────────┐
              │   CP-Engine (8 PE song song)    │
              │   8 × cp_block                  │
              │   Kernel=5, đệm=2, sải bước=1    │
              │   MaxPool cửa sổ 5 sải bước 5    │
              └────────────┬───────────────────┘
                           │ ▲ Ping-Pong SRAM (đệm liên tầng, 2 băng)
                           ▼ │
              ┌────────────────────────────────┐
              │   GAP / FC / Argmax             │
              └────────────┬───────────────────┘
                           ▼
                     result[1:0]  (lớp 0–3)
```

**Hình 3.2 — Sơ đồ luồng dữ liệu tổng thể của lõi tăng tốc CNN.**

Danh sách các module phần cứng và vai trò cho ở Bảng 3.7.

**Bảng 3.7 — Danh sách module RTL của lõi tăng tốc.**

| Module | Vai trò | Trình bày ở |
|--------|---------|:-----------:|
| `cp_mac` | Nhân 5 tap + cây cộng 3 tầng → tổng tích chập | 3.2.1 |
| `cp_accumulate_rescale` | Tích lũy đa kênh, cộng bias, tái tỉ lệ, ReLU | 3.2.1 |
| `cp_pool` | Gộp cực đại cửa sổ 5 | 3.2.1 |
| `cp_block` | Bọc 3 khối trên = một kênh đầu ra hoàn chỉnh | 3.2.1 |
| `cp_engine` | 8 cp_block song song + SRW + kho trọng số | 3.2.2 |
| `gap_fc_argmax` | GAP + FC + Argmax | 3.2.3 |
| `cnn_controller` | FSM điều phối toàn hệ thống | 3.2.4 |
| `ping_pong_sram` | Đệm liên tầng hai băng | 3.2.4 |
| `input_sram` | Bộ nhớ đầu vào 2500 × 8-bit | 3.2.4 |
| `ecg_core` | Bọc toàn bộ lõi (độc lập bus) | 3.2.5 |

**Quy ước Conv4 làm chuẩn tham chiếu.** Một nguyên tắc thiết kế xuyên suốt là **mọi tính
toán về độ rộng bus, độ sâu chuỗi trễ, độ rộng địa chỉ ROM đều lấy Conv4 (kênh vào = 8,
kênh ra = 8) làm chuẩn** — vì đây là tầng có tham số lớn nhất. Các tầng nhỏ hơn (Conv1
kênh vào 1; Conv2, Conv3 kênh vào 4) là tập con: phần cứng chạy đúng Conv4 thì chạy đúng
mọi tầng. Bảng 3.8 liệt kê các tài nguyên được cố định theo Conv4.

**Bảng 3.8 — Các tài nguyên phần cứng cố định theo chuẩn Conv4.**

| Tín hiệu / tài nguyên | Giá trị theo Conv4 | Ở Conv1..3 |
|-----------------------|:------------------:|:----------:|
| Độ rộng bộ đếm kênh `a` | 4-bit (0..7) | dùng chung, kênh vào nhỏ hơn |
| `in_ch` | 8 | 1 / 4 / 4 |
| Độ rộng địa chỉ ROM trọng số | 6-bit (kênh_ra × 8 + kênh_vào, max 63) | nhỏ hơn |
| Chuỗi trễ `a_d5` | 5 nhịp | dùng chung |
| Mặt nạ kênh `cp_en` | 8'hFF (8 kênh ra) | 8'h0F hoặc ít hơn |
| Số tổng bộ tích lũy | 8 tổng cục bộ | ít hơn |

Các tiểu mục sau mô tả lần lượt từng khối, từ khối tính toán nhỏ nhất (cp_block) tới toàn
hệ thống (luồng dữ liệu).

### 3.2.1. Thiết kế khối convolution-pool unit (cp_block)

Khối `cp_block` là đơn vị tính toán cơ sở, phụ trách **một kênh đầu ra**. Nó nhận vào một
cửa sổ 5 mẫu và 5 trọng số tương ứng mỗi nhịp, thực hiện tích chập một chiều, tái tỉ lệ về
INT8, rồi gộp cực đại. Về mặt cấu trúc, khối được tách thành ba khối con (chia thuần cấu
trúc, không đổi logic, đã kiểm chứng khớp bit): `cp_mac` (nhân và cộng cây),
`cp_accumulate_rescale` (tích lũy, cộng bias, tái tỉ lệ, ReLU) và `cp_pool` (gộp cực đại).
Giao diện cổng của `cp_block` cho ở Bảng 3.9.

**Bảng 3.9 — Giao diện cổng chính của cp_block.**

| Cổng | Hướng | Rộng | Ý nghĩa |
|------|:-----:|:----:|---------|
| `x_in` | vào | 40 | 5 mẫu cửa sổ, đóng gói 5×8-bit |
| `w` | vào | 40 | 5 trọng số, đóng gói 5×8-bit |
| `bias_in` | vào | 32 | Bias INT32 đã nhân thang |
| `a_in` | vào | 4 | Bộ đếm kênh, trễ 5 nhịp (`a_d5`) |
| `in_ch` | vào | 4 | Số kênh vào của tầng hiện tại |
| `nb` | vào | 4 | Số bit tái tỉ lệ (max dùng = 8) |
| `relu_en` | vào | 1 | Bật ReLU (chỉ Conv4) |
| `pool_write` | ra | 1 | Xung ghi sang Pong SRAM |
| `pool_out` | ra | 8 | Giá trị INT8 sau gộp |

Đường ống bên trong tổ chức theo chín tầng S1–S9, tóm tắt ở Bảng 3.10.

**Bảng 3.10 — Chín tầng đường ống của cp_block.**

| Tầng | Khối con | Thao tác | Bề rộng dữ liệu |
|:----:|----------|----------|:---------------:|
| S1 | cp_mac | 5 phép nhân 8×8 (5 DSP18) | → 16-bit |
| S2 | cp_mac | Cộng cặp: sum01, sum23 | → 17-bit |
| S3 | cp_mac | Cộng: sum0123 | → 18-bit |
| S4 | cp_mac | Cộng tap thứ 5 → `tree_out` | → 20-bit |
| S5 | cp_accum | Tích lũy đa kênh (+ bias + round gộp) | 32-bit |
| S6 | cp_accum | Dịch phải số học `>>> nb` | 32-bit |
| S7 | cp_accum | Bão hòa về [−127, 127] | → 8-bit |
| S8 | cp_accum | ReLU (chỉ Conv4) | 8-bit |
| S9 | cp_pool | Gộp cực đại cửa sổ 5 | 8-bit |

**S1 — Nhân (MAC).** Năm cặp (mẫu × trọng số) được nhân song song bằng năm bộ nhân
8×8→16-bit có dấu, ánh xạ vào 5 DSP18 của Cyclone V. Vì cả toán hạng đều có dấu (bù hai),
kết quả là tích 16-bit có dấu.

**S2–S4 — Cây cộng (adder tree).** Năm tích được cộng dồn qua ba tầng cộng có thanh ghi.
Bề rộng tăng dần để tránh tràn: hai cặp `sum01 = prod0 + prod1` và `sum23 = prod2 + prod3`
(17-bit), rồi `sum0123 = sum01 + sum23` (18-bit), rồi cộng nốt tap thứ năm được trễ đồng
bộ → `tree_out` (20-bit). Đây là tổng tích chập của một vị trí đầu ra cho **một cặp kênh
vào**. Việc chia cây cộng thành ba tầng thanh ghi giúp rút ngắn đường tổ hợp giữa các thanh
ghi, phục vụ đóng thời gian.

**S5 — Tích lũy đa kênh (accumulate).** Với các tầng có nhiều kênh vào (Conv2–Conv4), các
tổng tích chập của từng kênh vào được cộng dồn qua `in_ch` nhịp thành tích lũy 32-bit hoàn
chỉnh. Đáng chú ý, **bias và hằng số làm tròn `2^(nb-1)` được gộp thẳng vào số hạng khởi
tạo** của bộ tích lũy — cụ thể, ở nhịp kênh vào đầu tiên (khi bộ đếm kênh trễ `a_in = 0`):

```
Nếu a_in == 0:  acc ← tree_out + bias + 2^(nb-1)
Ngược lại:      acc ← acc + tree_out
```

Nhờ gộp như vậy, hai phép cộng (bias và cộng-làm-tròn) không còn nằm trên đường tới hạn ở
tầng tái tỉ lệ phía sau — một tối ưu định thời quan trọng. Phép gộp này **khớp bit tuyệt
đối** với công thức `(acc + bias + round) >> nb` vì phép cộng có tính kết hợp; và các biên
giá trị (bias ≤ 139, round ≤ 128, tích lũy ≤ ~4,19 triệu) không gây tràn số 32-bit có dấu.

**S6 — Dịch tái tỉ lệ (rescale).** Sau khi tích lũy hoàn tất, giá trị 32-bit được **dịch
phải số học `nb` bit** (`>>> nb`). Vì hằng số làm tròn đã gộp ở S5, tầng này thuần túy chỉ
là một bộ dịch thanh (barrel shifter) — **không có bộ nhân nào**. Đây chính là chỗ hiện
thực ưu điểm "0 DSP cho tái tỉ lệ" của phương pháp power-of-2.

**S7 — Bão hòa (clamp).** Kết quả dịch được kẹp về dải INT8 hợp lệ `[−127, 127]` bằng so
sánh và chọn (giá trị > 127 thành 127, < −127 thành −127).

**S8 — ReLU.** Chỉ ở Conv4 (`relu_en = 1`), giá trị âm bị đưa về 0. Các tầng khác đi thẳng
qua không đổi.

**S9 — Gộp cực đại (MaxPool).** Khối `cp_pool` duy trì một bộ so sánh cuốn (rolling
comparator) với bộ đếm `pool_cnt` chạy 0→4: ở giá trị hợp lệ đầu tiên nạp thẳng vào
`max_reg`, các giá trị sau chỉ cập nhật `max_reg` nếu lớn hơn; đến giá trị thứ năm
(`pool_cnt = 4`) phát tín hiệu `pool_write` kèm `max_reg` và reset bộ đếm. Bộ đếm được
glate bởi `compute_en` để không đếm nhầm giá trị rác trong pha mồi SRW.

**Ví dụ chạy tay (Conv1, vị trí đầu chuỗi).** Để minh họa toàn bộ đường ống, xét kênh ra 0
của Conv1 với năm trọng số [−2, 54, 29, 63, 65] (từ 40-bit `FE361D3F41`, Mục 3.1.4) và
cửa sổ đầu vào tại vị trí `t = 0`. Do đệm 2, cửa sổ là [0, 0, x[0], x[1], x[2]] với năm
mẫu đầu của tín hiệu INT8 là x[0..2] = [0, −3, −2] (hai vị trí đầu là đệm 0):

```
acc = (−2)·0 + 54·0 + 29·0 + 63·(−3) + 65·(−2)   (đệm ở hai tap đầu)
    = 0 + 0 + 0 + (−189) + (−130) = −319
```

*(Lưu ý: cách ghép tap–mẫu theo đúng quy ước tương quan chéo của PyTorch; ở đây minh họa
số học tích lũy.)* Sau khi cộng bias và tái tỉ lệ với `nb = 8` (round-half-up):

```
out = clamp( round_half_up(acc_có_bias / 2^8), −127, 127 ) = −1   (= 0xFF INT8)
```

Giá trị này khớp chính xác với điểm vàng (golden) `after_conv1[0] = 0xFF` do mô hình
Python xuất ra — một minh chứng vi mô cho tính khớp-bit sẽ được kiểm chứng có hệ thống ở
Chương 4. Đầu ra gộp cực đại của năm vị trí đầu cũng cho `pool1[0] = 0xFF = −1`, khớp
golden `after_pool1[0]`.

Đường tới hạn của toàn hệ thống nằm ở tầng S6 (dịch tái tỉ lệ); việc gộp hằng số làm tròn
vào S5 và tính trước số hạng làm tròn dưới dạng dây (wire) là hai biện pháp then chốt để
đóng thời gian ở tần số mục tiêu 100 MHz (chi tiết ở Chương 4).

### 3.2.2. Thiết kế khối engine unit (cp_engine)

Khối `cp_engine` đặt **8 khối `cp_block` chạy song song** để tính đồng thời 8 kênh đầu ra,
và bao quanh chúng các cơ chế cấp dữ liệu: cửa sổ thanh ghi trượt, chuỗi trễ đồng bộ, kho
trọng số và cổng ghi có chọn lọc. Đây là khối tiêu thụ phần lớn DSP (8 kênh × 5 nhân = 40
bộ nhân, trong đó Quartus ánh xạ một phần vào DSP18).

**Cửa sổ thanh ghi trượt (Shift-Register Window — SRW).** Mỗi kênh vào có một thanh ghi
trượt 5 ô 8-bit. Dữ liệu ECG (Conv1) hoặc bản đồ đặc trưng (Conv2–4) chảy qua SRW theo
từng nhịp; mỗi khi dịch, mẫu mới nhất vào ô 0, mẫu cũ nhất rời ô 4:

```
   [ô4]←[ô3]←[ô2]←[ô1]←[ô0]←  mẫu mới
   cũ nhất              mới nhất
```

Cơ chế này tạo ra cửa sổ tích chập 5 mẫu mà không cần đọc bộ nhớ nhiều lần cho cùng một
mẫu — mỗi mẫu chỉ đọc một lần rồi trượt qua cửa sổ, tiết kiệm băng thông bộ nhớ (đây là
điểm mấu chốt phân biệt kiến trúc streaming với kiến trúc đọc-lại-nhiều-lần). Có tổng cộng
8 SRW (một cho mỗi kênh vào, đủ cho Conv4). Thứ tự ô vật lý được ánh xạ lại về chỉ số tap
logic tại bộ chọn (MUX) để khớp đúng phép tương quan chéo (cross-correlation) của PyTorch:
`out[t] = Σ_k w[k] · x[t−2+k]`, trong đó tap k=0 ghép với mẫu cũ nhất (x[t−2]) và tap k=4
ghép với mẫu mới nhất (x[t+2]).

**Đệm không (zero-padding) hai đầu.** Để hiện thực đệm 2 mẫu ở đầu và cuối chuỗi, một tín
hiệu `pad_zero` được tạo khi địa chỉ đọc nằm ngoài vùng dữ liệu hợp lệ (nhỏ hơn 2, hoặc
lớn hơn hoặc bằng `in_len + 2`). Tín hiệu này được **ghi thanh ghi trễ đúng một nhịp** để
đồng bộ với độ trễ đọc SRAM đồng bộ một nhịp — khi đó ô SRW nhận giá trị 0 đúng thời điểm.
Việc đệm cả đầu lẫn cuối là cần thiết: thiếu đệm cuối, engine sẽ đọc dữ liệu ngoài vùng
hợp lệ sau khi hết chuỗi.

**Chuỗi trễ đồng bộ.** Đây là chi tiết định thời tinh tế nhất của engine. Đường ống từ bộ
chọn tới thanh ghi tích lũy sâu **đúng 5 nhịp**, phân rã như Bảng 3.11.

**Bảng 3.11 — Độ sâu đường ống 5 nhịp từ bộ chọn tới thanh ghi tích lũy.**

| Nhịp | Sự kiện |
|:----:|---------|
| N | `a` chọn SRW → `mux_comb` (đọc tổ hợp); phát địa chỉ đọc trọng số |
| N+1 | `mux_s1` ← `mux_comb`; `w_packed` ← trọng số (đọc đồng bộ) |
| N+2 | `prod` ← `mux_s1 × w_packed` (S1 nhân) |
| N+3 | `sum01`, `sum23` (S2) |
| N+4 | `sum0123` (S3) |
| N+5 | `tree_out` (S4) — cạnh cập nhật tích lũy đọc `a_d5`/`ce_d5`/`inch_d5` |

Do độ sâu này, các tín hiệu điều khiển đi kèm — bộ đếm kênh `a`, số kênh vào `in_ch`, tín
hiệu cho phép tính `compute_en` — đều được **trễ đồng bộ 5 nhịp** để tới đúng thời điểm mà
thanh ghi tích lũy cập nhật ở nhịp N+5, khớp với giá trị `a` đã điều khiển `mux_comb` ở
nhịp N. Việc trễ được hiện thực bằng ba **chuỗi thanh ghi dịch 5 tầng** (mỗi tầng một nhịp),
đặt tên theo quy ước "tín-hiệu-gốc + `_d` + số-tầng-trễ":

- `a_d1 → a_d2 → … → a_d5`: bản trễ 1..5 nhịp của bộ đếm kênh `a`; `a_d5` (trễ đúng 5 nhịp)
  là giá trị được khối tích lũy đọc để biết tổng tích chập đang tới thuộc kênh vào nào.
- `inch_d1 → … → inch_d5`: bản trễ của số kênh vào `in_ch` — cho khối tích lũy biết khi nào
  đã cộng đủ số kênh của tầng để chốt kết quả.
- `ce_d1 → … → ce_d5`: bản trễ của tín hiệu cho phép tính `compute_en` — chỉ cho phép cập
  nhật thanh ghi tích lũy với dữ liệu hợp lệ (loại giá trị rác trong pha mồi SRW).

Ba chuỗi đều là thanh ghi dịch thuần: mỗi nhịp, giá trị tầng trước chuyển sang tầng sau
(`a_d5 ← a_d4 ← … ← a_d1 ← a`). Chỉ ba tín hiệu ở tầng cuối (`a_d5`, `inch_d5`, `ce_d5`)
được nối vào khối tích lũy; các tầng trung gian chỉ để tạo độ trễ. Sai lệch dù chỉ một nhịp ở chuỗi trễ này sẽ khiến
cửa sổ gộp dịch sai vị trí — đây từng là nguyên nhân của một lỗi khớp-bit đã được xác định
bằng dò cấp-nhịp và sửa trong quá trình kiểm chứng (chi tiết ở Chương 4).

**Kho trọng số.** Ở bản thiết kế chính của khóa luận (nạp một lần), trọng số được nạp sẵn
vào bốn ROM mảng thanh ghi (một cho mỗi tầng) qua `$readmemh` và bake thẳng vào bitstream;
topology Chapman cố định trong bộ điều khiển. Kho trọng số này (`cp_weight_store`) được
đánh chỉ số theo `layer_state` và bộ đếm kênh `a`, cho ra từ trọng số 40-bit đúng một nhịp
trước khi khối nhân cần (khớp thời điểm N+1 ở Bảng 3.11). Do trọng số bake thẳng vào
bitstream, kho này không có cổng ghi từ bus — topology và bộ trọng số Chapman là cố định
cho toàn thiết kế.

**Cổng ghi có chọn lọc.** Không phải mọi tầng đều dùng đủ 8 kênh ra (Conv1, Conv2 chỉ dùng
4). Tín hiệu ghi `pool_write` của mỗi khối được **AND với mặt nạ kênh `cp_en`** trước khi
ghi vào Ping-Pong SRAM, nên chỉ các kênh đang hoạt động mới ghi kết quả. Dữ liệu ghi và
tín hiệu cho phép ghi được đóng gói theo kênh (`pong_din[ch*8 +: 8]`, `pong_we[ch]`).

### 3.2.3. Thiết kế khối gap-fc-argmax unit

Khối `gap_fc_argmax` thực hiện ba bước cuối của mạng — gộp trung bình toàn cục, kết nối
đầy đủ, và tìm lớp cực đại — dưới dạng một máy trạng thái con tuần tự, tổng cộng **22 nhịp**
sau khi Conv4 kết thúc. Phân bổ nhịp cho ở Bảng 3.12. Về cấu trúc, khối gồm ba khối con:
`gap_unit`, `fc_unit`, `argmax_unit`.

**Bảng 3.12 — Phân bổ nhịp của khối GAP/FC/Argmax.**

| Giai đoạn | Số nhịp | Thao tác |
|-----------|:-------:|----------|
| GAP | 6 | Cộng dồn + chia trung bình 8 kênh |
| FC | 10 | Nhân-cộng 8 đầu vào × 4 kênh ra |
| FC flush | 1 | Xả thanh ghi tích lũy |
| Argmax | 4 | So sánh 4 logit |
| Done | 1 | Chốt kết quả |
| **Tổng** | **22** | |

**GAP — Gộp trung bình toàn cục.** Với mỗi trong 8 kênh, cộng dồn 4 giá trị đầu ra của
Conv4 rồi chia trung bình. Do đầu vào GAP đã qua ReLU nên luôn không âm, phép chia trung
bình được hiện thực bằng **phép chia số nguyên làm tròn xuống (floor): `sum/4 = sum >> 2`**
— rẻ hơn nhiều so với chia số thực mà vẫn đủ chính xác. Mô hình Python vàng (golden) cũng
dùng đúng `floor(sum/4)` để khớp bit với phần cứng (đây là một trong các điểm từng gây lệch
giữa Python và RTL, đã hợp nhất về floor, chi tiết ở Chương 4).

**FC — Kết nối đầy đủ.** Vector 8 giá trị GAP được nhân-cộng với ma trận trọng số FC (4
kênh ra × 8 đầu vào) để tạo ra 4 logit. Vì `nb_fc = 0` nên **không có tái tỉ lệ ở FC**:
các logit INT32 thô được đưa thẳng sang bước tìm cực đại. Bias FC (đã nhân thang theo
`2^w_shift[fc]` để cùng thang với tích lũy) được nạp sẵn vào tích lũy trước khi thực hiện
các phép nhân-cộng.

**Argmax.** So sánh 4 logit để chọn chỉ số lớn nhất, cho ra `result[1:0]` là lớp dự đoán
(0–3). Vì argmax bất biến với thang tỉ lệ, việc bỏ tái tỉ lệ ở FC không ảnh hưởng kết quả.

**Ví dụ chạy tay (FC + Argmax).** Với mẫu kiểm tra thật (chỉ số 0, nhãn thật = lớp 3 = SR),
vector GAP 8 kênh sau Conv4 là [67, 9, 10, 27, 17, 18, 22, 28] (INT8). Sau nhân-cộng FC và
cộng bias, bốn logit INT32 thu được (giá trị vàng thật):

```
logit[0] = 2169   (AFIB)
logit[1] = 2137   (GSVT)
logit[2] = 2017   (SB)
logit[3] = 4581   (SR)   ← lớn nhất
```

Argmax chọn chỉ số 3 → lớp **SR**, khớp đúng nhãn thật. Bốn logit này cũng khớp bit chính
xác với logit vàng do mô hình Python xuất ra, xác nhận toàn bộ đường ống — từ tích chập,
tái tỉ lệ, GAP đến FC — nhất quán giữa phần mềm và phần cứng.

### 3.2.4. Thiết kế khối điều khiển và các khối phụ trợ

**Bộ điều khiển FSM (cnn_controller).** Đây là "nhạc trưởng" điều phối toàn bộ đường ống,
hiện thực bằng **một máy trạng thái hữu hạn thống nhất** — điều khiển tuần tự cả `cp_engine`
(bốn tầng tích chập) lẫn `gap_fc_argmax` (GAP/FC/Argmax) trong cùng một FSM, thay vì tách
thành nhiều bộ điều khiển. Giản đồ trạng thái mức trên ở Hình 3.3.

```
        start
  IDLE ────────► LOAD_INPUT ──► CONV1 ──► CONV2 ──► CONV3 ──► CONV4
   ▲                                                            │
   │                                                            │ layer_done
   │                                                            ▼
   │                                                         GAP_FC_S
   │                              done                          │  (5 sub-state)
   └──────────────── DONE_S ◄─────────────────────────────────┘
                        │  (nhận start mới → chạy lại, phục vụ dòng liên tục)
```

**Hình 3.3 — Giản đồ trạng thái của cnn_controller.**

**Tám trạng thái chính.** FSM gồm tám trạng thái, mã hóa 3-bit, mô tả ở Bảng 3.13.

**Bảng 3.13 — Tám trạng thái của cnn_controller.**

| Trạng thái | Mã | Vai trò |
|------------|:--:|---------|
| `IDLE` | 0 | Chờ; `busy=0`. Nhận `start` → sang LOAD_INPUT |
| `LOAD_INPUT` | 1 | Nạp tham số Conv1, phát `srw_rst`/`pool_rst`, rồi sang CONV1 ngay nhịp sau |
| `CONV1` | 2 | Tích chập tầng 1 (đọc Input SRAM) |
| `CONV2` | 3 | Tích chập tầng 2 (đọc Ping-Pong) |
| `CONV3` | 4 | Tích chập tầng 3 |
| `CONV4` | 5 | Tích chập tầng 4 (có ReLU) |
| `GAP_FC_S` | 6 | Điều phối GAP → FC → Argmax qua năm luồng con |
| `DONE_S` | 7 | Chốt `result`, phát `done`; nhận `start` mới → chạy lại |

**Hai bộ đếm lồng nhau.** Trong các trạng thái CONV, bộ điều khiển duy trì hai bộ đếm: bộ
đếm kênh vào `a` (chạy 0 tới `in_ch − 1` rồi quay vòng) và bộ đếm vị trí đầu ra `t`. Mỗi khi
`a` quay vòng — thời điểm phát tín hiệu `shift_en = (a == in_ch − 1)` — thì `t` tăng một và
SRW trượt một nhịp. Như vậy engine dành đúng `in_ch` nhịp cho mỗi vị trí đầu ra (mỗi nhịp
một kênh vào), khớp với cơ chế tích lũy đa kênh của cp_block.

**Tham số theo tầng.** Bộ điều khiển phát các tham số điều khiển của tầng hiện tại — `in_ch`,
`in_len`, `out_len`, `nb`, `relu_en`, `cp_en` — và nạp lại toàn bộ tại mỗi lần chuyển tầng.
Ở bản thiết kế chính (nạp một lần), các tham số này là **hằng số cứng cho topology Chapman**,
hiện thực bằng các hàm tra cứu theo chỉ số tầng. Bảng 3.14 tổng hợp.

**Bảng 3.14 — Tham số điều khiển theo tầng (topology Chapman cố định).**

| Tầng | `in_ch` | `in_len` | `out_len` | `nb` | `relu_en` | `cp_en` |
|------|:-------:|:--------:|:---------:|:----:|:---------:|:-------:|
| Conv1 | 1 | 2500 | 500 | 8 | 0 | 0x0F |
| Conv2 | 4 | 500 | 100 | 6 | 0 | 0x0F |
| Conv3 | 4 | 100 | 20 | 6 | 0 | 0xFF |
| Conv4 | 8 | 20 | 4 | 7 | 1 | 0xFF |

**Chuyển tầng (`layer_done`) và đảo băng.** Bộ điều khiển phát hiện kết thúc một tầng qua
tín hiệu `layer_done = (pong_addr == out_len − 1) && pool_write`, tức khi giá trị gộp cuối
cùng của tầng vừa được ghi. Tại nhịp `layer_done`, bộ điều khiển đồng thời: (i) nạp tham số
tầng kế tiếp; (ii) reset các bộ đếm `a`, `t`, `pong_addr` về 0 và bộ đếm nạp trước
`prefetch_cnt` về 0; (iii) phát `srw_rst`/`pool_rst` để xóa cửa sổ và bộ đếm gộp; (iv)
**đảo băng Ping-Pong** bằng `bank_sel <= ~bank_sel` — băng vừa ghi của tầng này trở thành
băng đọc của tầng sau. Riêng khi Conv4 kết thúc, thay vì sang một tầng CONV mới, FSM
chuyển sang `GAP_FC_S` và khởi động luồng con đầu tiên (GAP).

**Điều phối GAP/FC/Argmax bằng năm luồng con.** Trạng thái `GAP_FC_S` không phải một khối
đơn: bên trong nó, bộ điều khiển chạy một FSM con năm bước điều khiển `gap_fc_argmax` theo
đúng số nhịp mỗi bước (khớp Bảng 3.12), tổng cộng 22 nhịp:

| Luồng con | Nhịp | Bộ đếm điều khiển |
|-----------|:----:|-------------------|
| `GAP_SUB` | 6 | `gap_step` 0→5 |
| `FC_SUB` | 10 | `fc_step` 0→9 |
| `FC_FLUSH_S` | 1 | — (xả tích lũy) |
| `ARGMAX_SUB` | 4 | `argmax_step` 0→3 |
| `DONE_SUB` | 1 | chốt `result ← argmax_result`, phát `done`, sang DONE_S |

Việc gộp điều phối hậu xử lý vào chính bộ điều khiển chung (thay vì một FSM riêng trong
`gap_fc_argmax`) giữ cho khối GAP/FC/Argmax thuần túy là datapath, còn toàn bộ định thời tập
trung một chỗ — dễ kiểm chứng và dễ suy luận về số nhịp tổng.

Hai chi tiết định thời tinh tế đáng lưu ý:

- **Đếm nạp trước (prefetch).** Trước khi cho phép tính, bộ điều khiển phải mồi SRW bằng
  đúng 5 nhịp trượt (2 nhịp đệm + 3 nhịp dữ liệu) để cửa sổ chứa đúng vị trí đầu ra thứ
  nhất. Cụ thể, sau 5 nhịp trượt SRW = [x[2], x[1], x[0], 0, 0]; bộ chọn với `a=0` cho ra
  [0, 0, x[0], x[1], x[2]] = đúng cửa sổ đầu ra vị trí 0. Bộ đếm `prefetch_cnt` đếm các
  nhịp trượt **thật** (được gate bởi `!srw_rst`) và chỉ nâng `compute_en` sau nhịp thứ 5.
  Việc gate này cần thiết vì ở nhịp `srw_rst`, tín hiệu `shift_en` có thể bằng 1 (ở Conv1
  do `in_ch=1`) nhưng SRW **không** thực sự trượt — nếu không gate sẽ đếm thừa một nhịp.
- **Nhịp tim ghi gộp (heartbeat).** Do mọi kênh hoạt động ghi đồng thời, bộ điều khiển
  dùng tín hiệu ghi của kênh 0 làm đại diện để đếm số giá trị đã gộp (`pong_addr`) và xác
  định thời điểm kết thúc tầng (`layer_done`). Ràng buộc kèm theo là **kênh 0 phải luôn
  hoạt động** ở mọi tầng (mọi giá trị `cp_en` trong Bảng 3.14 đều có bit 0 = 1).

**Chế độ dòng liên tục.** Ở trạng thái `DONE_S`, bộ điều khiển giữ nguyên `result` nhưng vẫn
**chấp nhận một xung `start` mới** để quay lại LOAD_INPUT mà không cần reset — cho phép chạy
liên tiếp nhiều cửa sổ ECG (kịch bản theo dõi dòng liên tục) mà không mất chu kỳ khởi tạo.

**Ping-Pong SRAM (ping_pong_sram).** Bộ nhớ đệm liên tầng gồm **hai băng (bank) song công**,
mỗi băng 8 kênh × 512 mục × 8-bit (đệm 500 lên 512 cho vừa chế độ M10K). Trong khi tầng
hiện tại đọc từ băng này (băng "Ping") thì ghi kết quả sang băng kia (băng "Pong"); tại
mỗi chuyển tầng, vai trò hai băng đảo lẫn nhau (`bank_sel` lật). Cơ chế đệm kép này cho
phép tầng sau bắt đầu đọc ngay đầu ra tầng trước mà không tranh chấp truy cập, và lưu đồng
thời 8 kênh đặc trưng. Cổng đọc đồng bộ (trễ 1 nhịp), cổng ghi có `we` riêng theo kênh.
Bộ chọn băng ở đầu ra là tổ hợp (1 mức LUT) áp lên dữ liệu đã ghi thanh ghi, giữ đường tới
hạn ngắn. Toàn bộ 16 mảng nhớ (8 kênh × 2 băng) ánh xạ vào 16 khối M10K.

**Input SRAM (input_sram).** Bộ nhớ đầu vào **2500 × 8-bit** lưu một cửa sổ ECG, hiện thực
song-cổng đơn giản: cổng ghi từ máy tính chủ, cổng đọc đồng bộ (trễ 1 nhịp) cho engine. Máy
tính chủ ghi dữ liệu vào đây qua bus trước khi phát lệnh bắt đầu; chỉ Conv1 đọc trực tiếp
từ bộ nhớ này (Conv2–4 đọc từ Ping-Pong). Bộ nhớ ánh xạ vào khối M10K của Cyclone V.

Bảng 3.16 tổng hợp tổ chức bộ nhớ toàn hệ thống.

**Bảng 3.16 — Tổ chức bộ nhớ trên chip.**

| Bộ nhớ | Kích thước | Cổng | Ánh xạ | Người ghi | Người đọc |
|--------|:----------:|------|:------:|-----------|-----------|
| Input SRAM | 2500 × 8b | song-cổng đơn giản | M10K | Host (qua bus) | Conv1 |
| Ping-Pong (2 băng) | 8 ch × 512 × 8b | đọc/ghi song công | 16 × M10K | Conv tầng L | Conv tầng L+1 / GAP |
| Kho trọng số (ROM) | 116 từ × 40b | đọc đồng bộ | mảng FF | $readmemh | cp_engine |
| Kho bias (ROM) | 32 × 32b | đọc đồng bộ | MLAB | $readmemh | cp_engine |

### 3.2.5. Luồng dữ liệu của hệ thống mạng CNN

Kết hợp các khối trên, một lượt suy luận (inference) diễn ra theo năm bước:

1. **Nạp đầu vào.** Máy tính chủ ghi 2500 mẫu ECG (INT8) vào Input SRAM, rồi phát xung
   `start`. FSM chuyển từ IDLE sang LOAD_INPUT rồi CONV1.
2. **Conv1.** Engine đọc tuần tự từ Input SRAM qua SRW, tính 4 kênh ra, gộp cực đại /5, và
   ghi 500 giá trị × 4 kênh sang một băng Ping-Pong. Vì Conv1 chỉ 1 kênh vào, không có tích
   lũy đa kênh (mỗi vị trí đầu ra hoàn tất trong một nhịp `a`).
3. **Conv2 → Conv3 → Conv4.** Mỗi tầng đọc bản đồ đặc trưng của tầng trước từ Ping-Pong,
   tính tích chập đa kênh (tích lũy qua `in_ch` nhịp cho mỗi vị trí), tái tỉ lệ, gộp cực
   đại, và ghi sang băng còn lại. Conv4 thêm ReLU. Kích thước chuỗi thu gọn dần: 500 → 100
   → 20 → 4.
4. **GAP/FC/Argmax.** Sau Conv4, khối gap_fc_argmax gộp trung bình 8 kênh (mỗi kênh 4 giá
   trị), nhân-cộng FC ra 4 logit, và tìm lớp cực đại.
5. **Xong.** FSM phát xung `done` và chốt `result[1:0]`. Máy tính chủ đọc kết quả qua bus.

**Độ trễ.** Toàn bộ lượt suy luận là **tất định (deterministic)**: mỗi tầng tốn số nhịp cố
định, tổng hợp ở Bảng 3.17.

**Bảng 3.17 — Ước lượng số nhịp mỗi giai đoạn (bậc độ lớn).**

| Giai đoạn | Số nhịp xấp xỉ | Ghi chú |
|-----------|:--------------:|---------|
| Conv1 | ~2500 | 1 kênh vào × 2500 vị trí |
| Conv2 | ~2000 | 4 kênh vào × 500 vị trí |
| Conv3 | ~400 | 4 kênh vào × 100 vị trí |
| Conv4 | ~160 | 8 kênh vào × 20 vị trí |
| GAP/FC/Argmax | 22 | cố định |
| Chuyển tầng + mồi | ~134 | mồi SRW mỗi tầng |
| **Tổng** | **~5216** | |

Con số độ trễ chính xác (đo bằng mô phỏng) và thông lượng tương ứng được báo cáo ở Chương 4.

Một tối ưu bổ sung đã áp dụng là **nạp chồng lấp (overlap reload)**: dữ liệu ECG của lượt
suy luận kế tiếp có thể được nạp vào Input SRAM song song với quá trình tính toán của lượt
hiện tại (do chỉ Conv1 dùng Input SRAM, và chỉ ở đầu chuỗi), giúp giảm thời gian chờ giữa
các lượt liên tiếp trong kịch bản dòng liên tục.

**Lõi độc lập bus (ecg_core).** Toàn bộ các khối trên được bọc trong module `ecg_core`,
phơi ra một giao diện tối giản gồm: cổng đọc Input SRAM (`input_rd_addr`/`input_dout`), tín
hiệu điều khiển (`start`/`busy`/`done`) và `result[1:0]`. Việc tách lõi độc lập với bus cho
phép cùng một lõi tính toán tái dùng dưới nhiều lớp bao bus khác nhau (Avalon-MM, JTAG) mà
không sửa đổi — là cơ sở cho phần tích hợp ở Mục 3.3.

### 3.2.6. Sơ đồ và giao diện cổng của từng module

Để tiện tra cứu và tổng hợp thiết kế, mục này trình bày sơ đồ khối tổng quát cùng bảng giao
diện cổng của từng module trong lõi tính toán (tương tự Bảng 3.9 đã lập cho `cp_block`). Mọi
tên tín hiệu và bề rộng lấy đúng theo mã RTL của bản thiết kế chính (nạp trọng số một lần
qua ROM); các cổng ghi trọng số từ bus và cấu hình topology thời gian chạy thuộc biến thể
nạp lại trọng số, không liệt kê ở đây. Cây phân cấp module như Hình 3.4a.

```
ecg_core                          — lõi tính toán độc lập bus
├── cp_engine                     — 8 CP block song song + SRW + kho trọng số
│   ├── cp_weight_store           — ROM trọng số + bias (nạp một lần)
│   └── cp_block × 8              — mỗi block một kênh ra
│       ├── cp_mac                — S1–S4  nhân + cây cộng
│       ├── cp_accumulate_rescale — S5–S8  tích lũy + bias + tái tỉ lệ + ReLU
│       └── cp_pool               — S9     gộp cực đại
├── ping_pong_sram                — đệm bản đồ đặc trưng liên tầng (2 băng)
├── gap_fc_argmax                 — GAP → FC → Argmax
└── cnn_controller                — FSM điều phối Conv1–4 + GAP/FC
```

**Hình 3.4a — Cây phân cấp module của lõi tính toán (`input_sram` nằm ở wrapper).**

**Khối `cp_mac` (S1–S4).** Đường dữ liệu nhân-cộng thuần feed-forward: 5 bộ nhân có dấu
8×8 → cây cộng ba tầng → `tree_out` 20-bit.

```
 x_in[39:0] ─┐
             ├─► [5× MUL 8×8] ─► [cây cộng 3 tầng S2–S4] ─► tree_out[19:0]
   w[39:0] ─┘
```

**Bảng 3.9a — Giao diện cổng của cp_mac.**

| Cổng | Hướng | Rộng | Ý nghĩa |
|------|:-----:|:----:|---------|
| `clk` | vào | 1 | Xung nhịp |
| `x_in` | vào | 40 | 5 mẫu cửa sổ, đóng gói 5×8-bit |
| `w` | vào | 40 | 5 trọng số, đóng gói 5×8-bit |
| `tree_out` | ra | 20 | Tổng tích chập (mở rộng dấu Σ 5 tích) |

**Khối `cp_accumulate_rescale` (S5–S8).** Nhận `tree_out`, tích lũy đa kênh với bias và
hằng số làm tròn gộp sẵn ở nhịp đầu, dịch tái tỉ lệ `>>> nb`, bão hòa INT8, rồi ReLU
(chỉ Conv4).

```
 tree_out ─►[+ acc]─►[>>> nb]─►[clamp ±127]─►[ReLU]─► relu_out[7:0], relu_v
 bias_in ──►(gộp vào acc-init khi a_in==0)
```

**Bảng 3.9b — Giao diện cổng của cp_accumulate_rescale.**

| Cổng | Hướng | Rộng | Ý nghĩa |
|------|:-----:|:----:|---------|
| `clk` | vào | 1 | Xung nhịp |
| `rst` | vào | 1 | Reset |
| `pool_rst` | vào | 1 | Reset khi chuyển tầng |
| `tree_out` | vào | 20 | Tổng tích chập từ cp_mac |
| `bias_in` | vào | 32 | Bias INT32 đã nhân thang |
| `a_in` | vào | 4 | Bộ đếm kênh, trễ 5 nhịp |
| `in_ch` | vào | 4 | Số kênh vào của tầng |
| `compute_en_in` | vào | 1 | Cho phép đường ống, trễ 5 nhịp |
| `nb` | vào | 4 | Số bit tái tỉ lệ (max dùng = 8) |
| `relu_en` | vào | 1 | Bật ReLU (chỉ Conv4) |
| `relu_out` | ra | 8 | Kích hoạt INT8 |
| `relu_v` | ra | 1 | Cờ hợp lệ |

**Khối `cp_pool` (S9).** Bộ so sánh cuốn giữ giá trị cực đại qua cửa sổ 5 mẫu hợp lệ; đến
mẫu thứ năm phát `pool_write` kèm `pool_out` rồi reset bộ đếm.

**Bảng 3.9c — Giao diện cổng của cp_pool.**

| Cổng | Hướng | Rộng | Ý nghĩa |
|------|:-----:|:----:|---------|
| `clk` | vào | 1 | Xung nhịp |
| `rst` | vào | 1 | Reset |
| `pool_rst` | vào | 1 | Reset khi chuyển tầng |
| `relu_out` | vào | 8 | Kích hoạt INT8 từ khối tái tỉ lệ |
| `relu_v` | vào | 1 | Cờ hợp lệ |
| `compute_en_in` | vào | 1 | Cổng loại giá trị rác pha mồi |
| `pool_write` | ra | 1 | Xung ghi (AND với `cp_en` ngoài module) |
| `pool_out` | ra | 8 | Giá trị INT8 sau gộp → Pong SRAM |

**Khối `cp_engine` (8 PE).** Sở hữu mảng SRW, bộ chọn tap, chuỗi trễ `a_d5`, sinh địa chỉ
đọc SRAM (`t−2`), kho trọng số và cổng ghi gộp `pong_we = pool_write & cp_en`.

```
 input_sram_dout ─┐                                   ┌─► pong_din[63:0]
 ping_dout[63:0] ─┼─►[SRW ×8]─►[MUX tap]─►[cp_block ×8]┼─► pong_we[7:0]
     điều khiển ──┘   (a, in_ch, nb, cp_en, ...)       └─► sram_rd_addr[11:0]
```

**Bảng 3.9d — Giao diện cổng của cp_engine.**

| Cổng | Hướng | Rộng | Ý nghĩa |
|------|:-----:|:----:|---------|
| `clk` / `rst` | vào | 1 | Xung nhịp / reset |
| `a` | vào | 4 | Bộ đếm kênh 0..in_ch−1 |
| `in_ch` | vào | 4 | Số kênh vào của tầng |
| `in_len` | vào | 12 | Chiều dài chuỗi vào (2500/500/100/20) |
| `shift_en` | vào | 1 | = (a == in_ch−1), lệnh trượt SRW |
| `srw_rst` | vào | 1 | Xóa SRW khi chuyển tầng |
| `compute_en` | vào | 1 | Cho phép tính (0 khi mồi) |
| `nb` | vào | 4 | Số bit tái tỉ lệ của tầng |
| `relu_en` | vào | 1 | Bật ReLU (chỉ Conv4) |
| `cp_en` | vào | 8 | Mặt nạ kênh ra hoạt động |
| `layer_state` | vào | 3 | Tầng hiện tại (CONV1=2..CONV4=5) |
| `pool_rst` | vào | 1 | Reset bộ đếm gộp khi chuyển tầng |
| `input_sram_dout` | vào | 8 | Dữ liệu từ Input SRAM (chỉ Conv1) |
| `ping_dout` | vào | 64 | Dữ liệu từ Ping (8 kênh đóng gói) |
| `sram_rd_addr_in` | vào | 12 | Địa chỉ gốc từ controller (= t) |
| `pong_din` | ra | 64 | Dữ liệu ghi (8 kênh đóng gói) |
| `pong_we` | ra | 8 | Cho phép ghi theo từng kênh |
| `sram_rd_addr` | ra | 12 | Địa chỉ đọc → Input/Ping-Pong SRAM |

**Khối `cnn_controller` (FSM).** Điều phối toàn đường ống: dẫn `cp_engine` cho Conv1–4 rồi
`gap_fc_argmax` cho GAP/FC/Argmax. Giao diện chính ở Bảng 3.9e (các cổng cấu hình topology
thời gian chạy `cfg_*` thuộc biến thể nạp lại trọng số, không liệt kê).

**Bảng 3.9e — Giao diện cổng chính của cnn_controller.**

| Cổng | Hướng | Rộng | Ý nghĩa |
|------|:-----:|:----:|---------|
| `clk` / `rst` | vào | 1 | Xung nhịp / reset đồng bộ |
| `start` | vào | 1 | Xung khởi động 1 nhịp |
| `pool_write` | vào | 1 | Nhịp tim ghi từ cp_engine (kênh 0) |
| `argmax_result` | vào | 2 | Lớp argmax từ gap_fc_argmax |
| `a` | ra | 4 | Bộ đếm kênh 0..in_ch−1 |
| `t` | ra | 12 | Bộ đếm vị trí đầu ra |
| `shift_en` | ra | 1 | Lệnh trượt SRW |
| `srw_rst` | ra | 1 | Xóa SRW |
| `compute_en` | ra | 1 | Cho phép tính |
| `in_ch` / `in_len` / `nb` | ra | 4/12/4 | Tham số tầng hiện tại |
| `relu_en` | ra | 1 | Bật ReLU (Conv4) |
| `cp_en` | ra | 8 | Mặt nạ kênh ra |
| `bank_sel` | ra | 1 | Chọn băng Ping/Pong |
| `pong_addr` | ra | 12 | Địa chỉ ghi Pong |
| `pool_rst` | ra | 1 | Reset bộ đếm gộp |
| `fc_sub_state` / `gap_step` / `fc_step` / `argmax_step` | ra | 3/4/4/2 | Điều khiển gap_fc_argmax |
| `layer_state` | ra | 3 | Tầng hiện tại |
| `busy` | ra | 1 | 1 khi đang chạy |
| `done` | ra | 1 | Xung hoàn tất 1 nhịp |
| `result` | ra | 2 | Lớp dự đoán đã chốt |

**Khối `gap_fc_argmax`.** Máy trạng thái con tuần tự GAP → FC → Argmax, nhận dữ liệu Conv4
từ Ping SRAM, cho ra `result[1:0]`.

**Bảng 3.9f — Giao diện cổng chính của gap_fc_argmax.**

| Cổng | Hướng | Rộng | Ý nghĩa |
|------|:-----:|:----:|---------|
| `clk` / `rst` | vào | 1 | Xung nhịp / reset |
| `fc_sub_state` | vào | 3 | Trạng thái con GAP/FC/flush/Argmax/Done |
| `gap_step` | vào | 4 | Bước GAP 0..5 |
| `fc_step` | vào | 4 | Bước FC 0..9 |
| `argmax_step` | vào | 2 | Bước Argmax 0..3 |
| `ping_dout` | vào | 64 | Đầu ra Conv4 (8 kênh đóng gói) |
| `gap_rd_addr` | ra | 9 | Địa chỉ đọc Ping SRAM |
| `result` | ra | 2 | Chỉ số lớp argmax |

**Khối `ping_pong_sram`.** Đệm bản đồ đặc trưng hai băng, mỗi băng 8 kênh × 512 mục × 8-bit;
`bank_sel` hoán vai trò đọc (Ping) ↔ ghi (Pong).

**Bảng 3.9g — Giao diện cổng của ping_pong_sram.**

| Cổng | Hướng | Rộng | Ý nghĩa |
|------|:-----:|:----:|---------|
| `clk` | vào | 1 | Xung nhịp |
| `bank_sel` | vào | 1 | Chọn băng đọc/ghi |
| `wr_addr` | vào | 9 | Địa chỉ ghi Pong (0..499) |
| `din` | vào | 64 | Dữ liệu ghi (8 kênh đóng gói) |
| `we` | vào | 8 | Cho phép ghi theo từng kênh |
| `rd_addr` | vào | 9 | Địa chỉ đọc Ping (0..499) |
| `dout` | ra | 64 | Dữ liệu đọc (8 kênh đóng gói, trễ 1 nhịp) |

**Khối `input_sram`.** Bộ nhớ đầu vào 2500 × 8-bit, song-cổng đơn giản; nằm ở wrapper, lõi
chỉ đọc.

**Bảng 3.9h — Giao diện cổng của input_sram.**

| Cổng | Hướng | Rộng | Ý nghĩa |
|------|:-----:|:----:|---------|
| `clk` | vào | 1 | Xung nhịp |
| `wr_addr` | vào | 12 | Địa chỉ ghi (0..2499) |
| `din` | vào | 8 | Dữ liệu ghi (từ host) |
| `we` | vào | 1 | Cho phép ghi |
| `rd_addr` | vào | 12 | Địa chỉ đọc (0..2499) |
| `dout` | ra | 8 | Dữ liệu đọc (trễ 1 nhịp) |

**Lõi `ecg_core`.** Giao diện đối ngoại tối giản của lõi tính toán, phơi ra cho lớp bao bus.

**Bảng 3.9i — Giao diện cổng của ecg_core.**

| Cổng | Hướng | Rộng | Ý nghĩa |
|------|:-----:|:----:|---------|
| `clk` | vào | 1 | Xung nhịp |
| `rst` | vào | 1 | Reset đồng bộ (tích cực cao) |
| `input_rd_addr` | ra | 12 | Địa chỉ đọc → input_sram ở wrapper |
| `input_dout` | vào | 8 | Dữ liệu đọc input (trễ 1 nhịp) |
| `start` | vào | 1 | Khởi động một lượt suy luận |
| `busy` | ra | 1 | Đang bận |
| `done` | ra | 1 | Xung hoàn tất |
| `result` | ra | 2 | Lớp dự đoán (0..3) |

---

## 3.3. Tích hợp giao tiếp và điều khiển hệ thống

Lõi CNN (`ecg_core`) được thiết kế **độc lập với bus**: nó chỉ phơi ra một giao diện gọn
gồm cổng ghi Input SRAM, tín hiệu `start`/`busy`/`done` và `result`. Để lõi này giao tiếp
được với máy tính chủ, cần một lớp bao (wrapper) làm bộ chuyển đổi bus, và một cầu nối vật
lý giữa máy tính và FPGA. Mục này trình bày hai lớp đó.

### 3.3.1. Tích hợp giao tiếp Avalon-MM Wrapper với lõi CNN

**Lớp bao mỏng.** Module đỉnh `ecg_accelerator_top` là một lớp bao mỏng, chỉ gồm hai thành
phần: bộ chuyển đổi bus `avalon_slave` và lõi `ecg_core` (kèm Input SRAM). Cách tách này
giúp lõi tính toán có thể tái dùng dưới nhiều loại bus khác nhau mà không sửa đổi, trong
khi `avalon_slave` đảm nhiệm toàn bộ việc diễn dịch giao thức Avalon-MM (đã trình bày ở
Chương giới thiệu) thành các tín hiệu điều khiển riêng của lõi:

```
   Host ──avs_*──► avalon_slave ──(8 dây)──► ecg_core
                   (bộ chuyển bus)           (datapath + FSM)
```

**Hình 3.4 — Cấu trúc lớp bao Avalon-MM.**

**Bản đồ thanh ghi.** `avalon_slave` phơi ra không gian địa chỉ 14-bit, chia thành các
vùng chức năng cho ở Bảng 3.18.

**Bảng 3.18 — Bản đồ địa chỉ Avalon-MM của lõi tăng tốc.**

| Vùng địa chỉ | Loại | Chức năng |
|--------------|:----:|-----------|
| 0x0000 | Ghi | `sram_din` [7:0] — dữ liệu byte Input SRAM |
| 0x0001 | Ghi | `sram_wr_addr` [11:0] — địa chỉ ghi Input SRAM |
| 0x0002 | Ghi | `sram_we` [0] — cho phép ghi |
| 0x0003 | Ghi | `start` [0] — phát lệnh chạy (xóa cờ done) |
| 0x0004 | Đọc | `status` = {done, busy} |
| 0x0005 | Đọc | `result` [1:0] — lớp dự đoán |
| 0x1000–0x19C3 | Ghi | Cửa sổ dữ liệu ECG (burst): mỗi từ = một byte Input SRAM |

Đường thanh ghi mức thấp (0x0000–0x0005) dùng cho mô phỏng kiểm chứng (testbench). Cửa sổ
dữ liệu ECG (0x1000–0x19C3, tức 2500 địa chỉ) dùng cho máy tính chủ nạp nhanh 2500 mẫu, mỗi
từ ghi tương ứng một byte Input SRAM tại địa chỉ `addr − 0x1000`.

Vì trọng số được bake sẵn vào bitstream (Mục 3.2.2), bus Avalon-MM chỉ đảm nhiệm nạp dữ
liệu ECG đầu vào, phát lệnh chạy và đọc kết quả — không có đường ghi trọng số. Nhờ vậy bản
đồ thanh ghi gọn, và toàn bộ hành vi của lõi là cố định theo topology và bộ trọng số
Chapman đã nạp lúc tổng hợp.

### 3.3.2. Tích hợp hệ thống với IP JTAG-to-Avalon giao tiếp thông qua PC

Kịch bản triển khai trên board sử dụng **cầu nối JTAG-to-Avalon** thay cho bộ xử lý cứng
HPS. Lý do là bản Quartus Prime Lite dùng cho khóa luận không cung cấp IP HPS cho Cyclone V;
do đó, thay vì để một CPU trên chip làm chủ bus, ta dùng một lõi IP cầu nối do Intel cung
cấp, biến cổng JTAG thành một **master Avalon-MM** điều khiển được từ máy tính.

JTAG (Joint Test Action Group, chuẩn IEEE 1149.1) là cổng gỡ lỗi chuẩn của FPGA, kết nối
tới máy tính qua cáp USB-Blaster. Trên board, JTAG dùng một nhóm chân riêng (TAP — Test
Access Port) tách khỏi các chân vào/ra thông thường, và mặc định phục vụ nạp bitstream cùng
gỡ lỗi mạch. IP **JTAG-to-Avalon master** của Intel tận dụng chính đường JTAG này để chuyển
lệnh đọc/ghi từ máy tính thành các giao dịch Avalon-MM bên trong FPGA, biến cổng nạp cấu hình
thành một kênh điều khiển thời gian chạy mà không tiêu tốn thêm chân board nào.

**Kiến trúc tích hợp trên board.** Hệ thống trên board được ghép bằng công cụ **Platform
Designer** (Qsys) của Quartus — hệ thống Qsys tên `jtag_system` gồm ba IP: (i) **JTAG-to-Avalon
master** đóng vai master bus; (ii) các **cầu clock/reset** đưa xung 100 MHz và tín hiệu reset
từ ngoài vào; (iii) lõi **`ecg_core`** làm slave, có giao diện Avalon-MM nối trực tiếp tới
master JTAG ngay bên trong Qsys (do đó slave không lộ ra ngoài, chỉ còn các conduit clock/reset).
Module bọc `jtag_top` instantiate hệ thống Qsys này và bổ sung một **PLL** nhân xung 50 MHz →
100 MHz cùng mạch đồng bộ reset; ngoài `FPGA_CLK1_50` và nút nhấn reset, board không cần chân
nào khác vì JTAG dùng TAP riêng:

```
   PC ──USB-Blaster──► [JTAG-to-Avalon master IP] ──Avalon-MM──► ecg_core
   (System Console)     └──────── hệ thống Qsys `jtag_system` ────────┘ (avalon_slave + core)
                            PLL 50→100 MHz ──clk──► toàn hệ thống
```

**Hình 3.5 — Kiến trúc tích hợp trên board với cầu nối JTAG-to-Avalon (ghép bằng Platform Designer).**

Trên máy tính, công cụ **System Console** (đi kèm Quartus) chạy một kịch bản driver (Tcl)
đóng vai máy tính chủ, thực hiện tuần tự:

1. Nạp bitstream vào FPGA qua cáp USB-Blaster.
2. Ghi 2500 mẫu ECG của một bản ghi vào Input SRAM qua cửa sổ dữ liệu (Bảng 3.18).
3. Phát `start`, thăm dò `status` tới khi `done = 1`, đọc `result[1:0]`.
4. Lặp lại cho toàn bộ tập kiểm tra, đối chiếu với nhãn để tính độ chính xác.

Đây là con đường thực nghiệm trên phần cứng thật được sử dụng ở Chương 4: cùng một bitstream
(với trọng số Chapman bake sẵn) chạy toàn bộ tập kiểm tra Chapman và đối chiếu với dự đoán
của mô hình phần mềm.

**Các biến thể giao tiếp thay thế.** Ngoài JTAG-to-Avalon (đã chạy trên board thật), thiết
kế còn có hai biến thể tích hợp khác nhằm dự phòng và mở rộng: (i) biến thể dùng **soft-core
RISC-V Nios V/m** làm chủ bus với chương trình bare-metal chạy trên RAM trên chip (thay cho
System Console trên PC); (ii) biến thể **UART** cho phép máy tính giao tiếp trực tiếp qua
cổng nối tiếp mà không cần JTAG. Kết quả chạy trên board của các biến thể này được báo cáo
ở Chương 4.

Đến đây, hệ thống đã hoàn chỉnh từ mô hình phần mềm, qua lõi phần cứng, tới lớp giao tiếp
với máy tính. Chương tiếp theo trình bày các kết quả định lượng: độ chính xác phần mềm và
khảo sát lượng tử hóa, kết quả mô phỏng khớp-bit, kết quả chạy trên board FPGA, và đánh
giá tài nguyên, hiệu năng, năng lượng.
