# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1. Tín hiệu ECG và rối loạn nhịp tim

### 2.1.1. Sóng P-QRS-T và ý nghĩa lâm sàng

Điện tâm đồ (electrocardiogram — ECG) là phép đo hiệu điện thế theo thời gian, sinh ra
từ hoạt động khử cực (depolarization) và tái cực (repolarization) của các tế bào cơ
tim trong một chu kỳ tim. Xung động điện bắt nguồn tại nút xoang (sinoatrial node — nút
SA), nằm ở tâm nhĩ phải, có vai trò là "máy tạo nhịp" tự nhiên của tim. Từ nút SA, xung
động lan truyền qua hai tâm nhĩ, tới nút nhĩ-thất (atrioventricular node — nút AV), sau
đó theo bó His và mạng Purkinje lan tỏa đồng thời tới hai tâm thất. Mỗi giai đoạn lan
truyền này tạo ra một thành phần sóng đặc trưng có thể quan sát trên bề mặt cơ thể bằng
điện cực, tạo thành dạng sóng ECG gồm ba thành phần chính:

- **Sóng P** — tương ứng với quá trình khử cực tâm nhĩ. Biên độ nhỏ, thời gian ngắn
  (khoảng 80–100 ms), hình dạng thường tròn và đối xứng ở nhịp bình thường.
- **Phức bộ QRS** — tương ứng với quá trình khử cực tâm thất, có biên độ lớn nhất trong
  chu kỳ do khối lượng cơ thất lớn hơn nhiều so với cơ nhĩ. Thời gian bình thường dưới
  120 ms; QRS giãn rộng thường là dấu hiệu dẫn truyền bất thường trong thất.
- **Sóng T** — tương ứng với quá trình tái cực tâm thất, biên độ trung bình, thời gian
  dài hơn QRS.

Giữa các sóng là các khoảng thời gian (interval) mang ý nghĩa chẩn đoán riêng: khoảng
PR (từ đầu sóng P đến đầu QRS) phản ánh thời gian dẫn truyền qua nút AV; khoảng QT phản
ánh tổng thời gian khử cực và tái cực thất. Khoảng cách giữa hai đỉnh R liên tiếp
(khoảng RR) là đại lượng quan trọng nhất đối với bài toán phân loại nhịp: tần số tim
(heart rate) được suy ra trực tiếp từ RR theo công thức

```
HR (bpm) = 60 / RR (giây)
```

còn độ đều/không đều của chuỗi khoảng RR liên tiếp là căn cứ chính để phân biệt nhịp có
tổ chức (organized rhythm) với nhịp rung/loạn nhịp không đều.

Về mặt kỹ thuật thu nhận, ECG chuẩn lâm sàng sử dụng hệ thống **12 chuyển đạo**
(12-lead), mỗi chuyển đạo quan sát hoạt động điện tim từ một hướng chiếu khác nhau lên
mặt phẳng cơ thể. Tuy nhiên, đối với bài toán phân loại nhịp (rhythm classification) —
khác với bài toán định vị vùng thiếu máu cơ tim (localization) vốn cần đủ 12 chuyển đạo
— một chuyển đạo chi II (Lead II) thường đã đủ thông tin, vì Lead II có hướng chiếu gần
song song với trục điện học của tim nên cho sóng P và phức bộ QRS rõ nét nhất trong các
chuyển đạo chi. Đây là cơ sở để khóa luận này, cũng như nhiều công trình phân loại nhịp
khác trong Mục 2.4, chỉ sử dụng tín hiệu đơn kênh (single-lead) làm đầu vào cho mô hình,
giúp giảm đáng kể khối lượng dữ liệu và độ phức tạp phần cứng so với xử lý đồng thời
12 chuyển đạo.

Về đặc tính tín hiệu, ECG bề mặt có biên độ nhỏ (cỡ mV), dải tần hữu ích tập trung trong
khoảng 0.5–40 Hz, và thường bị nhiễu bởi ba nguồn chính: trôi đường nền (baseline
wander, tần số thấp do hô hấp/chuyển động điện cực), nhiễu điện lưới (powerline
interference, 50/60 Hz) và nhiễu cơ (electromyographic noise, tần số cao do co cơ). Các
đặc tính này chi phối trực tiếp bước tiền xử lý dữ liệu (chuẩn hóa biên độ, lấy mẫu lại
tần số) được trình bày ở Chương 3.

### 2.1.2. Bốn nhóm nhịp AFIB / GSVT / SB / SR

Rối loạn nhịp tim (arrhythmia) là bất kỳ sự sai lệch nào so với nhịp xoang bình thường
về nguồn gốc phát nhịp, tần số hoặc tính đều đặn của chu kỳ tim. Trên thực tế lâm sàng
có hàng chục dạng rối loạn nhịp được phân loại chi tiết theo mã bệnh (ví dụ theo chuẩn
SNOMED-CT), tuy nhiên với mục tiêu xây dựng một hệ phân loại gọn nhẹ triển khai được
trên thiết bị biên, khóa luận gộp các mã nhịp chi tiết của tập dữ liệu Chapman [1] về
bốn **nhãn gộp** (superclass) có ý nghĩa lâm sàng riêng biệt và cân bằng về mặt phân bố
dữ liệu: **AFIB**, **GSVT**, **SB** và **SR**. Đặc điểm phân biệt của bốn nhóm trên
sóng ECG được tóm tắt ở Bảng 2.1.

**Bảng 2.1** — Đặc điểm phân biệt bốn nhóm nhịp trên tín hiệu ECG

| Nhãn | Tên đầy đủ | Tần số tim (bpm) | Sóng P | Khoảng RR | Phức bộ QRS |
|---|---|---|---|---|---|
| **SR** | Sinus Rhythm (nhịp xoang bình thường) | 60–100 | Đều, đứng trước mỗi QRS | Đều | Hẹp (<120 ms) |
| **SB** | Sinus Bradycardia (nhịp chậm xoang) | <60 | Đều, đứng trước mỗi QRS | Đều (dài hơn SR) | Hẹp |
| **GSVT** | Grouped Supraventricular Tachycardia (nhóm nhịp nhanh trên thất) | >100 | Có thể biến dạng hoặc lẫn vào sóng T trước đó | Thường đều | Hẹp (nguồn gốc trên thất) |
| **AFIB** | Atrial Fibrillation (rung nhĩ) | Thay đổi, thường 100–175 | Không quan sát được, thay bằng dao động rung nhĩ (f-wave) biên độ nhỏ, tần số cao | **Không đều một cách bất thường** (irregularly irregular) | Hẹp (trừ khi có block dẫn truyền) |

Ba đặc điểm then chốt mà một bộ phân loại — dù là bác sĩ hay mô hình học máy — cần khai
thác để phân biệt bốn nhóm trên gồm: (i) **sự hiện diện và hình dạng của sóng P** (phân
biệt AFIB, nơi sóng P biến mất, với ba nhóm còn lại); (ii) **tần số tim** suy từ khoảng
RR trung bình (phân biệt SB tần số thấp với GSVT tần số cao, cả hai đối lập với SR ở
dải giữa); và (iii) **độ đều của chuỗi khoảng RR** (đặc trưng "không đều một cách bất
thường" gần như là dấu hiệu riêng của AFIB). Đây chính là các đặc trưng hình thái và
nhịp điệu mà một mạng nơ-ron tích chập cần học được từ dữ liệu thô, thay vì phải trích
xuất thủ công — nội dung được trình bày ở Mục 2.2.

Cần lưu ý rằng nhóm GSVT trong cách gộp nhãn của Chapman là một **nhóm hỗn hợp**, bao
gồm nhiều dạng nhịp nhanh có nguồn gốc trên thất (sinus tachycardia, supraventricular
tachycardia, atrial tachycardia, cuồng nhĩ...) có chung đặc điểm QRS hẹp và tần số cao
nhưng cơ chế điện sinh lý khác nhau. Việc gộp nhóm này giúp cân bằng số lượng mẫu giữa
bốn lớp, nhưng cũng đồng thời làm tăng phương sai nội-lớp (intra-class variance), là
một yếu tố cần cân nhắc khi phân tích kết quả phân loại và khả năng tổng quát hóa sang
tập dữ liệu khác ở Chương 3.

### 2.1.3. Hai tập dữ liệu sử dụng trong khóa luận

Khóa luận sử dụng hai tập dữ liệu ECG 12 chuyển đạo công khai, đóng hai vai trò khác
nhau: một tập dùng để huấn luyện và đánh giá chính, một tập độc lập dùng để kiểm chứng
khả năng tổng quát hóa sang phân bố dữ liệu khác (cross-dataset).

**Tập huấn luyện — cơ sở dữ liệu điện tâm đồ 12 chuyển đạo quy mô lớn phục vụ nghiên
cứu rối loạn nhịp.** Tên đầy đủ theo PhysioNet là *A large scale 12-lead
electrocardiogram database for arrhythmia study* (phiên bản 1.0.0) [1]. Tập này là kết
quả gộp dữ liệu từ hai nguồn: Đại học Chapman phối hợp Bệnh viện Nhân dân Thiệu Hưng
(Chapman University & Shaoxing People's Hospital) và Bệnh viện số Một Ninh Ba (Ningbo
First Hospital), tổng cộng 45.152 bản ghi ECG 12 chuyển đạo, tần số lấy mẫu 500 Hz, độ
dài 10 giây mỗi bản ghi, kèm nhãn chẩn đoán nhịp theo mã SNOMED-CT do bác sĩ xác nhận.
Công bố gốc mô tả tập dữ liệu là bài báo của Zheng và cộng sự trên *Scientific Data*
[1]. **Để ngắn gọn, từ đây trở đi khóa luận gọi tập dữ liệu này là "Chapman"** — cách
gọi tắt cũng được dùng phổ biến trong các công trình liên quan ở Mục 2.4.

**Tập kiểm chứng chéo — Georgia 12-Lead ECG Challenge Database.** Đây là một trong các
nguồn dữ liệu thành phần của cuộc thi *PhysioNet/Computing in Cardiology Challenge 2020*
[43], do Đại học Emory (Atlanta, bang Georgia, Hoa Kỳ) đóng góp, gồm 10.344 bản ghi ECG
12 chuyển đạo, cũng ở tần số 500 Hz và độ dài 10 giây. Vì được thu thập tại một quần thể
bệnh nhân, một hệ thống thiết bị và một vùng địa lý hoàn toàn khác so với tập huấn
luyện, Georgia đóng vai trò một phép thử **chuyển giao xa** (far-transfer): mô hình
không được huấn luyện trên bất kỳ mẫu nào của tập này, nên độ chính xác đo được phản ánh
trực tiếp khả năng tổng quát hóa chứ không phải khả năng ghi nhớ. Khóa luận gọi tắt tập
này là **"Georgia"**.

Điểm chung thuận lợi của hai tập là cùng định dạng WFDB, cùng tần số lấy mẫu 500 Hz và
cùng độ dài bản ghi 10 giây, nên có thể dùng chung một luồng tiền xử lý mà không cần
bước lấy mẫu lại (resampling) — chi tiết quy trình tiền xử lý, ánh xạ mã SNOMED-CT về
bốn nhãn ở Bảng 2.1 và cách chia tập được trình bày ở Chương 3.

---

## 2.2. Mạng nơ-ron tích chập một chiều cho ECG

### 2.2.1. Tích chập 1D, MaxPool, GAP, FC, argmax

Vì tín hiệu ECG là một chuỗi giá trị vô hướng theo thời gian, mạng nơ-ron tích chập một
chiều (1D-CNN) là kiến trúc tự nhiên để trích xuất đặc trưng cục bộ trực tiếp từ dữ liệu
thô, mà không cần bước biến đổi sang miền ảnh (ví dụ biến đổi thời gian-tần số) như cách
tiếp cận 2D-CNN. Công trình nền tảng của Kiranyaz và cộng sự [3] là một trong những
minh chứng đầu tiên cho khả năng phân loại ECG thời gian thực trực tiếp trên tín hiệu
thô bằng 1D-CNN. Phần dưới đây trình bày lần lượt các phép toán cấu thành một mạng
1D-CNN dùng trong khóa luận.

**Tích chập một chiều (1D convolution).** Với tín hiệu vào một kênh `x[n]` và bộ lọc
(kernel/filter) `w[k]` có độ dài `K`, phép tích chập rời rạc tại vị trí `n` được định
nghĩa:

```
y[n] = Σ (k=0..K-1) x[n + k - P] · w[k]  +  b
```

trong đó `P` là số điểm đệm (padding) hai bên để giữ nguyên độ dài chuỗi đầu ra (padding
kiểu "same"), và `b` là hệ số thiên lệch (bias). Khi tín hiệu vào có nhiều kênh
(`in_ch` kênh), mỗi kênh đầu ra `oc` là tổng tích chập trên **tất cả** kênh vào, cộng
dồn theo kênh:

```
y_oc[n] = Σ (ic=0..in_ch-1) Σ (k=0..K-1) x_ic[n + k - P] · w_{oc,ic}[k]  +  b_oc
```

Số phép nhân-cộng (MAC) để tính một điểm ra là `in_ch × K`, và tổng số MAC cho một lớp
tích chập tỉ lệ với `in_ch × out_ch × K × out_len` — đây chính là đại lượng chi phối
trực tiếp số lượng khối nhân (multiplier) cần thiết khi ánh xạ lên phần cứng, sẽ được
phân tích ở Chương 4. Trường tiếp nhận (receptive field) — phạm vi các mẫu đầu vào ảnh
hưởng tới một điểm ra — tăng dần qua mỗi lớp tích chập và mỗi lớp lấy mẫu con, cho phép
các lớp sâu hơn "nhìn thấy" ngữ cảnh thời gian rộng hơn (ví dụ hình dạng cả một chu kỳ
tim) dù kernel mỗi lớp chỉ rộng vài mẫu.

**Lấy mẫu con cực đại (Max Pooling).** Sau mỗi lớp tích chập, một cửa sổ trượt kích
thước `S` (không chồng lấp, bước nhảy — stride — bằng `S`) được áp dụng để lấy giá trị
lớn nhất trong cửa sổ, làm giảm độ dài chuỗi theo hệ số `S`:

```
p[m] = max( y[mS], y[mS+1], ..., y[mS+S-1] )
```

MaxPool có hai vai trò: (i) giảm số điểm dữ liệu cần xử lý ở các lớp sau, giúp giảm chi
phí tính toán và bộ nhớ theo cấp số nhân qua các lớp; và (ii) giữ lại đặc trưng có phản
hồi mạnh nhất trong mỗi cửa sổ thời gian — phù hợp với bản chất bài toán phát hiện đỉnh
sóng (ví dụ đỉnh phức bộ QRS) vốn quan trọng hơn giá trị trung bình cục bộ.

**Lấy trung bình toàn cục (Global Average Pooling — GAP).** Sau khi qua các lớp tích
chập và lấy mẫu con, thay vì "trải phẳng" (flatten) toàn bộ bản đồ đặc trưng thành một
vector dài rồi đưa vào lớp kết nối đầy đủ (cách làm truyền thống), khóa luận áp dụng GAP
[10] — lấy trung bình theo chiều thời gian cho từng kênh:

```
g_oc = (1 / L) · Σ (n=0..L-1) y_oc[n]
```

với `L` là độ dài còn lại của chuỗi ở kênh `oc` sau lớp tích chập cuối. GAP có ưu điểm
quan trọng đối với mô hình nhẹ triển khai phần cứng: (i) số tham số của lớp kết nối đầy
đủ theo sau chỉ phụ thuộc vào **số kênh** thay vì **số kênh × độ dài chuỗi**, giảm mạnh
số tham số và giảm nguy cơ quá khớp (overfitting); (ii) về phần cứng, GAP chỉ cần một bộ
cộng dồn (accumulator) và một phép chia, không cần bộ đệm lưu toàn bộ bản đồ đặc trưng
để "trải phẳng".

**Lớp kết nối đầy đủ (Fully Connected — FC).** Vector đặc trưng sau GAP (kích thước
bằng số kênh của lớp tích chập cuối) được ánh xạ tuyến tính sang không gian số lớp cần
phân loại (ở đây là 4 nhãn):

```
z_j = Σ (i) g_i · W_{j,i}  +  b_j ,     j = 0..3
```

Kết quả `z_j` — thường gọi là logit — là giá trị chưa chuẩn hóa, phản ánh mức độ "ủng
hộ" của mô hình đối với nhãn `j`.

**Hàm kích hoạt phi tuyến (ReLU) và Argmax.** Giữa các lớp tích chập, hàm kích hoạt
`ReLU(x) = max(0, x)` được chèn vào để đưa tính phi tuyến vào mạng, giúp mô hình học
được các quan hệ phức tạp hơn một chuỗi phép biến đổi tuyến tính. Cuối cùng, quyết định
phân loại được đưa ra bằng phép **argmax** trên vector logit:

```
class = argmax_j ( z_j )
```

nghĩa là chọn nhãn có giá trị logit lớn nhất, không cần chuẩn hóa qua hàm softmax khi
suy luận (inference) vì phép so sánh thứ tự giữa các logit không đổi qua softmax — đây
là một điểm tối ưu quan trọng cho phần cứng vì tránh được phép tính hàm mũ.

Toàn bộ chuỗi phép toán — tích chập, kích hoạt, lấy mẫu con, lặp lại qua nhiều lớp, rồi
GAP, FC, argmax — tạo thành một phép biến đổi tiến (forward pass) từ tín hiệu ECG thô
sang một trong bốn nhãn nhịp. Kiến trúc cụ thể (số lớp, số kênh mỗi lớp, kích thước
kernel) được lựa chọn cho bài toán của khóa luận sẽ được trình bày chi tiết ở Chương 3.

### 2.2.2. Huấn luyện: hàm mất mát, Adam, lan truyền ngược

**Hàm mất mát Cross-Entropy.** Bài toán phân loại 4 nhãn là một bài toán phân loại đa
lớp (multi-class classification). Để huấn luyện, vector logit `z` được chuẩn hóa qua
hàm softmax thành phân bố xác suất:

```
p_j = exp(z_j) / Σ_k exp(z_k)
```

và độ lệch giữa phân bố dự đoán `p` và nhãn thật `y` (dạng one-hot) được đo bằng hàm mất
mát entropy chéo (cross-entropy loss):

```
L = − Σ_j y_j · log(p_j)  =  − log(p_{j*})
```

với `j*` là chỉ số nhãn đúng. Hàm mất mát này càng nhỏ khi mô hình càng "tự tin" và
"chính xác" vào nhãn thật, và có đạo hàm dạng đóng thuận tiện cho lan truyền ngược.

**Lan truyền ngược (Backpropagation).** Việc huấn luyện mạng nơ-ron là bài toán tối ưu
tìm bộ tham số `θ` (trọng số và bias mọi lớp) sao cho tối thiểu hóa hàm mất mát trung
bình trên tập huấn luyện. Lan truyền ngược là thuật toán tính gradient `∂L/∂θ` cho mọi
tham số bằng quy tắc chuỗi (chain rule), lan truyền đạo hàm từ lớp ra ngược về lớp vào
qua từng phép toán (tích chập, ReLU, pooling, GAP, FC). Với gradient thu được, tham số
được cập nhật theo hướng ngược gradient để giảm hàm mất mát — nguyên lý hạ gradient
(gradient descent):

```
θ ← θ − η · ∂L/∂θ
```

với `η` là tốc độ học (learning rate).

**Adam optimizer.** Thay vì hạ gradient thuần túy với tốc độ học cố định, khóa luận sử
dụng thuật toán tối ưu **Adam** (Adaptive Moment Estimation) [7] — một trong những
thuật toán tối ưu phổ biến nhất cho huấn luyện mạng sâu nhờ khả năng hội tụ nhanh và ổn
định. Adam duy trì ước lượng động (moment) bậc một (trung bình động của gradient) và
bậc hai (trung bình động của bình phương gradient):

```
m_t = β1 · m_{t-1} + (1 − β1) · g_t
v_t = β2 · v_{t-1} + (1 − β2) · g_t²
m̂_t = m_t / (1 − β1^t)          (hiệu chỉnh độ lệch — bias correction)
v̂_t = v_t / (1 − β2^t)
θ_t = θ_{t-1} − η · m̂_t / (√v̂_t + ε)
```

trong đó `g_t` là gradient tại bước `t`, `β1, β2` là hệ số suy giảm (thường 0.9 và
0.999), `ε` là hằng số nhỏ tránh chia cho 0. Ước lượng moment bậc hai cho phép Adam tự
điều chỉnh tốc độ học riêng cho từng tham số — tham số có gradient biến động lớn được
cập nhật thận trọng hơn — giúp huấn luyện ổn định hơn so với hạ gradient tốc độ học cố
định, đặc biệt phù hợp với mô hình nhỏ và tập dữ liệu có phân bố lớp không hoàn toàn cân
bằng như bài toán ECG.

**Vấn đề quá khớp và các kỹ thuật điều chuẩn.** Khi số tham số mô hình đủ lớn so với
lượng dữ liệu huấn luyện, mô hình có nguy cơ quá khớp (overfitting) — học thuộc nhiễu và
đặc điểm riêng của tập huấn luyện thay vì quy luật tổng quát, dẫn đến hiệu năng kém trên
dữ liệu chưa từng thấy. Hai kỹ thuật điều chuẩn (regularization) phổ biến trong CNN là
**chuẩn hóa theo lô** (Batch Normalization — BN) [21], giúp ổn định phân bố đầu vào mỗi
lớp qua các bước huấn luyện, và **Dropout** [22], ngẫu nhiên loại bỏ một phần nơ-ron
trong huấn luyện để tránh phụ thuộc quá mức vào một tổ hợp đặc trưng cụ thể. Cả hai kỹ
thuật này đều được cân nhắc nhưng không được áp dụng trong kiến trúc cuối cùng của khóa
luận: GAP (Mục 2.2.1) đã đóng vai trò điều chuẩn tự nhiên bằng cách giảm mạnh số tham số
lớp FC, trong khi việc gộp cả BN lẫn Dropout sẽ làm phức tạp đáng kể luồng tính toán số
nguyên bit-exact cần thiết cho việc ánh xạ sang phần cứng ở Chương 3 và Chương 4. Đây là
một minh họa cho việc lựa chọn kỹ thuật huấn luyện trong khóa luận không chỉ dựa trên
tiêu chí độ chính xác phần mềm thuần túy, mà còn phải cân nhắc tính tương thích với mục
tiêu triển khai phần cứng.

### 2.2.3. Nén mô hình: tỉa kênh có cấu trúc và lượng tử hóa INT8/STE

Một mô hình CNN huấn luyện bằng số thực dấu phẩy động 32-bit (float32) thông thường có
số tham số và độ phức tạp tính toán vượt quá khả năng của một lõi phần cứng nhỏ gọn,
tiêu thụ ít năng lượng. Hai kỹ thuật nén mô hình bổ trợ nhau được áp dụng: **tỉa kênh có
cấu trúc** (structured channel pruning) làm giảm số kênh — do đó giảm số phép MAC và số
tham số — và **lượng tử hóa** (quantization) làm giảm số bit biểu diễn mỗi tham số và
phép tính.

**Tỉa kênh có cấu trúc (Structured Pruning).** Tỉa mô hình nói chung nhằm loại bỏ các
thành phần ít đóng góp vào hiệu năng dự đoán. Có hai hướng tiếp cận: tỉa **không cấu
trúc** (unstructured), loại bỏ từng trọng số riêng lẻ, tạo ra ma trận thưa (sparse)
nhưng khó tận dụng hiệu quả trên phần cứng số học dày đặc (dense arithmetic) như FPGA vì
vẫn cần logic định địa chỉ phức tạp cho dữ liệu thưa; và tỉa **có cấu trúc**
(structured), loại bỏ toàn bộ một kênh (channel) hoặc bộ lọc (filter), giữ nguyên cấu
trúc kết nối dày đặc của các kênh còn lại. Khóa luận lựa chọn hướng có cấu trúc vì kết
quả sau tỉa vẫn là một mạng dày đặc thông thường với số kênh nhỏ hơn — ánh xạ trực tiếp
sang phần cứng mà không cần cơ chế xử lý dữ liệu thưa chuyên biệt.

Tiêu chí đánh giá mức độ quan trọng của một kênh để quyết định tỉa hay giữ có thể dựa
trên độ lớn (magnitude) của trọng số — ví dụ chuẩn L1 (L1-norm) của bộ lọc tương ứng
[13]:

```
Importance(filter) = Σ |w_i|   (tổng trị tuyệt đối trọng số trong bộ lọc)
```

hoặc dựa trên ảnh hưởng của kênh đó tới hàm mất mát, ước lượng bằng khai triển Taylor
bậc một (first-order Taylor expansion) [12]:

```
Importance(channel) ≈ | ∂L/∀output · output |   (tích gradient và giá trị kích hoạt)
```

Tiêu chí Taylor tận dụng thông tin gradient sẵn có từ quá trình lan truyền ngược
(Mục 2.2.2), phản ánh trực tiếp mức độ ảnh hưởng của kênh tới hàm mất mát thay vì chỉ
dựa trên độ lớn trọng số, và thường cho kết quả tốt hơn ở mô hình đã huấn luyện hội tụ.
Sau khi tỉa bỏ các kênh ít quan trọng nhất theo một tỉ lệ mục tiêu, mô hình được huấn
luyện lại (fine-tune) để phục hồi phần độ chính xác bị mất do loại bỏ tham số — quy
trình lặp "tỉa rồi tinh chỉnh" là cách tiếp cận tiêu chuẩn trong nén mô hình [11].

**Lượng tử hóa (Quantization).** Song song với việc giảm số kênh, lượng tử hóa giảm số
bit biểu diễn mỗi giá trị — từ float32 (32 bit) xuống số nguyên có dấu 8-bit (INT8) —
mang lại ba lợi ích trực tiếp cho phần cứng: (i) giảm 4 lần dung lượng bộ nhớ lưu trữ
trọng số; (ii) phép nhân số nguyên có chi phí phần cứng (diện tích, độ trễ, năng lượng)
thấp hơn đáng kể so với phép nhân dấu phẩy động cùng độ rộng bit; (iii) toàn bộ luồng
tính toán có thể triển khai bằng số học số nguyên xác định (deterministic), không cần
khối xử lý dấu phẩy động (floating-point unit) tốn tài nguyên. Có hai hướng lượng tử
hóa chính:

- **Lượng tử hóa sau huấn luyện** (Post-Training Quantization — PTQ): mô hình được
  huấn luyện bình thường bằng float32, sau đó các trọng số và giá trị kích hoạt được
  ánh xạ sang số nguyên dựa trên thống kê biên độ quan sát được (hiệu chỉnh — 
  calibration), không cần huấn luyện lại.
- **Huấn luyện nhận biết lượng tử** (Quantization-Aware Training — QAT) [14]: quá trình
  lượng tử hóa (làm tròn, giới hạn biên độ) được **mô phỏng ngay trong vòng lặp huấn
  luyện** (gọi là fake-quantize — lượng tử hóa giả lập), để mô hình "học thích nghi"
  với nhiễu lượng tử hóa, thường cho độ chính xác cao hơn PTQ ở cùng độ rộng bit.

**Ước lượng thẳng qua (Straight-Through Estimator — STE).** Khó khăn kỹ thuật của QAT
nằm ở chỗ phép làm tròn (round) — thành phần bắt buộc của lượng tử hóa — có đạo hàm bằng
0 hầu như khắp nơi, khiến gradient không thể lan truyền ngược qua phép làm tròn theo
cách thông thường. STE [15] giải quyết vấn đề này bằng cách: ở lượt tính tiến (forward
pass), áp dụng đúng phép làm tròn; còn ở lượt lan truyền ngược (backward pass), **coi
đạo hàm của phép làm tròn xấp xỉ bằng 1** — tức là "đi thẳng qua" (straight-through) như
thể không có phép làm tròn nào xảy ra:

```
forward:   x_q = round(x)
backward:  ∂x_q/∂x  ≈  1     (thay vì đạo hàm thật, gần như luôn bằng 0)
```

Xấp xỉ này tuy không chính xác về mặt giải tích nhưng hoạt động hiệu quả trong thực tế
vì trên một khoảng đủ nhỏ, phép làm tròn gần như là một phép "cộng nhiễu" không thiên
lệch (unbiased), không cản trở hướng tối ưu tổng thể của gradient.

**Lượng tử hóa theo lũy thừa của 2 (Power-of-2 Quantization).** Trong các phương pháp
lượng tử hóa, có một nhánh riêng chọn hệ số co giãn (scale factor) là lũy thừa của 2
thay vì một số thực bất kỳ [30], [17] — được gọi là lượng tử hóa power-of-2 hay lượng
tử hóa logarit (logarithmic quantization). Lợi ích cốt lõi của cách chọn này là phép
**co giãn giá trị** (rescale) — vốn cần thực hiện sau mỗi phép tích lũy để đưa kết quả
về đúng thang đo INT8 — chỉ cần một phép **dịch bit** (bit-shift) thay vì một phép
**nhân** với hệ số thực. Vì phép dịch bit trên FPGA được tổng hợp thành dây nối
(wiring) hoặc thanh ghi dịch đơn giản — không tốn khối nhân DSP — trong khi lượng tử
hóa tổng quát (general-scale, theo hệ số thực bất kỳ) [14] cần một phép nhân số nguyên
cho mỗi lần co giãn, sự khác biệt này có ý nghĩa trực tiếp tới số lượng khối DSP cần
dùng trên FPGA — nội dung sẽ được phân tích định lượng ở Mục 2.4.2 và Chương 3.

Kết hợp cả hai kỹ thuật — tỉa kênh có cấu trúc và lượng tử hóa QAT power-of-2 với STE —
tạo thành đường ống nén mô hình hoàn chỉnh, biến đổi mô hình float32 ban đầu thành một
mô hình nhỏ gọn với trọng số INT8, sẵn sàng để ánh xạ sang một lõi phần cứng chuyên
dụng — chủ đề của Mục 2.3 và Chương 4.

---

## 2.3. Nền tảng thiết kế số trên FPGA

### 2.3.1. Kiến trúc Cyclone V: ALM, DSP, M10K

FPGA (Field-Programmable Gate Array) là một loại vi mạch logic khả trình, cho phép cấu
hình lại cấu trúc mạch số sau khi sản xuất, khác với ASIC (Application-Specific
Integrated Circuit) vốn có cấu trúc mạch cố định ngay từ khâu chế tạo. Đặc tính khả
trình lại cho phép FPGA triển khai các mạch số chuyên biệt — như một lõi tăng tốc CNN —
với mức độ song song hóa và pipeline tùy biến theo đúng đặc điểm của bài toán, đồng thời
vẫn giữ được chi phí phát triển và thời gian đưa ra thị trường thấp hơn nhiều so với
thiết kế ASIC.

Về cấu trúc bên trong, một FPGA hiện đại gồm ba loại tài nguyên chính: khối logic khả
trình (chứa bảng tra — Look-Up Table, LUT — và thanh ghi), khối nhân chuyên dụng
(dùng cho các phép toán số học tốc độ cao), và khối nhớ nhúng (dùng làm bộ đệm trên
chip). Trên dòng **Intel Cyclone V**, ba loại tài nguyên này có tên gọi và đặc điểm cụ
thể như sau [31]:

- **ALM (Adaptive Logic Module)** — đơn vị logic cơ bản của Cyclone V, gồm một cấu trúc
  LUT thích ứng (adaptive LUT) có thể chia thành nhiều LUT nhỏ hơn để tổng hợp nhiều
  hàm logic khác nhau, cùng với các thanh ghi (flip-flop) để lưu trạng thái. Toàn bộ
  logic tổ hợp (mạch điều khiển, bộ cộng, bộ so sánh, MUX) và các máy trạng thái hữu hạn
  (Finite State Machine — FSM) của thiết kế được tổng hợp thành ALM.
- **DSP18 (khối DSP độ chính xác thay đổi — variable-precision DSP block)** — khối
  cứng chuyên dụng cho phép nhân và cộng dồn tốc độ cao, có thể cấu hình làm một phép
  nhân 18×18-bit hoặc tách thành nhiều phép nhân độ rộng nhỏ hơn (ví dụ hai phép nhân
  9×9-bit). Đây là tài nguyên trực tiếp thực hiện các phép nhân trong tích chập
  (Mục 2.2.1) — với dữ liệu INT8, mỗi khối DSP18 có thể đảm nhận một hoặc nhiều phép
  nhân MAC tùy cách ánh xạ.
- **M10K (khối nhớ nhúng 10-kbit)** — khối RAM hai cổng (dual-port) dung lượng 10.240
  bit mỗi khối, có thể cấu hình linh hoạt về độ rộng dữ liệu và độ sâu địa chỉ. Trên
  Cyclone V, M10K được dùng làm bộ đệm trên chip (on-chip buffer) cho dữ liệu trung
  gian giữa các lớp mạng và cho bộ nhớ trọng số — thay thế nhu cầu truy cập bộ nhớ
  ngoài (DDR) tốc độ chậm và tốn năng lượng hơn nhiều so với truy cập BRAM nội bộ.

Bo mạch DE10-Standard sử dụng trong khóa luận có FPGA Cyclone V mã hiệu
`5CSXFC6D6F31C6` với 41.910 ALM, 112 DSP18 và 397 khối M10K (chi tiết vai trò từng loại
tài nguyên trong thiết kế cụ thể của khóa luận đã được giới thiệu ở Bảng công cụ/board
mạch Chương 1, và sẽ được phân tích định lượng khi trình bày kết quả tổng hợp ở
Chương 5). Ba loại tài nguyên trên tạo thành "ngân sách phần cứng" mà toàn bộ thiết kế
lõi CNN accelerator ở Chương 4 phải được cân đối bên trong.

### 2.3.2. Giao diện Avalon-MM và pipeline

**Avalon Memory-Mapped (Avalon-MM).** Để một lõi tính toán chuyên dụng (như lõi CNN của
khóa luận) có thể giao tiếp với thế giới bên ngoài — nhận lệnh điều khiển, nạp dữ liệu
đầu vào, đọc kết quả — cần một giao thức bus chuẩn hóa. **Avalon-MM** là chuẩn giao tiếp
bus dạng ánh xạ bộ nhớ (memory-mapped) do Intel/Altera định nghĩa, dùng phổ biến trong
hệ thống thiết kế bằng công cụ Platform Designer (trước đây gọi là Qsys). Một giao dịch
Avalon-MM cơ bản gồm vai trò **chủ (master)** — bên khởi tạo giao dịch đọc/ghi, ví dụ
một bộ điều khiển JTAG-to-Avalon hoặc lõi xử lý HPS — và vai trò **tớ (slave)** — bên
tiếp nhận và đáp ứng giao dịch, ở đây là lõi CNN accelerator. Các tín hiệu chính của một
cổng Avalon-MM tớ gồm: địa chỉ (`address`), dữ liệu ghi/đọc (`writedata`/`readdata`),
tín hiệu cho phép đọc/ghi (`read`/`write`), và tín hiệu yêu cầu chờ (`waitrequest`) để
tớ có thể tạm dừng chủ khi chưa sẵn sàng phục vụ giao dịch. Bằng cách ánh xạ các thanh
ghi điều khiển, vùng nhớ trọng số và vùng nhớ dữ liệu vào một không gian địa chỉ thống
nhất, phía chủ (PC, thông qua JTAG-to-Avalon) có thể nạp trọng số, nạp tín hiệu ECG, ra
lệnh bắt đầu suy luận và đọc kết quả phân loại chỉ bằng các giao dịch đọc/ghi thông
thường — cơ chế cụ thể được trình bày ở Chương 4.

**Pipeline trong thiết kế số.** Một trong những kỹ thuật cơ bản nhất để tăng tần số hoạt
động (clock frequency) của một mạch số là **pipeline hóa** (pipelining): chia một phép
tính tổ hợp phức tạp, có đường trễ (propagation delay) dài, thành nhiều tầng nhỏ hơn,
ngăn cách nhau bởi thanh ghi (register). Mỗi tầng chỉ cần thực hiện một phần nhỏ của
phép tính trong một chu kỳ xung nhịp (clock cycle), giúp giảm đường trễ dài nhất giữa
hai thanh ghi liên tiếp — gọi là **đường tới hạn** (critical path). Tần số tối đa mà
mạch có thể hoạt động ổn định (Fmax) bị giới hạn trực tiếp bởi độ trễ của đường tới hạn:

```
Fmax ≈ 1 / (T_critical_path + T_setup)
```

Đánh đổi của pipeline là **độ trễ** (latency, tính bằng số chu kỳ để một dữ liệu đi hết
toàn bộ pipeline) tăng lên, nhưng **thông lượng** (throughput, số kết quả hoàn thành mỗi
chu kỳ ở chế độ ổn định) không đổi hoặc tăng, vì các tầng có thể xử lý dữ liệu của nhiều
"lượt tính" khác nhau đồng thời, tương tự dây chuyền sản xuất công nghiệp. Đây là
nguyên lý cốt lõi để một lõi CNN đạt được tần số hoạt động cao dù mỗi phép tích chập tự
thân gồm nhiều tầng cộng dồn (adder tree) nối tiếp — kỹ thuật phân tầng pipeline cụ thể
cho khối tích chập của khóa luận được trình bày ở Chương 4. Việc phân tích timing —
xác định đường tới hạn, kiểm tra ràng buộc thời gian thiết lập/giữ (setup/hold) tại mỗi
thanh ghi — được thực hiện tự động bởi công cụ phân tích timing tĩnh (static timing
analysis) tích hợp trong bộ công cụ tổng hợp, và là cơ sở cho các số liệu Fmax báo cáo ở
Chương 5. Các nguyên lý số học số nguyên có dấu, biểu diễn số cố định (fixed-point) và
số học bão hòa (saturating arithmetic — dùng cho phép giới hạn biên độ/clamp sau lượng
tử hóa) áp dụng trong thiết kế cũng dựa trên nền tảng kiến trúc máy tính và thiết kế số
tiêu chuẩn [38], [30].

### 2.3.3. Ánh xạ luồng dữ liệu: fully-mapped vs time-multiplexed

Khi triển khai một mạng CNN nhiều lớp lên FPGA, có nhiều cách tổ chức phần cứng khác
nhau để ánh xạ cùng một tập phép toán (tích chập, pooling...) — gọi chung là lựa chọn
**luồng dữ liệu** (dataflow) của bộ tăng tốc. Sze và cộng sự [26] đưa ra một cách phân
loại kinh điển dựa trên đại lượng nào được "giữ cố định" tại chỗ (stationary) trong bộ
xử lý để tối thiểu hóa việc di chuyển dữ liệu — trọng số cố định (weight-stationary),
đầu ra cố định (output-stationary) hay đầu vào cố định (input-stationary). Ở mức kiến
trúc hệ thống cao hơn, hai hướng tổ chức phổ biến cho một bộ tăng tốc CNN nhiều lớp là:

- **Kiến trúc ánh xạ đầy đủ / streaming theo lớp** (fully-mapped / per-layer streaming
  dataflow): mỗi lớp mạng được tổng hợp thành một khối phần cứng vật lý riêng, các khối
  nối tiếp nhau tạo thành một đường ống xuyên suốt toàn mạng. Dữ liệu "chảy" qua lần
  lượt các khối phần cứng của từng lớp, cho phép các lớp xử lý đồng thời (tương tự
  pipeline ở cấp độ lớp), nhưng tổng tài nguyên phần cứng cần dùng tỉ lệ thuận với
  **tổng độ phức tạp của tất cả các lớp cộng lại**. Đây là hướng tiếp cận của công trình
  đối chứng gần nhất — Liu và cộng sự [18] — và của các khung công cụ tự động hóa ánh xạ
  CNN [28].
- **Kiến trúc một bộ tính toán dùng chung / dùng lại theo thời gian**
  (single-computation-engine / time-multiplexed dataflow): chỉ một khối phần cứng tính
  toán (tích chập/pooling) được tổng hợp, và khối này được **dùng lại tuần tự** cho tất
  cả các lớp mạng thông qua một bộ điều khiển máy trạng thái hữu hạn (FSM). Tài nguyên
  phần cứng chỉ phụ thuộc vào **lớp phức tạp nhất**, không phải tổng các lớp, đổi lại
  tổng thời gian suy luận bằng tổng thời gian xử lý tuần tự từng lớp.

Bài khảo sát toolflow của Venieris và cộng sự [28] hệ thống hóa hai hướng trên thành hai
nhóm kiến trúc đối lập trong không gian thiết kế bộ tăng tốc CNN trên FPGA, là khung lập
luận được khóa luận sử dụng khi lựa chọn kiến trúc dùng lại theo thời gian (time-
multiplexed single-engine) cho lõi CNN đề xuất — do quy mô mạng nhỏ (bốn lớp tích chập,
tổng cộng 640 tham số sau tỉa) khiến việc cấp phát phần cứng riêng cho mỗi lớp theo
hướng fully-mapped trở nên lãng phí (các lớp có độ phức tạp rất chênh lệch nhau).

Bên trong một khối tính toán tích chập theo hướng dòng chảy (streaming), một kỹ thuật tổ
chức bộ đệm phổ biến là **cửa sổ trượt dùng thanh ghi dịch** (shift-register window —
SRW) kết hợp **bộ đệm hai vùng** (ping-pong / double buffering) giữa các lớp [27]: dữ
liệu đầu vào được nạp tuần tự vào một chuỗi thanh ghi dịch có độ dài bằng kích thước
kernel, mỗi chu kỳ dịch chuyển một mẫu mới vào và loại mẫu cũ nhất ra, cho phép khối
tích chập luôn có sẵn đúng `K` mẫu liền kề cần thiết để tính một điểm ra mà không cần
truy cập ngẫu nhiên vào bộ nhớ; còn bộ đệm ping-pong cho phép lớp sau bắt đầu đọc dữ liệu
từ một vùng nhớ trong khi lớp trước vẫn đang ghi vào vùng nhớ còn lại, tránh xung đột
truy cập. Đây là kỹ thuật tổ chức bộ nhớ được áp dụng cho khối tích chập của khóa luận,
trình bày chi tiết ở Chương 4.

Về hướng **song song hóa** (parallelism) bên trong khối tính toán, có thể chọn song song
hóa theo **kênh** (channel-parallel) — mỗi bộ xử lý phần tử (Processing Element — PE) xử
lý độc lập một kênh đầu ra, phù hợp khi số kênh nhỏ và cố định qua các lớp — hoặc song
song hóa theo **vị trí thời gian** (position-parallel / SIMD theo chiều dài chuỗi) —
nhiều PE cùng tính các điểm ra liền kề nhau trên cùng một kênh trong cùng một chu kỳ,
phù hợp khi cần rút ngắn thời gian xử lý một chuỗi dài. Hai hướng song song hóa này đánh
đổi khác nhau giữa số lượng PE, độ phức tạp bộ điều khiển và độ trễ suy luận. Khóa luận
chọn hướng **song song theo kênh với 8 PE**, vì mô hình sau tỉa có số kênh ra nhỏ và cố
định ở bội số của 2 (4-4-8-8) nên ánh xạ trực tiếp một PE cho mỗi kênh ra, giữ bộ điều
khiển ở dạng máy trạng thái phẳng; đồng thời độ trễ 52 µs đạt được đã nhanh hơn yêu cầu
của bài toán đo ECG liên tục nhiều bậc độ lớn, nên việc rút ngắn thêm độ trễ bằng song
song hóa theo vị trí không mang lại lợi ích thực tế mà chỉ làm tăng số bộ nhân và độ phức
tạp điều khiển.

---

## 2.4. Tổng quan công trình liên quan

### 2.4.1. Từ mô hình tiền nhiệm đến mô hình nhẹ

Các công trình học sâu cho phân loại ECG trải dài từ mô hình rất sâu, độ chính xác cao
nhưng không hướng tới triển khai biên, tới các mô hình nhẹ chuyên biệt cho phần cứng
nhúng. Ở đầu phổ độ sâu, mạng 34 lớp của Hannun và cộng sự [4] đạt độ chính xác phân
loại ngang mức bác sĩ tim mạch chuyên khoa trên dữ liệu ECG di động quy mô lớn; kiến
các kiến trúc học sâu dư (Residual Network — ResNet) chứng minh mạng có thể sâu hàng
chục, hàng trăm lớp nhờ kết nối tắt (skip connection) giải quyết vấn đề suy giảm
gradient; các mô hình lai CNN-LSTM kết hợp tầng tích chập trích đặc trưng với tầng hồi
tiếp mô hình hóa chuỗi cũng đạt độ chính xác rất cao trên bài toán bốn nhóm nhịp. Điểm
chung của nhóm mô hình này là hàng trăm nghìn tới hàng triệu tham số, phù hợp triển khai
trên máy chủ hoặc GPU nhưng vượt xa khả năng của một lõi phần cứng nhúng công suất thấp.

Ở hướng ngược lại, các nghiên cứu triển khai trực tiếp trên FPGA buộc phải cân bằng
giữa độ chính xác và độ phức tạp phần cứng. Bảng 2.2 tóm tắt một số công trình tiêu
biểu theo hướng phân loại ECG trên FPGA.

**Bảng 2.2** — Một số công trình phân loại ECG trên FPGA liên quan

| Công trình | Nền tảng | Kiến trúc mô hình | Tập dữ liệu |
|---|---|---|---|
| Wess et al. [19] | FPGA (ISCAS) | MLP + PCA | MIT-BIH |
| Carreras et al. [20] | FPGA | Temporal Convolutional Network (TCN) | — |
| Srivastava et al. [21] | Artix-7 | Probabilistic Neural Network | MIT-BIH (8 lớp) |
| Ingolfsson et al. [22] | FPGA (AICAS) | ECG-TCN (wearable) | ECG5000 |
| Cheng/Wei et al. [23] | Zynq-7045 | 1D-CNN | Chapman |
| Ran et al. [24] | Zynq-7020 + ARM | CNN quy mô lớn | MIT-BIH |
| Rawal et al. [25] | Zynq UltraScale | 1D-CNN | CinC-2017 / MIT-BIH / PTB |
| **Liu et al. [18]** | **Cyclone V** | **CNN "fully-mapped"** | **Chapman** |
| **Khóa luận này** | **Cyclone V (DE10-Standard)** | **1D-CNN nhẹ (640 tham số)** | **Chapman + Georgia** |

Trong số các công trình trên, **Liu và cộng sự [18]** là công trình đối chứng gần nhất và
trực tiếp nhất với khóa luận: cùng sử dụng dòng FPGA Cyclone V, cùng tập dữ liệu
Chapman, và cùng hướng lượng tử hóa power-of-2 cho phép co giãn dịch bit. Sự tương đồng
này cho phép so sánh trực tiếp về tài nguyên phần cứng, tần số hoạt động và hiệu năng
năng lượng ở Chương 5, đồng thời cũng là cơ sở để khóa luận xác định rõ khoảng trống kỹ
thuật cần giải quyết — trình bày ở Mục 2.4.2 và 2.4.3.

Việc lựa chọn hướng mô hình **nhẹ** (640 tham số, so với hàng trăm nghìn tới hàng triệu
tham số của các mô hình ở đầu phổ độ sâu) trong khóa luận không chỉ là một ràng buộc kỹ
thuật, mà phản ánh xu hướng chung của nhánh nghiên cứu triển khai biên: LightX3ECG [5]
và mô hình bimodal CNN của Yoon và Kang [6] (95,08% trên Chapman chỉ với Lead II) là
hai ví dụ cho thấy độ chính xác cao vẫn đạt được với kiến trúc gọn nhẹ hơn đáng kể so
với nhóm mô hình sâu, khi bài toán được giới hạn đúng phạm vi cần thiết (đơn/thiểu số
chuyển đạo, số lớp nhịp vừa phải).

### 2.4.2. Lượng tử hóa power-of-2 vs general-scale

Như đã trình bày ở Mục 2.2.3, lượng tử hóa INT8 có hai nhánh chính phân biệt bởi cách
chọn hệ số co giãn (scale factor): **general-scale**, cho phép hệ số co giãn là bất kỳ
số thực dương nào, xác định để tối thiểu hóa sai số lượng tử hóa [14]; và **power-of-2**,
giới hạn hệ số co giãn về dạng `2^k` với `k` nguyên [30], [17]. Bảng 2.3 tóm tắt khác
biệt cốt lõi giữa hai hướng ở mức khái niệm; số liệu định lượng thực nghiệm (độ chính
xác, số khối DSP tiết kiệm được) được trình bày ở Chương 3.

**Bảng 2.3** — So sánh khái niệm lượng tử hóa power-of-2 và general-scale

| Tiêu chí | Power-of-2 (log quantization) | General-scale |
|---|---|---|
| Dạng hệ số co giãn | `2^k` (k nguyên) | Số thực dương bất kỳ |
| Phép toán co giãn (rescale) | Dịch bit (bit-shift) | Nhân với hệ số thực |
| Tài nguyên phần cứng cho rescale | Không cần khối nhân (0 DSP) | Cần khối nhân (≥1 DSP mỗi phép rescale) |
| Độ phân giải hệ số co giãn | Thô (rời rạc theo lũy thừa 2) | Mịn (liên tục) |
| Sai số lượng tử hóa kỳ vọng | Cao hơn (do độ phân giải hệ số thô hơn) | Thấp hơn (khớp sát biên độ thực) |

Sự đánh đổi cốt lõi là: general-scale khớp sát hơn với phân bố biên độ thực của dữ liệu
nên về lý thuyết cho sai số lượng tử hóa nhỏ hơn, nhưng phải trả giá bằng khối nhân phần
cứng cho **mỗi lần co giãn** — mà trong một mạng CNN nhiều lớp, phép co giãn xảy ra sau
mỗi lớp tích chập, tức là chi phí này lặp lại nhiều lần trong toàn bộ pipeline. Ngược
lại, power-of-2 chấp nhận hệ số co giãn thô hơn để đổi lấy việc loại bỏ hoàn toàn khối
nhân cho rescale — vốn là một trong những tài nguyên khan hiếm nhất (DSP18) trên FPGA.

Đáng chú ý, việc dùng power-of-2 tự nó **không phải là kỹ thuật mới** — đã được thiết
lập trong tài liệu từ trước [30], [31], và cũng đã được áp dụng trong công trình đối
chứng Liu và cộng sự [18]. Điểm khác biệt nằm ở cách **làm tròn** (rounding) khi thực
hiện phép dịch bit: dịch bit số học đơn thuần (arithmetic right-shift) tương đương với
làm tròn về phía âm vô cùng — gọi là **làm tròn cắt** (floor truncation) — là cách làm
của [7], trong khi cộng thêm một nửa đơn vị bit thấp nhất trước khi dịch — gọi là **làm
tròn nửa lên** (round-half-up) — cho sai số làm tròn kỳ vọng nhỏ hơn về mặt thống kê,
với chi phí phần cứng chỉ là một phép cộng thêm (không cần khối nhân), tức là vẫn giữ
nguyên ưu điểm "0 DSP" của power-of-2:

```
floor truncation:    out = acc >> nb
round-half-up:        out = (acc + 2^(nb-1)) >> nb
```

Sự khác biệt này, cùng với việc thiếu vắng một phân tích định lượng so sánh trực tiếp
power-of-2 và general-scale trên cùng một mô hình, cùng một tập dữ liệu, là một trong
những khoảng trống kỹ thuật mà khóa luận hướng tới lấp đầy — trình bày cụ thể ở
Mục 2.4.3 và triển khai thực nghiệm ở Chương 3.

### 2.4.3. Phân tích khoảng trống

Tổng hợp từ các mục 2.4.1 và 2.4.2, Bảng 2.4 hệ thống hóa các hạn chế thường gặp trong
nhánh nghiên cứu ECG-FPGA hiện có và định hướng giải quyết tương ứng của khóa luận.

**Bảng 2.4** — Phân tích khoảng trống so với công trình liên quan

| Hạn chế thường gặp trong công trình ECG-FPGA hiện có | Hướng giải quyết của khóa luận |
|---|---|
| Dùng dịch bit theo lũy thừa 2 nhưng làm tròn cắt (floor truncation), không phân tích định lượng đánh đổi power-of-2 so với general-scale [18] | Làm tròn nửa lên (round-half-up) thay floor, kèm khảo sát ablation định lượng power-of-2 so với general-scale so với floor (độ chính xác, số DSP) |
| Báo độ chính xác ở mức mô phỏng phần mềm (INT8 simulation) nhưng không chứng minh khớp bit-exact với mạch RTL thực tế | Khung kiểm chứng bit-exact — đối chiếu từng điểm kiểm tra trung gian giữa mô hình Python "vàng" (golden reference) và mô phỏng RTL |
| Đánh giá chỉ trên một tập dữ liệu duy nhất, không trả lời được câu hỏi mô hình có tổng quát hóa (generalize) sang phân bố dữ liệu khác hay không | Đối chiếu cross-check trên tập dữ liệu độc lập (Georgia), phân tích riêng phần suy giảm do lượng tử hóa và phần suy giảm do dịch chuyển phân bố dữ liệu |
| Trọng số nạp cố định vào bitstream lúc tổng hợp, không thể thay đổi khi triển khai | Cơ chế nạp trọng số lúc chạy qua giao diện Avalon-MM, cho phép cùng một bitstream chạy với nhiều bộ trọng số khác nhau |
| Thiếu khảo sát định lượng đánh đổi giữa các phương án tổ chức luồng dữ liệu/song song hóa trên cùng một bài toán | Khảo sát không gian thiết kế (DSE) giữa hai biến thể song song hóa theo kênh và theo vị trí trên cùng một kiến trúc mô hình |

Bốn định hướng trên — làm tròn nửa lên có ablation định lượng, khung kiểm chứng
bit-exact, nghiên cứu chuyển giao đa tập dữ liệu, và khảo sát không gian thiết kế luồng
dữ liệu — được triển khai cụ thể lần lượt ở Chương 3 (phương pháp mô hình và lượng tử
hóa), Chương 4 (kiến trúc phần cứng) và Chương 5 (kiểm định, tổng hợp và đánh giá) của
khóa luận.
