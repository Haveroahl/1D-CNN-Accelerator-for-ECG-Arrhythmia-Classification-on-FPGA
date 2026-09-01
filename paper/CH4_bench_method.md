# Chương 4 — Phương pháp đo latency hai nền tảng phần mềm

Mục này mô tả cách đo latency của hai baseline phần mềm (PyTorch INT8 và C
portable) để so sánh với accelerator RTL. Nguồn: `software/python/bench_sw_vs_hw.py`
và `hardware/fpga/sw/niosv/bench_c_host.c` + `cnn_sw.c`.

---

## 4.x.1 Nguyên tắc thiết kế phép đo

Một so sánh phần cứng–phần mềm chỉ có ý nghĩa khi **duy nhất nền tảng thay đổi**.
Bốn nguyên tắc sau được áp dụng để bảo đảm điều đó.

### (a) Đồng nhất thuật toán — cùng số học INT8 bit-exact

Cả ba nền tảng thực thi **cùng một pipeline INT8**, không phải cùng một mô hình
xấp xỉ nhau:

```
acc_int32 = Σ (x_int8 · w_int8) + bias_scaled
out_int8  = clamp( (acc + 2^(nb-1)) >> nb , -127, 127 )
[ReLU nếu Conv4] → MaxPool(K=5,S=5) → GAP(floor) → FC → Argmax
```

Đây là điểm khác biệt phương pháp luận so với các công trình trước (ví dụ Liu
2023 so *float* Keras trên CPU với *INT8* trên FPGA): so sánh như vậy đổi **hai
biến cùng lúc** (thuật toán + nền tảng), nên tỉ số thu được không quy được cho
kiến trúc. Ở đây số học là hằng số, do đó chênh lệch latency **chỉ** phản ánh
kiến trúc thực thi.

Tính đồng nhất này không phải giả định mà được **kiểm chứng**: cả hai baseline
được so nhãn dự đoán với golden Python trên toàn bộ 500 mẫu, kết quả
**0/500 sai khác** (`BIT-EXACT PASS`). Đây là hệ quả trực tiếp của framework
bit-exact (đóng góp C2).

### (b) Đồng nhất biên đo — compute-only

Accelerator được đo bằng 5216 chu kỳ FSM, tức **chỉ phần tính toán**, không gồm
thời gian nạp dữ liệu qua bus. Để so sánh công bằng, hai baseline phần mềm cũng
được đo compute-only:

- dữ liệu đã nằm sẵn trong RAM trước khi bấm đồng hồ;
- không tính thời gian đọc file, tiền xử lý, hay chuyển đổi kiểu.

Nếu tính cả tiền xử lý, latency phần mềm sẽ tăng và bảng sẽ có lợi cho phần cứng
một cách không chính đáng.

### (c) Đồng nhất dữ liệu vào

Cả hai baseline chạy trên **đúng cùng 500 bản ghi** của tập test Chapman, theo
đúng thứ tự. Cụ thể, vòng đo PyTorch đồng thời ghi lại vector INT8 đầu vào và
nhãn dự đoán ra `inp.bin` / `pred.bin`; baseline C nạp đúng hai tệp này. Nhờ đó
không tồn tại khả năng hai baseline chạy trên phân phối dữ liệu khác nhau.

### (d) batch = 1

Accelerator xử lý một bản ghi mỗi lần, nên mọi baseline dùng batch = 1. Dùng
batch lớn sẽ cho CPU lợi thế amortize mà phần cứng không có, làm phép so sánh
mất ý nghĩa vật lý.

---

## 4.x.2 Nền tảng 1 — PyTorch INT8 (mức framework)

**Mục đích.** Đại diện cho cách đo phổ biến trong y văn ECG-FPGA: chạy mô hình
bằng framework học sâu (Keras/PyTorch) trên CPU rồi lấy tỉ số với FPGA.

**Cài đặt.** Hàm `int8_forward_golden()` — chính là hàm sinh golden reference cho
kiểm chứng RTL — được gọi lặp lại, mỗi lần một mẫu, trong ngữ cảnh
`torch.no_grad()` và `model.eval()`. Số luồng đặt tường minh bằng
`torch.set_num_threads()`.

**Đồng hồ.** `time.perf_counter()` — bộ đếm đơn điệu (monotonic) độ phân giải cao
của Python, không bị ảnh hưởng bởi thay đổi giờ hệ thống.

**Trình tự đo.**

```
1. Nạp checkpoint, dựng model, chuyển eval mode
2. Nạp toàn bộ tensor test vào RAM
3. Warm-up: 20 lần suy luận (không tính giờ)
4. Với i = 0..499:  t0 ← perf_counter();  infer(i);  t[i] ← perf_counter() − t0
5. Thống kê: median, mean, min, max, p95
```

**Vai trò của warm-up.** 20 lần chạy đầu bị loại khỏi thống kê để triệt tiêu các
hiệu ứng khởi động một lần: cấp phát bộ đệm nội bộ, khởi tạo thread pool, nạp
mã vào cache lệnh, và làm nóng cache dữ liệu. Không có warm-up, mẫu đầu tiên có
thể chậm hơn hàng chục lần và làm sai lệch mean.

---

## 4.x.3 Nền tảng 2 — C portable biên dịch `-O2`

**Mục đích.** Loại bỏ chi phí framework để thu được latency của **bản thân thuật
toán** trên CPU. Đây là baseline tham chiếu chính.

**Cài đặt.** `cnn_sw.c` (117 dòng) — C scalar thuần, viết cho soft-core RISC-V
Nios V/m, gồm vòng lặp lồng `oc → p → j → ic → kk`, không dùng SIMD intrinsics,
không `restrict`, không đa luồng. Đây **cùng một tệp nguồn** mà firmware Nios V
biên dịch, nên baseline phần mềm và firmware luôn đồng bộ.

Biên dịch: `gcc -O2` (mingw-w64 GCC 14.2.0) — mức tối ưu quy ước cho bản dựng
embedded.

**Đồng hồ.** `QueryPerformanceCounter` trên Windows (`clock_gettime(CLOCK_MONOTONIC)`
trên POSIX), bọc trong hàm `now_us()`. Đây là bộ đếm phần cứng độ phân giải sub-µs,
cần thiết vì thời gian đo chỉ cỡ ~85 µs.

**Trình tự đo.** Giống hệt nền tảng 1 (warm-up 20 mẫu, đo từng mẫu, cùng thống kê),
cộng thêm bước **kiểm chứng bit-exact**: mỗi mẫu so nhãn dự đoán với `pred.bin`,
báo `BIT-EXACT PASS` khi 0 sai khác. Nếu sai khác, số latency bị coi là không hợp lệ.

**Ý nghĩa của baseline này.** Nó cho biết bao nhiêu phần trong latency của
nền tảng 1 là *phép tính thực sự* và bao nhiêu là *chi phí điều phối framework*.

---

## 4.x.4 Xử lý thống kê

| Đại lượng | Lý do dùng |
|---|---|
| **median** | Số chính được báo. Bền với ngoại lai do OS scheduling (một lần bị preempt không kéo lệch median như kéo lệch mean) |
| mean | Báo kèm để đối chiếu; lệch nhiều so với median là dấu hiệu đuôi phân phối dài |
| min | Xấp xỉ latency "không bị nhiễu" — giới hạn dưới của nền tảng |
| **p95** | Đại diện worst-case thực dụng, dùng cho lập luận về tính tiền định |
| max/min | Hệ số phân tán, định lượng jitter |

Mỗi cấu hình được chạy lặp **3–5 lần độc lập** và báo khoảng giá trị, thay vì tin
vào một lần chạy duy nhất — vì tải hệ thống nền có thể thay đổi giữa các lần.

**Vì sao p95 quan trọng hơn median trong ứng dụng này.** Thiết bị theo dõi ECG
liên tục phải được thiết kế theo **trường hợp xấu nhất**, không theo trung bình.
Accelerator có p95 = median = max (cố định 5216 chu kỳ mọi đầu vào), trong khi
hai baseline phần mềm phân tán 2.0–4.4× do OS scheduling và cache. Đây là khác
biệt về *bản chất*, không phải về *mức độ*.

---

## 4.x.5 Cấu hình thực nghiệm

| Thành phần | Chi tiết |
|---|---|
| CPU | Intel Core i7-11850H, 8C/16T, 2.50 GHz base |
| OS | Windows 11 Pro build 26200 |
| Framework | PyTorch 2.12.0+cpu, Python 3.14.4 |
| Trình biên dịch C | mingw-w64 GCC 14.2.0, `-O2` |
| Accelerator | Cyclone V 5CSXFC6D6F31C6 @ 100 MHz |
| Số luồng | 1 (đơn luồng cho cả hai baseline) |
| Tập mẫu | 500 bản ghi test Chapman, batch = 1 |
| Warm-up | 20 mẫu, loại khỏi thống kê |

---

## 4.x.6 Giới hạn của phương pháp

Cần nêu tường minh khi trình bày kết quả:

1. **Mức tối ưu biên dịch.** Số báo cáo ứng với `-O2`. Các mức tối ưu cao hơn kích
   hoạt bộ auto-vectorize của GCC trên kernel này và làm giảm đáng kể latency CPU;
   do đó biên 1.65× là **đặc thù cho bản dựng `-O2` / lớp embedded**, không giữ
   được với bản dựng desktop tối ưu mạnh. Luôn nêu cờ biên dịch kèm con số.

2. **Baseline C đơn luồng.** So 1 luồng CPU với 8 PE là có lợi cho phần cứng.
   Không đo đa luồng vì với 2500 mẫu, khối lượng tính mỗi lần suy luận quá nhỏ để
   bù chi phí điều phối luồng — chính các hàng PyTorch đa luồng chứng minh điều
   này (tăng luồng làm *chậm* đi).

3. **Chưa có baseline ARM / soft-core.** Không có board Cortex-A, nên chưa có đối
   chứng cùng lớp thiết bị. `main.c` đã có sẵn đo bằng bộ đếm `mcycle` của RISC-V
   trên Nios V/m nhưng chưa chạy on-board.

4. **Chưa đo công suất CPU.** Không có RAPL/HWiNFO trên máy thí nghiệm, và TDP
   danh định là chỉ số thiết kế nhiệt chứ không phải phép đo tải thực. Do đó
   **không** báo tỉ số năng lượng CPU-vs-FPGA.

---

## 4.x.7 Tóm tắt: vì sao phép đo này defendable

| Rủi ro phương pháp thường gặp | Cách xử lý ở đây |
|---|---|
| So khác thuật toán (float vs INT8) | Cùng pipeline INT8, verify 0/500 sai khác |
| So khác dữ liệu vào | Cùng `inp.bin`, cùng thứ tự |
| So khác biên đo (có/không tiền xử lý) | Cả ba đều compute-only |
| Nhiễu OS làm lệch kết quả | median + warm-up + lặp 3–5 lần |
| Baseline yếu làm phần cứng trông tốt | Báo **cả** hàng framework và hàng C biên dịch, chỉ rõ hàng framework thổi phồng ~5× |
| batch lớn thiên vị CPU | batch = 1 trên mọi nền tảng |
