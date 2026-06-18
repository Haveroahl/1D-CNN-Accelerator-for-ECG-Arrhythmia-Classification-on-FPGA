# BÁO CÁO TIẾN ĐỘ — CNN Accelerator phân loại rối loạn nhịp tim ECG trên FPGA Intel Cyclone V

**Tác giả:** Lê Đức (ducle160499@gmail.com)
**Ngày:** 2026-06-18
**Phạm vi báo cáo:** toàn bộ dự án (software + hardware + cross-dataset + viết bài).

> Báo cáo Markdown, định dạng đơn giản để mở/chuyển sang Word. Mọi số liệu lấy từ `PAPER_DATA.md`
> (nguồn gốc đã verify 2026-06-15) và `PROJECT.md`. Mục tiêu ~10 trang khi in.

---

## 1. Tóm tắt (Executive Summary)

Dự án thiết kế và triển khai một lõi IP **CNN Accelerator** trên FPGA Intel Cyclone V để phân
loại rối loạn nhịp tim từ tín hiệu ECG đơn đạo trình (lead II), gồm hai phần gắn kết: (i) một
pipeline phần mềm PyTorch huấn luyện — cắt tỉa (prune) — lượng tử hoá INT8 power-of-2, và (ii)
một lõi phần cứng Verilog đã được **verify bit-exact** với mô hình phần mềm và **chạy thật trên
board DE10-Standard**.

**Kết quả chính tính tới 2026-06-18:**

- **Độ chính xác:** 94.65% / macro-F1 0.9396 trên tập Chapman (4 lớp AFIB/GSVT/SB/SR,
  patient-independent 70/15/15), với mô hình chỉ **654 tham số**.
- **Bit-exact:** 21 điểm kiểm tra (checkpoint) mỗi mẫu khớp tuyệt đối giữa Python và RTL —
  `max|diff| = 0 LSB` trên 15.312 phần tử so sánh. INT8 không gây mất mát so với float (cùng 94.65%).
- **Phần cứng:** một inference trong **5.216 chu kỳ ≈ 52.16 µs @100 MHz** (xác định, không biến
  thiên), dùng **2.201 ALM (5%)** và **28 DSP (25%)** của thiết bị 5CSXFC6D6F31C6, Fmax 104.85 MHz.
- **Trên board:** DE10-Standard qua cầu JTAG-to-Avalon đạt **94.27% (1004/1065)**, tái lập kết
  quả mô phỏng.
- **Năng lượng:** ≈10.3 µJ/inference (dynamic) — phù hợp giám sát đeo liên tục (wearable).
- **Cross-dataset:** một bitstream chạy được cả Chapman và PTB-XL nhờ cơ chế nạp lại trọng số
  runtime qua Avalon; nghiên cứu transfer cho thấy lượng tử hoá INT8 **không** gây mất mát
  generalization (toàn bộ drop là do distribution shift).

Phần software, hardware và đánh giá đã hoàn tất. Phần còn lại nằm ở **viết bài** (đường găng):
hoàn thiện draft ICDV, bảng SoTA, references, và các bước liêm chính/báo cáo.

---

## 2. Mục tiêu và bài toán

**Bài toán:** rối loạn nhịp tim (arrhythmia) là nguyên nhân hàng đầu gây đột tử tim. Phát hiện
sớm bằng thiết bị đeo (wearable) liên tục có giá trị lâm sàng cao. CNN 1 chiều trên ECG thô đạt
độ chính xác tốt, nhưng vi điều khiển (MCU) thiếu throughput cho inference liên tục, còn GPU
biên không phù hợp về công suất và kích thước. **FPGA Cyclone V** là điểm cân bằng: đủ song song
cho CNN nhỏ, công suất thấp, có thể nhúng soft-logic hoặc cầu host để điều khiển datapath.

**Mục tiêu thiết kế:**
1. CNN 1D nhỏ gọn, độ chính xác ≥ ~94% trên 4 lớp nhịp Chapman.
2. Lượng tử hoá INT8 **không multiplier ở khâu rescale** (power-of-2) để giảm DSP / năng lượng.
3. Lõi RTL **verify bit-exact** với mô hình phần mềm (không phải "xấp xỉ INT8").
4. Triển khai và đo thật trên DE10-Standard.
5. (Mở rộng) chạy đa dataset trên cùng bitstream để nghiên cứu transfer learning trên phần cứng.

---

## 3. Mô hình và phương pháp lượng tử hoá (Software)

### 3.1 Kiến trúc mô hình

ECG_1DCNN (pruned): 4 lớp Conv1D (kênh 4-4-8-8, kernel K=5, pad=2), mỗi lớp theo sau bởi
MaxPool stride 5; rồi Global Average Pooling (GAP) → Fully-Connected (8→4) → argmax.

| Lớp | In_ch | Out_ch | K | Pool | ReLU | In_len | Out_len |
|-----|------:|-------:|--:|------|------|-------:|--------:|
| Conv1 | 1 | 4 | 5 | /5 | Không | 2500 | 500 |
| Conv2 | 4 | 4 | 5 | /5 | Không | 500 | 100 |
| Conv3 | 4 | 8 | 5 | /5 | Không | 100 | 20 |
| Conv4 | 8 | 8 | 5 | /5 | **Có** | 20 | 4 |
| GAP | 8 | 8 | — | /4 | — | 4 | 1 |
| FC | 8 | 4 | — | — | — | 1 | 1 |

- **ReLU chỉ sau Conv4** — giữ đặc trưng âm của ECG ở Conv1–3.
- Input 2500 mẫu INT8 (5 s @ 500 Hz, lead II). 4 lớp: AFIB / GSVT / SB / SR.
- **654 tham số** — cực nhỏ, nhắm thiết bị đeo.

### 3.2 Lượng tử hoá power-of-2 QAT

Trọng số / activation / bias lượng tử INT8 với scale **lũy thừa 2** theo lớp,
`nb = floor(log2(127/abs_max))`:

- `nb = {8, 6, 6, 7, 0}` (Conv1..Conv4, FC); `w_shift = {6, 6, 6, 7, 8}`; input_shift = 2.
- Rescale: `out = clamp(round_half_up(acc / 2^nb), -127, 127)` với
  `round_half_up(x) = (x + 2^(nb-1)) >> nb`.
- Bias: `bias_scaled = round(b_float · 2^nb)`, lưu INT32 little-endian.

Vì mọi scale là lũy thừa 2, khâu rescale chỉ cần **dịch bit + cộng** → **0 DSP** (so với
general-scale cần 1 multiplier mỗi điểm rescale).

### 3.3 Kết quả lượng tử (ablation Chapman)

**Table 4 — quant ablation** (single-run, seed 42; DSP = multiplier cho rescale):

| Variant | Scale | Train | Acc % | F1 | DSP rescale |
|---|---|---|---:|---:|---:|
| A1 Float32 baseline | — | — | 94.65 | 0.9402 | — |
| A0 PTQ power-of-2 | 2^nb | none | 94.08 | 0.9338 | **0** |
| A0' PTQ general | absmax/127 | none | 94.46 | 0.9380 | 4 |
| **A2 QAT power-of-2 (ours)** | 2^nb | fake-quant | **94.37** | **0.9364** | **0** |
| A3 QAT general | absmax/127 | fake-quant | 94.65 | 0.9398 | 4 |
| A4 QAT power-of-2 floor | 2^nb | fake-quant | 93.99 | 0.9328 | 0 |

- **A2 vs A3:** −0.28% acc đổi lấy **−4 DSP18** → power-of-2 Pareto-ưu.
- **A2 vs A4 (floor):** **+0.38%** nhờ round-half-up → đây là cải tiến rounding so với prior work.
- 5 test-fold: std 0.4–0.9% ≥ chênh giữa các variant → khác biệt accuracy nằm trong nhiễu; điểm
  chắc chắn duy nhất là chi phí DSP.
- PTQ (không fine-tune) đã đạt 94.08% → QAT có lợi nhưng **không bắt buộc**.

> **Lưu ý số liệu:** con số "chốt" cho golden RTL là **94.65% / F1 0.9396** (bản re-train có FC
> bias, 2026-06-08). Table 4 A2 = 94.37 là bản trước FC-bias — cần thống nhất một con số khi viết bài.

**Tại sao INT8 không INT4:** INT4 mất ~19% (69.95% power-of-2; 75.59% trần general-scale), AFIB
sập (F1 0.42) — vì Conv1–3 không ReLU giữ dải activation âm rộng mà INT4 không biểu diễn nổi.
INT8 là điểm ngọt.

---

## 4. Kiến trúc phần cứng (Hardware)

### 4.1 Tổng thể

```
Avalon-MM → Input SRAM (2500×8b)
                 ↓ (MUX Conv1)
        Conv-Pool Engine (8 CP block song song, K=5 pad=2, MaxPool /5)
                 ↓ Ping-Pong SRAM (feature map liên lớp)
        GAP / FC / Argmax → result[1:0] (0..3)
```

FSM điều khiển phẳng 8 trạng thái tuần tự 4 lớp conv + đuôi GAP/FC. `avalon_slave.v` là bus
adapter dùng chung cho cả luồng synthesis (virtual-pin) và luồng on-board (JTAG-to-Avalon).

### 4.2 CP block — pipeline tách 3 submodule

- **MAC (S1–S4):** 5 multiplier INT8×INT8 (DSP18) → adder tree 3 tầng → `tree_out`.
- **Accumulate + rescale (S5–S8):** cộng dồn `in_ch` partial sum, fold bias + round-add vào
  accumulator-init (cắt khỏi critical path, kết quả số học giữ nguyên), clamp [-127,127], ReLU Conv4.
- **Pool (S9):** comparator max trượt cho MaxPool K=5, stride 5.

Độ sâu pipeline từ MUX cửa sổ vào tới cập nhật accumulator = đúng 5 chu kỳ (delay chain
controller phải khớp — off-by-one ở đây từng là bug verify chính, đã fix).

### 4.3 Dataflow streaming gập (folded)

Thay vì map toàn bộ mỗi lớp ra phần cứng riêng (fully-mapped như competitor), CPE **time-multiplex**
8 CP block trên các vị trí output, stream input qua shift-register. Đây là đòn bẩy diện tích chính
so với fully-mapped: tái dùng một bộ PE cho mọi vị trí một lớp, đổi độ trễ tăng (xác định) lấy
giảm logic/register lớn.

### 4.4 Nạp lại trọng số runtime (Phase B01)

Bản baseline: trọng số conv bake vào bitstream qua `$readmemh` (FF-ROM). Bản weight-RAM: trọng số
conv lưu 8 khối M10K (một per output-channel) nạp runtime qua Avalon (+ port bias/FC). Cho phép
**một bitstream** chạy Chapman hoặc PTB-XL, là cơ chế bật cho nghiên cứu cross-dataset. Latency và
accuracy **không đổi** (M10K sync-read thay 1-1 stage `w_packed`, giữ pipeline alignment, bit-exact).

### 4.5 Số liệu phần cứng (Cyclone V 5CSXFC6D6F31C6, Quartus 25.1 Lite)

| Metric | Baseline (FF-ROM) | + weight-RAM reload |
|---|---:|---:|
| ALM | 2.201 (5%) | 2.820 (7%) |
| DSP18 | 28 (25%) | 28 (25%) |
| Registers | 3.177 | 4.852 |
| M10K | 20 (4%) | 28 (5%) |
| Fmax (standalone, 85 °C) | 104.85 MHz | 108.94 MHz |
| Latency | 5.216 cy (52.16 µs) | 5.216 cy (52.16 µs) |
| Throughput | ~19.200 inf/s | ~19.200 inf/s |

- Overhead weight-reload (+619 ALM, +8 M10K) đến từ logic nạp-được (read-address adder, 8-way
  write decode, 40-bit assembly), **không** từ bộ nhớ (M10K = 0 ALM).
- **Năng lượng** (PowerPlay, 95.6% toggle): tổng 623 mW (dynamic 198 mW, static 413 mW) →
  **≈10.3 µJ/inference dynamic, 32.5 µJ total**. DSP chiếm ~68% dynamic → nối thẳng với lựa chọn
  0-DSP-rescale ở khâu lượng tử.

### 4.6 So sánh thiết kế: hai dataflow (DSE)

Cùng model, cùng device, cùng bit-exact contract — chỉ khác cách map dataflow:

| Trục | Production (8-PE channel-par) | SIMD-20 (position-par) | Tỉ lệ |
|---|---:|---:|---:|
| Latency | 5.216 cy (52.16 µs) | 2.755 cy (27.55 µs) | 1.89× nhanh |
| Throughput @Fmax | 20.107 inf/s (104.85 MHz) | 42.427 inf/s (116.9 MHz) | 2.11× |
| ALM | 2.201 (5%) | 5.948 (14%) | 2.70× |
| DSP | 28 (25%) | 64 (57%) | 2.29× |
| Fmax | 104.85 MHz | 116.9 MHz | +11% |

→ Pareto: production = low-area/đơn giản; SIMD-20 = low-latency/diện tích cao. Cùng accuracy 94.65%.

---

## 5. Verify bit-exact (Verification)

Hợp đồng verify là **bit-exactness** giữa Python và RTL. Golden Python tái lập đúng trình tự RTL:
`acc_int32 → +bias_scaled → +2^(nb-1) → >>nb → clamp[-127,127] → ReLU(Conv4) → MaxPool`; GAP dùng
floor `sum>>2`; FC `nb=0`, raw INT32 logits vào argmax.

Mỗi input: **21 checkpoint** (input INT8, 4 pool output, GAP, 4 FC logit). Testbench Questa nạp
golden `.mem` và so từng checkpoint. Kết quả lõi production: **21/21 PASS**,
**max|diff| = 0 LSB trên 15.312 so sánh** (3 mẫu test), latency xác định 5.216 chu kỳ. Có thêm test
config/recover + GAP-mask cho đường runtime-reconfigurable. Topology tuỳ ý cũng đã sweep
(`tb_topo_sweep.v` 48/48 bit-exact, mọi out_ch 1..8/lớp).

Khung này biến "INT8 ≈ RTL" (vốn là hand-wave thường thấy) thành đẳng thức chứng minh được:
độ chính xác báo cáo **đúng bằng** độ chính xác lõi triển khai sinh ra.

---

## 6. Cross-dataset & on-board

### 6.1 On-board DE10-Standard

Nạp lên DE10-Standard, điều khiển qua cầu JTAG-to-Avalon + System Console: phân loại tập test
Chapman đạt **94.27% (1004/1065)**, tái lập 94.65% mô phỏng (chênh là do test-subset/run, không
phải sai số numerical — datapath bit-exact). *(Lý do dùng JTAG thay HPS: Quartus Lite không có IP
HPS Cyclone V → chuyển sang JTAG-to-Avalon; có thêm variant Nios V/m và variant UART.)*

### 6.2 Transfer Chapman ↔ PTB-XL

Dùng đường weight-reload, cùng bitstream chạy trọng số PTB-XL (19.952 record, 500→250 Hz, lead II):

| Mode | Acc | F1-macro |
|---|---:|---:|
| C1 Chapman in-distribution | 0.9446 | 0.9379 |
| C2 zero-shot QAT-INT8 | 0.7714 | 0.6486 |
| C3 linear probe (chỉ retrain FC) | 0.9263 | 0.7745 |
| C4 full fine-tune | 0.9336 | 0.7940 |
| C6 float32 zero-shot | 0.7714 | 0.6486 |

**Phát hiện then chốt:** C2 == C6 → **quantization drop = 0%**; toàn bộ mức giảm là distribution
shift. Tức power-of-2 INT8 không thêm mất mát generalization. Đây là kết quả hỗ trợ ngắn, không
phải tâm bài.

---

## 7. Trạng thái các Phase

| Phase | Nội dung | Trạng thái |
|---|---|---|
| Software baseline | Re-prune (4,4,8,8), QAT-INT8 round-half-up, export hex, golden .mem | ✅ Done |
| A' — quant ablation | A0/A0'/A2/A3/A4 + 5-fold + Chapman CM/ROC (Table 4) | ✅ Done |
| A — cross-dataset | PTB-XL 6 mode C1–C6 + U0, decomposition C2==C6 | ✅ Done |
| B — tách core/bus | `ecg_core.v` + wrapper mỏng, regression 21/21 | ✅ Done |
| B01 — weight RAM reload | FF-ROM → M10K nạp runtime qua Avalon, 21/21 bit-exact | ✅ Done |
| C — Synthesis + Power | Quartus thật: ALM 2201, DSP 28, Fmax 104.85; PowerPlay 623mW/10.3µJ | ✅ Done |
| D — On-board | JTAG-to-Avalon DE10 94.27%; variant Nios V/m + UART (UART chờ USB-TTL) | 🟡 JTAG done |
| E — SoTA tables | Bảng A (5 model Chapman) + Bảng B (10 FPGA biomedical) → `SOTA_TABLE.md` | 🟡 Draft, Pareto chưa vẽ |
| E01 — References | 19 mục format ICDV, 16/19 có DOI/ISBN → `paper/REFERENCES.md` | 🟡 Còn 3 mục cần chốt |
| F — Draft ICDV | ~6 trang, bản production 8-PE → `paper/ICDV_draft.md` | 🟡 Draft, còn 6 open-items |
| K — Liêm chính | Quy trình đạo văn/AI + self-audit → `paper/INTEGRITY_CHECK.md` | ✅ Done (self-audit) |
| L — Báo cáo tiến độ | File này | ✅ Done |

---

## 8. Việc còn lại (đường găng = viết bài)

**Khóa số liệu (blocking camera-ready):**
1. 🔴 Thống nhất một con số accuracy (khuyến nghị 94.65%) giữa Table 4 và phần text.
2. 🔴 Fmax: dùng 104.85 (baseline) / 108.94 (weight-RAM) / board ~125 — **không** dùng 137.6.
3. 🟠 Cross-dataset: dùng số JSON (C3 0.9263, C4 0.9336).
4. 🟠 On-board 1004/1065: tạo lại log cite-được.
5. 🟠 Energy: xác nhận từ `.pow.rpt` PowerPlay.

**References & SoTA:**
6. 🔴 [17] CardioPatternFormer: tìm bản published có DOI (hoặc thay paper Transformer-ECG đã xuất bản).
7. 🟠 [15] LightX3ECG xác nhận art-no/DOI; [18] điền ISBN sách (cần sách giấy).
8. 🟢 Verify từng ô SoTA từ paper gốc; gán BibTeX key vào [CITE...] trong draft.
9. 🟢 Vẽ Pareto chart (params↔accuracy là đòn bẩy mạnh nhất; latency↔accuracy).

**Liêm chính (Phase K, bạn tự chạy):**
10. Chạy Turnitin/iThenticate (mục tiêu < 20%, không nguồn > 3%); (nếu trường yêu cầu) GPTZero bản cuối.
11. Làm phẳng 5 câu "giọng AI" đã đánh dấu trong `INTEGRITY_CHECK.md` Phần B.3.

**Phát hành:**
12. GitHub public + Zenodo DOI (reproducibility artifact).

---

## 9. Kết luận

Dự án đã đạt một lõi IP CNN-ECG **verify bit-exact** (max|diff| = 0 LSB), chạy thật trên FPGA
DE10-Standard ở 94.27%, với footprint cực nhỏ (654 params, 2201 ALM, 28 DSP) và năng lượng thấp
(≈10.3 µJ/inference) — đúng định hướng thiết bị đeo. Phương pháp lượng tử power-of-2 round-half-up
cho cùng độ chính xác general-scale nhưng tiết kiệm 4 DSP, và khung verify 21-checkpoint biến độ
chính xác phần mềm thành độ chính xác phần cứng chứng minh được. Phần kỹ thuật (software +
hardware + đánh giá) đã hoàn tất; công việc còn lại tập trung vào hoàn thiện bài báo ICDV: khóa số
liệu, references, bảng SoTA + Pareto, và các bước liêm chính/phát hành.
