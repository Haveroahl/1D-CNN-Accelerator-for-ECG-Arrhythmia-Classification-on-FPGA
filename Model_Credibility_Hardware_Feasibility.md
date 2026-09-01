# Model Credibility & Hardware Feasibility

> Draft cho Discussion (Section 7). Mọi số dẫn từ kết quả thật: `ablation_quant/TABLE4_FINAL.md`,
> `cross_eval/ptbxl_cross_eval.json`, Phase C synthesis (`output_files/*.summary`), PowerPlay (`tb_top.vcd`).

---

## 7.x Model Credibility After Cross-Check

Độ tin cậy của mô hình được đánh giá trên **hai trục độc lập** — *quantization fidelity* (lượng tử
hoá có trung thực với float không) và *generalization* (model có tổng quát ra phân phối khác không) —
vì hai trục này trả lời hai câu hỏi khác nhau và cross-check cho kết quả khác nhau ở mỗi trục.

### (a) Quantization fidelity — độ tin cậy cao

Cross-check giữa float32 và INT8 cho thấy lượng tử hoá power-of-2 **gần như không làm mất accuracy**:

- **In-distribution (Chapman):** A2 (QAT power-of-2 INT8) đạt **94.37%** so với float32 **94.65%** —
  chênh chỉ **−0.28%**, nhỏ hơn cả độ lệch chuẩn giữa các lần train (~0.4%).
- **Bit-exact verification:** 21/21 checkpoint (input, 4 pool output, GAP, FC logits) match **bit-exact**
  giữa Python QAT model và RTL simulation. Nghĩa là con số đo bằng phần mềm **chính là** con số sẽ chạy
  trên phần cứng — không có khe hở "INT8 simulation vs RTL" thường gặp trong literature.
- **Decomposition quant vs distribution:** ở cross-dataset zero-shot, INT8 (C2, 77.14%) **bằng đúng**
  float32 (C6, 77.14%). Tức là **0%** phần drop khi đổi dataset đến từ lượng tử hoá; toàn bộ đến từ
  distribution shift. Đây là bằng chứng mạnh rằng INT8 trung thực với float kể cả khi rời phân phối nguồn.

→ **Kết luận trục (a):** về lượng tử hoá và tính tái lập, mô hình **đạt độ tin cậy cao** — đủ để khẳng
định cái deploy lên FPGA tương đương cái đánh giá bằng phần mềm.

### (b) Generalization — độ tin cậy có điều kiện, đã được định lượng

Cross-dataset Chapman→PTB-XL cho drop lớn nhưng **đã được phân tích bản chất, không phải lỗi năng lực**:

| Mode | Acc | F1-macro | Diễn giải |
|---|---|---|---|
| C1 in-distribution | 94.46% | 0.938 | Baseline |
| C2 zero-shot INT8 | 77.14% | 0.649 | Distribution-shift cost |
| C3 linear-probe (chỉ train lại FC) | 92.63% | 0.775 | Conv features của A vẫn hữu dụng |
| C6 zero-shot float32 | 77.14% | 0.649 | = C2 → quant drop = 0% |

- Drop về **77% là chi phí distribution-shift**, không phải thiếu năng lực hay lỗi lượng tử: U0
  (unpruned float32) cũng ~77.2%, xác nhận pruning/quant không gây thêm tổn thất.
- Confusion matrix C2 cho thấy lỗi tập trung ở ranh giới lâm sàng **SB/SR** (F1 SB chỉ 0.31): hai dataset
  định nghĩa ngưỡng nhịp chậm (≈60 bpm) khác nhau → đây là vấn đề **biên lâm sàng**, không giải được
  bằng tinh chỉnh model.
- **Linear-probe phục hồi lên 92.63%** chỉ bằng cách train lại lớp FC (giữ nguyên conv) → chứng minh
  feature extractor học được biểu diễn **có ý nghĩa, transfer được**, không overfit nhiễu của Chapman.
  Đây là điểm **củng cố** độ tin cậy của model, không phải làm giảm.

→ **Kết luận trục (b):** generalization tin cậy **có điều kiện** — model không zero-shot tốt qua dataset
mới (đặc tính chung của 1D-CNN học end-to-end, bám phân phối nguồn), nhưng feature có giá trị và phục hồi
nhanh bằng linear-probe. Quan trọng: mức drop đã được **decompose và định lượng**, nên là *limitation
hiểu rõ*, không phải *điểm yếu không kiểm soát*.

### Lưu ý về robustness (trung thực với reviewer)

5-fold trong nghiên cứu này là **test-fold robustness trên checkpoint cố định** (đo variance của
test-split + re-quantization), **không phải leak-free k-fold CV train-lại**. std quan sát được 0.4–0.9%
**≥ mọi khác biệt accuracy giữa các quant-variant** trong Table 4. Do đó nghiên cứu **không claim**
variant nào "accuracy tốt nhất"; khác biệt chắc chắn duy nhất là **DSP cost** (power-of-2 = 0 vs
general-scale = 4 DSP18 ở khâu rescale).

---

## 7.y Tại sao mức độ tin cậy này đủ để hiện thực phần cứng

Lập luận cốt lõi: **tính khả thi phần cứng phụ thuộc trục (a), không phụ thuộc trục (b).** Phần cứng cần
một model *cố định, trung thực bit-exact, accuracy in-distribution cao, và rẻ tài nguyên* — tất cả đều
nằm ở trục (a), nơi độ tin cậy đã cao. Generalization yếu (trục b) là vấn đề *khoa học dữ liệu*, được giải
quyết ở tầng phần mềm (linear-probe / fine-tune) **trước khi** nạp weight, không cản trở việc đúc IP core.

### Bằng chứng khả thi — đã hiện thực, không phải dự đoán

1. **Bit-exact đảm bảo tương đương SW↔HW:** 21/21 checkpoint match → không có rủi ro "model chạy đúng
   trong Python nhưng sai trên FPGA". Đây là tiền đề bắt buộc để dám đưa xuống phần cứng, và đã thỏa.

2. **Synthesis thật trên Cyclone V (5CSXFC6D6F31C6), không estimate:**
   - DSP **28/112 (25%)**, ALM **2,261/41,910 (5%)**, Registers 3,196, RAM 20/553 (4%).
   - **Timing PASS @ 100 MHz**, worst setup slack **+0.508 ns**, TNS = 0 → Fmax ≈ **105 MHz**.
   - Latency deterministic **5,216 cycle = 52.16 µs/inference**, throughput ~19,200 inf/s.
   - Footprint nhỏ áp đảo so với fully-mapped SoTA (ALM 5% vs ~51%) → phù hợp wearable.

3. **Lượng tử hoá power-of-2 làm phần cứng rẻ hơn mà không hy sinh accuracy:**
   - Rescale chỉ là **shift + add → 0 DSP** (so với general-scale cần 1 multiplier/rescale = +4 DSP18),
     trong khi accuracy ngang nhau (Δ < std). Cross-check cho phép chọn phương án rẻ DSP một cách
     *có cơ sở định lượng*, không phải đánh đổi mù.
   - DSP chiếm **68% dynamic power** (135/198 mW) → loại multiplier khỏi rescale vừa giảm DSP count vừa
     giảm dynamic power — biến trục (a) thành lợi thế năng lượng trực tiếp.

4. **INT8 là điểm vận hành đúng, đã chứng minh bằng số:** bit-width ablation cho thấy INT4 mất ~19%
   accuracy (AFIB F1 sập còn 0.42) kể cả ở trần general-scale. INT8 vừa đủ giữ morphology ECG (đặc biệt
   AFIB), vừa khả thi phần cứng → không cần và không nên xuống thấp hơn.

### Năng lượng (PowerPlay, VCD thật 95.6% toggle)

- Total 623 mW (Dynamic 198 mW / Static 413 mW) → **Energy/inference = 10.3 µJ (dynamic) / 32.5 µJ
  (total)**. Static cao là *device tax* của chip lớn DE10 (có HPS), không phải lỗi thiết kế; dynamic
  energy bất biến theo clock.

### Khoảng trống còn lại (trung thực)

Tính khả thi đã chứng minh đến mức **synthesis + timing + power thật**. Còn lại:
- **On-board validation (Phase D):** model mới verify ở simulation + synthesis, **chưa chạy thật trên
  board**. Hướng triển khai: **Nios V/m soft-core** (Quartus Lite không có HPS hard IP cho Cyclone V) —
  on-chip RAM bare-metal, nạp ECG → start → đọc result.
- **Cross-dataset on-hardware:** nạp weight PTB-XL qua Avalon cần cơ chế weight-reload (weight RAM) —
  enabling mechanism cho C3, chưa hiện thực.

→ **Kết luận:** mức độ tin cậy hiện tại — *quantization bit-exact trung thực, accuracy in-distribution
94.4%, footprint 5% ALM / 25% DSP, timing PASS @ 105 MHz, 0-DSP rescale* — **đủ và đã được dùng** để hiện
thực một IP core verify đầy đủ. Generalization yếu không cản trở điều này vì nó được xử lý ở tầng phần mềm
trước khi nạp weight. Bước duy nhất còn lại để khép vòng "khả thi hoàn toàn" là validation on-board
(Phase D), đang triển khai.
