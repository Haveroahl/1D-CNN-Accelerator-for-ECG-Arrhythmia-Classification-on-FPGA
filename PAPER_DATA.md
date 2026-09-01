# PAPER_DATA.md — Số liệu thật, gom 1 chỗ để viết luận văn

> Nguồn duy nhất khi viết. MỌI số ở đây verify từ report/json/golden gốc.
> **Cập nhật 2026-07-30**: (a) đổi dataset chính Chapman → **ningba** (re-train 2026-07-28);
> (b) đồng bộ số synth với **ROM build** (`hardware/RTL/`, compile 2026-07-07);
> (c) **thu hẹp phạm vi** — xem khung dưới.
> Chỗ nào doc cũ lệch → ghi chú "⚠️ doc cũ ghi X, dùng Y".
> Số chưa verify được từ file → 🔲 NEEDS-SOURCE, phải chạy/tìm lại trước khi đưa vào bài.

> ## ⚠️ PHẠM VI KHÓA LUẬN
> Khóa luận **chỉ** trình bày bản **RTL ROM** (`hardware/RTL/`) chạy trên **DE10-Standard**
> (Cyclone V). Các hướng sau **KHÔNG đưa vào bài** và đã được xóa khỏi file này:
>
> | Đã loại | Lý do loại khỏi phạm vi |
> |---|---|
> | Biến thể **SIMD-20** position-parallel + bảng DSE 2 dataflow | Không thuộc thiết kế nộp; độ trễ 52 µs đã dư 4 bậc độ lớn nên song song theo vị trí vô ích |
> | Cơ chế **weight-load qua Avalon** (`RTL_weight/`, Phase B01) | Bản nộp dùng ROM `$readmemh`; weight-load chỉ là công cụ nghiên cứu cross-dataset |
> | **Elastic-Pareto** (nhiều topology / 1 bitstream) | Phụ thuộc weight-load ở trên |
> | Port **DE0-Nano / Cyclone IV E** + power gate-level SDF | Ngoài device mục tiêu; năng lượng chỉ báo số DE10 |
>
> **Năng lượng chỉ dùng số DE10**: **536.08 mW → 27.96 µJ/inf**, **kèm caveat confidence Low**
> (§6a). Không dùng số DE0-Nano (246 mW / 12.84 µJ) nữa.
> Số cũ 805.53 mW / 42.02 µJ đo trên netlist 06-22 + VCD sai chế độ → **đã thay** (§6a).
>
> Các mục đã xóa được sao lưu tại `hardware/fpga/output_files/*` (số gốc vẫn còn trong
> report Quartus) và trong git history (`git show f81beb9`, `ec81a36`, nhánh `feature/simd-spec`)
> nếu sau này cần dùng cho bài báo riêng.

---

## 0. Cấu hình chung (mọi synth/sim)
- **Device**: Intel Cyclone V `5CSXFC6D6F31C6` (DE10-Standard), speed grade C6.
- **Tool**: Quartus Prime 25.1std Lite, ModelSim/Questa FSE.
- **Clock target (SDC)**: 100 MHz. Fmax đọc ở Slow 1100mV 85C model.
- **Model**: ECG_1DCNN pruned (4,4,8,8), **640 params**, 4 class (AFIB/GSVT/SB/SR).
- **Input**: 2500 INT8 (lead II, 500→250 Hz). **Quant**: power-of-2 round-half-up.
- **nb = {8, 7, 6, 7, 0}** · **w_shift = {6, 7, 6, 7, 7}** · **input_shift = 2**
  (⚠️ ningba re-train: Conv2 nb 6→7 và w_shift 6→7, fc w_shift 8→7 so với Chapman cũ.
  RTL `cnn_controller.v:78` `cfg_nb_of` hard-code đúng 8/7/6/7 — đã verify.)
- **RTL build của luận văn** = `hardware/RTL/` **ROM single-load**: 580 hệ số Conv INT8 +
  32 hệ số FC + bias INT32 nạp bằng `$readmemh` vào ROM, topology Chapman hard-code trong
  `cnn_controller.v`, **không có cổng bus/cfg để ghi trọng số**. Đây là bản duy nhất được
  báo cáo trong bài.

---

## 1. Software — Accuracy chính (ningba)

> **Dataset train/eval chính = ningba** (`data/ningba_processed/ningbo_dataset_clip16.npz`):
> Chapman-Shaoxing mở rộng bằng Chapman-Ningbo, **33,143 record** sau khi loại nửa
> Chapman để tránh leakage. Input clip ±16 để giữ `input_shift=2`. Test = **4,973** record.
> 12-lead resting ECG 10 s, 500 Hz → lead II, downsample 250 Hz (2500 mẫu).
> Nguồn số: `software/python/results/ningba/EVAL_TABLES.md` + `results.json`.

### Table 1 — Pipeline accuracy (ningba, test 4,973)

| Model | Params | Accuracy | F1-macro | macro-AUC | Nguồn |
|---|---:|---:|---:|---:|---|
| Float32 dense | 1,244 | **95.35** | 0.9478 | — | `results.json` (best_epoch 54) |
| Float32 (QAT ckpt, pruned) | 640 | **95.03** | 0.9446 | 0.9938 | `EVAL_TABLES.md` Table A |
| **QAT-INT8 bit-exact (ships trên RTL)** | **640** | **94.27** | **0.9356** | **0.9712** | `EVAL_TABLES.md` Table B |

- **Float32 → INT8: −0.76 pp acc / −0.90 pp F1.** Prediction agreement INT8↔float = **0.9761**.
- INT8 = **bit-exact GAP** (integer floor `sum/4`) = đúng số RTL ROM cho ra.
- Pruning 1,244 → 640 params (**−48.55%**) tốn −0.32 pp acc (95.35 → 95.03 float32).

### Per-class (ningba INT8 bit-exact)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| AFIB | 0.9267 | 0.9062 | 0.9163 | 1,130 |
| GSVT | 0.8698 | 0.9459 | 0.9063 | 869 |
| SB | 0.9756 | 0.9810 | 0.9783 | 1,791 |
| SR | 0.9670 | 0.9172 | 0.9414 | 1,183 |
| **macro** | — | — | **0.9356** | 4,973 |

Float32 đối chiếu: AFIB 0.9294 · GSVT 0.9220 · SB 0.9789 · SR 0.9481 → macro 0.9446.

Plots: `results/ningba/int8_eval/ningba_cm_{float32,int8}.png`, `ningba_roc_{float32,int8}.png`.

> ⚠️ **Số Chapman gốc 94.65% / F1 0.9396 là baseline LỊCH SỬ** (dataset Chapman 10,646,
> re-train 2026-06-08). Dùng cho: Table 4 quant-ablation, on-board JTAG demo, SoTA table
> cột Chapman. **KHÔNG trộn với 94.27% ningba** — hai dataset khác nhau.
> Khi viết: nói rõ dòng nào là Chapman, dòng nào là ningba.

---

## 2. Table 4 — Quantization ablation (Chapman, single-run, seed=42)
Nguồn: `results/ablation_quant/TABLE4_FINAL.md`. **Chạy trên Chapman gốc**, chưa re-run ningba.

| Variant | Scale | Train | Acc % | F1 | DSP rescale |
|---|---|---|---:|---:|---:|
| A1 Float32 baseline | — | — | 94.65 | 0.9402 | — |
| A0 PTQ power-of-2 | 2^nb | none | 94.08 | 0.9338 | **+0** |
| A0' PTQ general | absmax/127 | none | 94.46 | 0.9380 | +4 |
| **A2 QAT power-of-2 (ours)** | 2^nb | fake-quant | **94.37** | **0.9364** | **+0** |
| A3 QAT general | absmax/127 | fake-quant | 94.65 | 0.9398 | +4 |
| A4 QAT power-of-2 floor | 2^nb | fake-quant | 93.99 | 0.9328 | +0 |

Đọc: A2 vs A3 = −0.28 pp acc nhưng **−4 DSP18**. A2 vs A4 floor = **+0.38 pp**
(round-half-up có lợi, 0 DSP cost). → đây là claim C1, KHÔNG claim "phát minh power-of-2".

> ⚠️ Table 4 là **Chapman ablation** (2026-06-02, trước FC-bias). ICDV nên trình bày
> Table 4 = "quantization-scheme ablation on Chapman" (nội bộ nhất quán, so A2↔A3↔A4
> cùng điều kiện), tách khỏi Table 1 = "deployed accuracy on ningba". Không cần re-gen.

### Bit-width: tại sao INT8 không INT4 — `TABLE4_FINAL.md` (bảng phụ 0)
| Variant | W/A | Acc % | F1 | AFIB F1 |
|---|---|---:|---:|---:|
| A2 INT8 (ours) | 8/8 | 94.37 | 0.9364 | — |
| QAT INT4 power-of-2 | 4/4 | 69.95 | 0.660 | — |
| QAT INT4 general (ceiling) | 4/4 | 75.59 | 0.704 | **0.42** |

→ INT4 mất ~19 pp kể cả trần general; AFIB sập → INT8 là sweet-spot (no-ReLU Conv1-3
giữ activation âm dải rộng).

### 5 test-fold robustness — `kfold/kfold_summary.json`
- A0 PTQ p2: acc 94.10 ± 0.61%, F1 0.9345 ± 0.66%
- A0' PTQ general: acc 94.19 ± 0.43%, F1 0.9353 ± 0.48%
- std 0.4–0.9 pp ≥ chênh giữa variant → khác biệt acc nằm trong nhiễu; điểm chắc chắn
  = DSP cost (p2 = 0 vs general = 4).
> Label trung thực: "5 test-fold robustness (re-quant per fold), KHÔNG phải leak-free CV".

---

## 3. Cross-dataset — Georgia (chính) + PTB-XL (phụ)

### 3a. Georgia — FAR-TRANSFER, zero-shot ⭐
Nguồn: `results/georgia/EVAL_TABLES.md` (commit ec81a36).
`georgia_by_class/` **5,459 record**, input clip ±16. **Zero-shot**: model train trên
ningba, KHÔNG fine-tune Georgia. Hệ thu khác (Emory) → far-transfer thật.

| Metric | Float32 | **INT8 (bit-exact RTL)** | Δ |
|---|---:|---:|---:|
| Accuracy | 0.9291 | **0.9300** | **+0.09 pp** |
| F1-macro | 0.9142 | **0.9151** | **+0.09 pp** |
| macro-AUC | 0.9813 | 0.9580 | −0.0233 |
| INT8↔Float32 agreement | — | 0.9749 | — |

Per-class INT8: AFIB P0.8309/R0.8237/F1 0.8273 (692) · GSVT 0.9208/0.9455/0.9329 (1,192)
· SB 0.9537/0.9625/0.9581 (1,521) · SR 0.9513/0.9328/0.9420 (2,054).

**Key finding**: INT8 bám float32 trong **0.1 pp** (thậm chí nhẹ hơn) → **quantization
KHÔNG phải nguồn drop** ngay cả dưới far-transfer. AFIB precision ~0.83 do **composition
shift** (GSVT Georgia bị sinus-tach chi phối), không phải lỗi mapping cũng không phải
quantization. AUC 0.958 >> F1 0.915 → phần lỗi còn lại là **threshold/decision boundary**,
không phải representation.

Plots: `results/georgia/int8_eval/georgia_cm_{float32,int8}.png`, `georgia_roc_*.png`.

> ⚠️ `georgia_by_class` hiện **5,459** record. Memory cũ ghi 5,552 / gốc 5,606 (thăm dò
> GSVT). Số **dùng cho bài = 5,459** (khớp EVAL_TABLES.md đã commit).

### 3b. Ningbo zero-shot (khi model train Chapman-only) — near-transfer
Nguồn: `results/cross_eval/ningbo_c2_report.json`. **Đây là setup CŨ** (train Chapman,
test Ningbo). Sau khi ningba thành dataset train chính thì đây là **in-family** rồi.

C2 zero-shot toàn bộ 33,143 record: acc **0.9257**, macro-F1 0.9175, macro-AUC **0.9868**.

| Class | Precision | Recall | F1 | AUC | Support |
|---|---:|---:|---:|---:|---:|
| AFIB | 0.8971 | 0.8635 | 0.8800 | 0.9830 | 7,533 |
| GSVT | 0.8772 | 0.9112 | 0.8939 | 0.9886 | 5,788 |
| SB | 0.9482 | 0.9839 | 0.9657 | 0.9970 | 11,937 |
| SR | 0.9549 | 0.9074 | 0.9306 | 0.9787 | 7,885 |
| **macro** | **0.9194** | **0.9165** | **0.9175** | **0.9868** | 33,143 |

CM (rows=true): AFIB[6505,471,341,216] GSVT[427,5274,30,57] SB[117,10,11745,65]
SR[202,257,271,7155]. Figures: `results/figures/ningbo_c2_{confusion_matrix,roc}.png`.

**C2 == C6** (INT8 == float32 zero-shot) → quantization drop = **0%**.

Thu thập: Chapman-Shaoxing dùng hệ **GE MUSE** (4.88 µV/LSB), Ningbo dùng thiết bị
**Zhejiang Cachet Jetboom** (1 µV/LSB) → **different-hardware**, không phải "same device";
cùng họ PhysioNet/WFDB → **same-family near-transfer**. z-score xóa gain nên hợp lệ.
Mapping 4-class khớp đúng Chapman `RHYTHM_TO_4CLASS`: AFIB{AFib,AFlutter}
GSVT{ST,SVT,AT,AVNRT,AVRT,SAAWR} SB{SBrad} SR{SR, Sinus-Irregularity SI/SA}.

### 3c. PTB-XL (cross-dataset PHỤ, shift LỚN) — `results/cross_eval/ptbxl_cross_eval.json`
19,952 record, 500→250 Hz, lead II, patient-indep 70/15/15. SR chiếm 84% (imbalanced).

| Mode | Acc | F1-macro |
|---|---:|---:|
| C1 Chapman in-distribution | 0.9446 | 0.9379 |
| C2 zero-shot QAT-INT8 | 0.7714 | 0.6486 |
| C3 linear probe | 0.9263 | 0.7745 |
| C4 full fine-tune | 0.9336 | 0.7940 |
| C5 from-scratch PTB-XL | 0.9263 | 0.7686 |
| C6 float32 zero-shot | 0.7714 | 0.6486 |

**C2 == C6 → quantization drop = 0%.** Zero-shot 77% vì shift lớn (dataset Đức, quy ước
SCP khác). Ba mức shift cho phổ generalization đầy đủ:
**Ningbo (same-family, 92.6%) → Georgia (far, khác hệ thu, 93.0%) → PTB-XL (khác quy ước, 77.1%)**.
Ở CẢ BA mức, INT8 ≈ float32 → claim "power-of-2 INT8 không thêm generalization loss" vững.
> ⚠️ PROJECT.md ghi C3=0.9249 / C4=0.9329; JSON thật = C3 0.9263 / C4 0.9336. **Dùng số JSON.**

---

## 4. Hardware — Production ROM build (8-PE channel-parallel), `ecg_accelerator_top`
Nguồn: `hardware/fpga/output_files/ecg_accelerator_top.{fit.summary,sta.rpt}`
(compile **2026-07-07**, top = `hardware/RTL/` ROM single-load, sau các fix
split-module + S5b-fold + nb-narrow).

| Metric | Giá trị | Nguồn |
|---|---:|---|
| ALM | **2,120 / 41,910 (5%)** | fit.summary |
| DSP | **28 / 112 (25%)** | fit.summary |
| Registers | **3,158** | fit.summary |
| M10K (RAM blocks) | 20 / 553 (4%) | fit.summary |
| Block mem bits | 85,536 / 5,662,720 (2%) | fit.summary |
| Pins | 83 / 499 (17%) | fit.summary |
| **Fmax** | **108.46 MHz** (Slow 1100mV 85C) | sta.rpt |
| **Latency** | **5,216 cy ≈ 52.16 µs @100MHz** | tb_top.v đo `$time`, 21/21 bit-exact |
| Throughput | ~19,200 inf/s @100MHz (≈20,800 @Fmax) | 1/(5216 × 10 ns) |

> ⚠️ **Số cũ trong PAPER_DATA/SOTA_TABLE: ALM 2,201 / Reg 3,177 / Fmax 104.85 MHz**
> (compile 2026-06-15, trước khi RTL/ thành ROM build). **Bản ROM hiện tại nhẹ hơn và
> nhanh hơn: 2,120 ALM / 3,158 Reg / 108.46 MHz.** → DÙNG SỐ MỚI. Cần sửa `SOTA_TABLE.md`
> dòng B1 (đang ghi 2,201 ALM / Fmax 104.85) và `paper/ICDV_draft.md`.
>
> ⚠️ **KHÔNG dùng 137.6 MHz** (memory/commit 32f7a11) — đó là internal reg-to-reg path của
> config bias-fold standalone, không phải Fmax toàn thiết kế.

### 4b. Bit-exact verification (C2) — trạng thái hiện tại
- **Chapman golden, RTL ROM**: `tb_top.v` **21/21 bit-exact** (input + 4 pool + GAP + logits × 3 sample).
- **ningba golden, RTL ROM**: **7/7 bit-exact**, max|diff| = 0 (`tb_bitexact1.v`).
  ⚠️ Bẫy đã gặp: `$readmemh` đọc hex từ **cwd của sim** (`fpga/simulation/questa/`),
  KHÔNG phải `hardware/RTL/` → phải sync `*_w.hex` vào `questa/` sau mỗi lần re-export.
- **Ý nghĩa 2 bộ trọng số**: datapath giữ khớp-bit khi đổi CẢ trọng số LẪN tham số dịch
  `nb` (Conv2 6→7 do hiệu chỉnh của bộ dữ liệu mới) → mạch thực thi đúng **đặc tả lượng tử
  hóa**, không phụ thuộc một bộ số cụ thể. Đây là cách phát biểu defendable cho C2.
- **Unit/integration TB**: `tb_cp_block` 23 PASS + `tb_layer` 8 PASS trên RTL split hiện tại.
- **Full-set golden**: `results/{ningba,georgia}/test_golden/fullset/expected_argmax.hex`
  (toàn test set, so accuracy RTL ≈ SW). Consistency PASS 100%.
- ⚠️ Testbench `tb_cpb_cycle_probe.v` / `tb_top_probe.v` còn tham chiếu `acc_final_r`/
  `acc_final_v` đã bị xóa (commit 369f200 fold S5b) → **sẽ compile lỗi** nếu chạy lại.

---

## 5. On-board DE10-Standard (Phase D)
- JTAG-to-Avalon + System Console (`jtag_top.v`), weight Chapman.
- Kết quả: **94.27% (1004/1065 test)** khớp Python Chapman 94.65%.
- 🔲 **NEEDS-SOURCE**: số 1004/1065 hiện CHỈ trong memory, chưa thấy log `.log/.md` committed.
  Trước khi đưa vào bài: tìm lại JTAG log hoặc **chạy lại on-board** để có log cite được.
- ⚠️ Trùng số ngẫu nhiên: board Chapman = 94.27% và ningba INT8 = 94.27%. **KHÁC NHAU
  hoàn toàn** (dataset khác, test set khác). Viết bài phải phân biệt rõ, đừng gộp.
- Variant Nios V/m RISC-V: sim 3/3 PASS, compile PASS (Quartus 25.1 bỏ Nios II).
- Variant UART: RTL + pin (W15/AK2) + host script READY, chưa chạy (chờ USB-TTL 3.3V).
- Quartus Lite **không có IP HPS Cyclone V** → `soc_top.v`/HPS chỉ để tham khảo.

---

## 6. Năng lượng (PowerPlay)

### 6a. Production trên Cyclone V (DE10-Standard) — ✅ từ report
Nguồn: `hardware/fpga/output_files/ecg_accelerator_top.pow.rpt`
(Quartus 25.1std, **re-run 2026-07-31** trên netlist ROM build + VCD cửa-sổ-suy-luận).
- **Total 536.08 mW / Dyn 110.89 mW / Static 412.12 mW / I/O 13.08 mW**
  → **27.96 µJ total** / 5.78 µJ dyn per inference (× 52.16 µs).
- **Phân rã dynamic theo block** (bán được cho C1):

  | Block | Power | % dynamic |
  |---|---|---|
  | **DSP** | **59.72 mW** | **54 %** |
  | Combinational | 15.40 mW | 14 % |
  | Register | 14.49 mW | 13 % |
  | M10K | 12.00 mW | 11 % |
  | Clock enable | 9.18 mW | 8 % |

  → DSP chiếm **quá nửa** dynamic power. Nối thẳng vào C1: power-of-2 rescale dùng
  **0 DSP** (general-scale cần 4 DSP18) → cắt trực tiếp vào thành phần tốn nhất.
- ⚠️ **Confidence vẫn = `Low`** — phải kèm caveat. Lý do CHÍNH XÁC (từ mục "Signal
  Activities" của report):

  | Loại tín hiệu | Từ VCD | Vectorless (suy đoán) |
  |---|---|---|
  | I/O pin | 83 (**100 %**) | 0 |
  | Register | 2042 (**70.4 %**) | 860 (29.6 %) |
  | Combinational | 129 (**2.5 %**) | 5053 (97.5 %) |

  Register phủ 70 % nhưng combinational chỉ 2.5 % → đây là **giới hạn cấu trúc của
  RTL-level VCD**, không phải lỗi cấu hình: sau khi Fitter gộp/tổng hợp LUT, tên node
  combinational không còn ánh xạ về tên RTL để VCD gán vào. Nâng lên `Medium/High`
  bắt buộc phải **gate-level sim** (netlist `.vo` + SDF) — đã làm thử ở DE0-Nano, rất
  chậm trên Questa FSE free → ngoài phạm vi luận văn.
  → Phát biểu defendable: "**ước lượng**, mạnh ở so sánh tương đối (tĩnh↔động,
  DSP↔phần còn lại), không phải số đo tuyệt đối".
- 💡 **Điểm bán được dù confidence Low**: static 412.12 mW **gấp 3.7× dynamic**
  110.89 mW dù thiết kế chỉ dùng 5 % ALM → die Cyclone V SoC (có ARM cứng) rò rỉ cố
  định. Hệ quả: phần RTL tối ưu được chỉ ~21 % ngân sách công suất →
  **chọn device die nhỏ hiệu quả hơn tối ưu logic**. Luận điểm dựa trên **tỉ lệ**
  tĩnh/động nên vững kể cả khi số tuyệt đối lệch ±30 %.

#### Vì sao số đổi 805.53 → 536.08 mW (phải giải thích được nếu bị hỏi)
Không phải thiết kế thay đổi, mà **VCD đầu vào trước đây sai chế độ**:
- Bản cũ nạp `tb_top.vcd` **từ 0 ns**, gồm cả reset + nhiều test case nạp SRAM qua
  Avalon. Nhưng đó lại là hướng làm **tăng** dynamic sai lệch: hoạt động bus I/O
  (83 pin toggle) bị tính vào, trong khi lõi tính toán phần lớn thời gian đứng yên.
- Bản mới (`ecg_power.vcd`, `testbench/tb_power_vcd.v`) dump **đúng 5219 chu kỳ** từ
  START đến done, tắt dump lúc nạp dữ liệu → mô tả đúng chế độ "accelerator đang làm
  việc", là chế độ cần cho energy/inference.
- Static gần như không đổi (413.84 → 412.12 mW) — hợp lý, static không phụ thuộc
  activity. Toàn bộ chênh lệch nằm ở dynamic (377.72 → 110.89 mW).
- ⚠️ Số cũ **623 mW / 198 mW / 32.5 µJ / "95.6 % toggle"** KHÔNG có trong bất kỳ
  report nào — ghi từ ký ức, đã bỏ. Số **805.53 mW / 42.02 µJ** có thật nhưng đo trên
  netlist cũ (2026-06-22, bản weight-load) + VCD sai chế độ → **thay bằng 536.08 mW**.

## 7. Ablation âm (đã làm, dùng để CỦNG CỐ baseline — không phải thất bại)
| Hướng thử | Kết quả | Kết luận |
|---|---|---|
| **INT4** | 69.95% (p2) / 75.59% (general ceiling) | −19 pp, AFIB sập → INT8 là sàn |
| **Depthwise-separable** | float chạm 94% chỉ khi weight/MAC 1.5–2× baseline; p2-INT8 drop 3.6–9.5 pp | DW chỉ thắng ở model lớn → NO-GO |
| **ANN→SNN naive convert** | cap ~58% (vs 94%) | maxpool + signed + feature-map nhỏ → SNN cần retrain, không free-convert |
| **HR/RR fusion** | cross-dataset +13 pp nhưng phá in-distribution | không ship; SB/SR là clinical rate-boundary |
| **Ping-pong pack 512×16** | M10K 20→12 nhưng fail timing Avalon I/O | không ship |

→ Viết thành 1 đoạn "design decisions justified by negative ablations" — reviewer thích
vì chứng minh baseline không phải chọn bừa.

---

## 8. Việc phải dọn trước khi viết (checklist)
- [ ] 🔴 **Đồng bộ số synth mới vào `SOTA_TABLE.md` B1 + `paper/ICDV_draft.md`**:
      ALM 2,201→**2,120** · Reg 3,177→**3,158** · Fmax 104.85→**108.46 MHz** ·
      power 623 mW→**536.08 mW** · energy 10.3 µJ→**27.96 µJ**.
      (số trung gian 805.53 mW / 42.02 µJ cũng đã lỗi thời — xem §6a)
- [ ] 🔴 **Tách rõ hai dataset trong bài**: ningba (deployed, **94.27%**) vs Chapman
      (ablation + board, **94.65%**). Đừng để người đọc nghĩ ta đổi số tùy tiện.
      Đề xuất: Table 1 = ningba deployed; Table 4 = Chapman quant-ablation (footnote rõ).
- [ ] 🔴 **Rà `SOTA_TABLE.md` + `paper/ICDV_draft.md` + `paper/PROGRESS_REPORT.md`** để loại
      SIMD-20 / weight-load / DE0-Nano theo phạm vi mới (xem khung đầu file).
- [ ] 🟠 SoTA table so sánh với Liu: hiện `SOTA_TABLE.md` dùng 94.65 (Chapman) vs Liu 92.95.
      Vẫn hợp lệ (cùng Chapman) nhưng **đừng claim thắng accuracy** — Δ1.7 pp không kèm
      CI/std, và Liu là fully-mapped khác trade-off. Bán bằng **area/params**.
- [ ] 🟠 On-board 1004/1065: tìm log hoặc chạy lại để cite được.
- [ ] 🟠 SoTA cột "Acc" của Liu đang bị điền = GOPS/W (bug) → sửa `SOTA_TABLE.md`.
- [ ] 🟢 Pareto chart cho SoTA (accuracy vs area/params) — chưa vẽ.
- [x] 🟢 Georgia far-transfer: INT8 93.00% / F1 0.9151, INT8≈float trong 0.1 pp (§3a).
- [x] 🟢 ningba bit-exact RTL ROM: 7/7, max|diff|=0 (§4b).
- [x] 🟢 Thu hẹp phạm vi: đã xóa SIMD/weight-load/Elastic-Pareto/DE0 khỏi file này (§đầu).
