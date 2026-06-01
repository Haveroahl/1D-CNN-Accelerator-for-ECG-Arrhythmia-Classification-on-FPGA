# Q3 Paper Proposal — Reconfigurable INT8 CNN Accelerator for Multi-Dataset ECG Arrhythmia Classification on Intel Cyclone V FPGA

**Tác giả**: Lê Đức (ducle160499@gmail.com)
**Ngày**: 2026-05-24
**Target venue**: Electronics (MDPI) hoặc Sensors (MDPI) — Q3, IF ~2.9–3.4

---

## 1. Tiêu đề đề xuất

> **"Power-of-Two QAT and Cross-Dataset Transferability of an INT8 1D-CNN ECG Classifier on FPGA: A Bit-Exact Methodology and Empirical Study"**

Tiêu đề phụ ngắn (nếu Letter):
> *"Bit-Exact Power-of-Two INT8 1D-CNN for ECG on Cyclone V: Quantization Methodology and Chapman↔MIT-BIH Transfer Study"*

### Pitch novelty (một câu để cover letter / abstract)
> "This paper presents (i) a bit-exact power-of-two QAT methodology for 1D-CNN ECG classification with quantitative ablation against general-scale INT8, and (ii) the first empirical cross-dataset transfer study (Chapman ↔ MIT-BIH) on an FPGA-deployed quantized ECG model, enabled by a lightweight runtime weight-reload mechanism on Intel Cyclone V."

**Lưu ý chiến lược**: Runtime-reconfigurable weight được pitch là **enabling mechanism**, KHÔNG phải đóng góp chính — vì kỹ thuật này phổ biến trong FPGA NN accelerator (DPU, FINN, NVDLA). Đóng góp chính là (a) methodology QAT power-of-2 bit-exact và (c) empirical cross-dataset study.

---

## 2. Motivation & Gap Statement

### 2.1. Vấn đề
- ECG arrhythmia là nguyên nhân đột tử tim hàng đầu; phát hiện sớm bằng wearable / edge device cứu được rất nhiều ca.
- CNN 1D đạt accuracy cao nhưng deploy trên MCU không đủ throughput, deploy trên GPU không phù hợp wearable (power, form-factor).
- FPGA Cyclone V là sweet-spot: low-power, parallelism đủ cho CNN nhỏ, có HPS để host driver.

### 2.2. Gap của các công trình trước (sẽ chứng minh trong Related Work)
| Hạn chế thường thấy trong ECG-FPGA literature | Hướng giải quyết của paper này |
|---|---|
| Dùng general-scale INT8 (cần DSP nhân scale), không phân tích cost-accuracy của power-of-2 shift | ⭐ **Power-of-2 QAT methodology + ablation** vs general-scale (DSP, energy, accuracy) |
| Báo accuracy float vs INT8 nhưng không bit-exact với RTL → "INT8 simulation" và RTL thường lệch | ⭐ **Round-half-up bit-exact pipeline** (21 checkpoint match Python ↔ RTL) |
| Đánh giá chỉ trên 1 dataset, không trả lời được câu hỏi generalization | ⭐ **Empirical cross-dataset study** Chapman ↔ MIT-BIH trên FPGA thực |
| Chỉ báo accuracy + LUT/FF, thiếu Energy/inference (chỉ số quan trọng cho wearable) | **Đo Power qua PowerPlay + Energy/inference µJ** |
| Hard-code weight bitstream → không thể làm transfer learning study trên hardware | Lightweight Avalon-MM weight reload (enabling mechanism, không phải novelty) |

### 2.3. Đóng góp (Contribution list — đưa vào Introduction)

**Đóng góp chính (research contributions):**

1. **C1 — Power-of-Two QAT methodology với phân tích định lượng**: QAT scheme với `nb` và `w_shift` chọn theo `floor(log2(127/abs_max))`, round-half-up rescale. **Ablation systematic** so với general-scale INT8 trên cùng model/dataset: chỉ ra trade-off DSP-count, energy, accuracy drop.
2. **C2 — Bit-exact verification framework**: 21 golden checkpoints (input, 4 pool outputs, GAP, FC logits) match bit-exact giữa Python QAT model và RTL simulation — methodology reproducible cho các CNN INT8 khác.
3. **C3 — Empirical cross-dataset transfer study trên FPGA INT8**: Đánh giá định lượng zero-shot / linear probe / full fine-tune giữa Chapman và MIT-BIH trên cùng IP core thật (không phải software simulation). Trả lời câu hỏi: "INT8 quantized ECG model có generalize giữa datasets không?".

**Đóng góp phụ (engineering / enabling):**

4. **C4 — IP core architecture**: 8-PE SIMD CP-block pipeline (5-stage), 52.16 µs/inference @ 100 MHz deterministic; 580 INT8 weights, ~5K cycles. Reproducible artifact.
5. **C5 — Lightweight runtime weight reload**: Avalon-MM weight RAM (dual-port M10K) cho phép cross-dataset study mà không re-compile. *Đây là enabling mechanism cho C3, không pitch là novelty chính.*

---

## 3. Mục tiêu nghiên cứu (Research Questions)

**RQ chính (cho contributions C1, C3):**
- **RQ1 (methodology)**: Power-of-2 QAT (chỉ shift) đánh đổi gì so với general-scale INT8 (cần multiplier) trên cùng CNN 1D ECG? Cụ thể: accuracy drop (%), DSP saved, energy saved.
- **RQ2 (bit-exact)**: Round-half-up rescale có cần thiết không? So với floor truncation thì accuracy chênh bao nhiêu? Có reproducible 100% giữa Python golden và RTL không?
- **RQ3 (cross-dataset)**: CNN 1D INT8 train trên Chapman, deploy FPGA, đánh giá MIT-BIH:
  - Zero-shot accuracy?
  - Linear probe (freeze conv, retrain FC) gain bao nhiêu?
  - Full fine-tune gain bao nhiêu?
  - Có phải drop accuracy chủ yếu đến từ quantization hay từ distribution shift?

**RQ phụ (cho contributions C4, C5):**
- **RQ4 (architecture cost)**: Lightweight weight RAM + Avalon loader làm tăng resource bao nhiêu, có ảnh hưởng Fmax không?
- **RQ5 (benchmarking)**: Energy/inference của Cyclone V INT8 so với MCU INT8 và GPU edge FP16 ở mức nào?

---

## 4. Phương pháp luận (Methodology)

### 4.1. Model architecture
- 4 Conv1D layers (kênh 4-4-8-8, K=5, pad=2) + MaxPool /5 sau mỗi layer + GAP + FC(8→4) + Argmax.
- Input 2500 INT8 (5s @ 500 Hz, lead II).
- ReLU **chỉ** sau Conv4 (preserve negative ECG features trong Conv1-3).

### 4.2. Quantization — Power-of-2 QAT (⭐ contribution C1)
- Per-layer shift bits `nb = {8,6,6,7,0}`, weight shift `{6,6,6,7,8}` chọn theo `floor(log2(127/abs_max))`.
- Rescale formula: `out = clamp(round_half_up(acc / 2^nb), -127, 127)` với `round_half_up(x) = (x + 2^(nb-1)) >> nb`.
- Bias: `bias_scaled = round(b_float × 2^nb)`, INT32 little-endian.
- Hardware cost: chỉ cần barrel shifter + adder, **0 DSP cho rescale** (so với general-scale cần 1 multiplier mỗi rescale).

**Ablation comparison (sẽ implement trong Phase A')**:

| Variant | Scale | Rescale op | DSP/rescale | Expected accuracy |
|---|---|---|---|---|
| Float32 baseline | — | — | — | upper bound |
| **Power-of-2 QAT (ours)** | `2^nb` | shift + add | 0 | ~94.65% |
| General-scale INT8 QAT | `s ∈ ℝ` | mul + shift | 1 | reference |
| Power-of-2 + floor (no round) | `2^nb` | shift | 0 | đo drop |

→ Bảng này → Table 3 trong paper, định lượng cost-benefit của power-of-2.

### 4.3. Bit-exact pipeline (⭐ contribution C2)
- Python golden tuân thủ **đúng** sequence như RTL: `acc_int32 → +bias_scaled → +2^(nb-1) → >>nb → clamp[-127,127] → ReLU(nếu có) → MaxPool`.
- GAP: `floor(sum/4) = sum >> 2` (integer, không phải float average).
- FC: `nb=0`, raw INT32 logits → argmax.
- 21 checkpoints per sample: input + 4 pool outputs + GAP + 4 logits + argmax — match 100% với RTL.

### 4.4. Hardware architecture — 2 versions
- **V1 (baseline, đã done)**: Weight ROM với `$readmemh`, embed bitstream — để đo resource baseline.
- **V2 (enabling mechanism cho C3)**: Weight RAM dual-port M10K, Avalon-MM loader, 12-bit address space.

V2 chỉ là phương tiện để chạy cross-dataset transfer trên cùng bitstream — không pitch là contribution chính. Tuy vậy vẫn báo cáo overhead (RQ4) trong Results để minh bạch.

### 4.5. Verification flow
```
Python QAT-INT8 model
    ├→ generate_golden.py → 21 .mem checkpoints (input, pool1-4, gap, logits, argmax)
    │
RTL simulation (ModelSim)
    ├→ tb_top.v load .mem → compare bit-exact each checkpoint
    │
On-board (DE10-Standard)
    └→ HPS driver load weight + input → result via Avalon → match Python
```

### 4.6. Datasets
| Dataset | Use | Split |
|---|---|---|
| **Chapman ECG** (12-lead, 10646 records) | Train + primary eval | Patient-independent 70/15/15 |
| **MIT-BIH Arrhythmia** (47 records, 360Hz → 500Hz) | Cross-dataset eval | AAMI class mapping |

### 4.7. Ablation studies — phục vụ trực tiếp C1 và C3

**Nhóm A (cho C1 — methodology):**
| Ablation | Mục đích | Metric |
|---|---|---|
| A1. Float32 baseline | Upper bound | Acc, F1 |
| A2. **Power-of-2 QAT** (ours) | Đề xuất chính | Acc, F1, DSP, energy |
| A3. General-scale INT8 QAT | So sánh trực tiếp | Acc, F1, DSP, energy |
| A4. Power-of-2 + floor (no round-half-up) | Chứng minh round-half-up cần thiết | Acc drop |
| A5. Pruned vs dense | Footprint trade-off | Params, Acc |

**Nhóm C (cho C3 — cross-dataset):**
| Ablation | Mục đích | Metric |
|---|---|---|
| C1. Chapman→Chapman (in-distribution) | Baseline | Acc, F1, per-class |
| C2. Chapman→MIT-BIH zero-shot | Distribution shift cost | Acc drop |
| C3. + Linear probe (retrain FC only) | Cost-benefit nhẹ | Acc gain, params updated |
| C4. + Full fine-tune | Upper bound transfer | Acc gain |
| C5. MIT-BIH from-scratch | Reference | Acc |
| C6. INT8 vs Float32 cross-dataset | Quantization hay distribution gây drop? | Decomposition |

**Nhóm phụ (cho C5):**
| Ablation | Mục đích |
|---|---|
| E1. V1 ROM vs V2 RAM | Resource + Fmax overhead |

---

## 5. Kế hoạch thực nghiệm (Experimental Plan)

### Phase A' — QAT ablation (Tuần 1, ngày 1-2) ⭐ cho C1
- Implement 3 quantization variants trên cùng pruned model (4-4-8-8):
  - A2: Power-of-2 QAT round-half-up (đã có) — `software/python/quantization/qat_int8.py`.
  - A3: General-scale INT8 QAT — viết `qat_int8_general.py`, scale `s = abs_max/127` (float), simulate hardware rescale `round(acc * s_out / (s_in * s_w))`.
  - A4: Power-of-2 + floor — fork A2, đổi `(acc + 2^(nb-1)) >> nb` thành `acc >> nb`.
- Train 5-fold patient-independent split.
- Đo: accuracy, F1-macro, per-class F1, DSP estimate (count multiplier trong rescale chain).
- **Output**: `results/ablation_quant/{variant}_kfold.json`.

### Phase A — Cross-dataset eval (Tuần 1, ngày 3-4) ⭐ cho C3
- Download MIT-BIH (PhysioNet wfdb), preprocess: resample 360→500Hz, lead MLII, segment 5s window (2500 samples).
- Map class AAMI ↔ Chapman 4-class (N→SR, S→GSVT, V→GSVT, F/Q→drop) — document rationale rõ trong paper.
- Patient-independent split MIT-BIH 70/15/15.
- 5 modes (đối ứng ablation nhóm C):
  - C2: Zero-shot (Chapman QAT-INT8 weight → predict MIT-BIH).
  - C3: Linear probe (freeze conv, retrain FC, INT8 QAT lại FC).
  - C4: Full fine-tune (unfreeze all, INT8 QAT toàn bộ).
  - C5: From-scratch MIT-BIH.
  - C6: Float32 zero-shot + fine-tune để decompose quantization vs distribution shift.
- **Output**: `results/cross_eval/{mode}_metrics.json` + confusion matrices + per-class F1.

### Phase B — Lightweight weight RAM (Tuần 1, ngày 5-6) — enabling mechanism cho C3
- Refactor `cp_engine.v`: thay weight FF array bằng `weight_ram` interface.
- Tạo `weight_ram.v`: dual-port M10K, write từ Avalon, read combinational/1-cy.
- Mở rộng `avalon_slave.v`: 5-bit → 12-bit address, address map:
  - `0x000-0x07F`: weight + bias + FC (~580 INT8 words packed)
  - `0x080-0x09C`: input ECG buffer
  - `0x0A0`: control/status
- HPS C driver: `load_weights(path)`, `load_ecg(buf)`, `run_inference()`, `read_result()`.
- Regression: 21/21 bit-exact phải PASS với weight load via Avalon.

### Phase C — Synthesis & power measurement (Tuần 2, ngày 1-3)
- Quartus Compile (device 5CSXFC6D6F31C6, SDC 100 MHz).
- TimeQuest: report Fmax thực, WNS.
- **PowerPlay**: dùng `.vcd` từ simulation full inference làm activity input → dynamic + static power.
- Energy/inference = Power × 52.16 µs.
- Resource report: ALM, M10K, MLAB, DSP18, FF — so sánh V1 vs V2.

### Phase D — On-board validation (Tuần 2, ngày 4-5)
- Program `.sof` vào DE10-Standard.
- HPS driver chạy test set Chapman → đếm match với Python (target: 100% match RTL sim, ~94.65% accuracy).
- Load weight MIT-BIH (Phase A fine-tune) → đo accuracy on-board → match Phase A.
- Đo latency thực bằng performance counter (ARM A9 cycle counter).

### Phase E — Benchmark & SoTA comparison (Tuần 2, ngày 6-7)
- Lập bảng so sánh ≥ 6 paper ECG-FPGA (xem Section 6.2 dưới).
- Vẽ Pareto front: accuracy vs energy, accuracy vs latency, params vs accuracy.
- So sánh head-to-head với 1-2 paper dùng general-scale INT8 (cho RQ1).

### Phase F — Writing & submission (Tuần 3)
- Draft → internal review → submit.

---

## 6. Đo lường & Báo cáo (Metrics & Reporting)

### 6.1. Bắt buộc trong paper
| Metric | Đơn vị | Status hiện tại |
|---|---|---|
| Accuracy (test set Chapman) | % | ✅ 94.65% |
| F1-macro, per-class F1 | — | ✅ 0.94 |
| Confusion matrix | — | 🔲 Cần re-generate |
| ROC / AUC | — | 🔲 Cần thêm |
| K-fold (5-fold) mean ± std | — | 🔲 |
| Patient-independent split | — | 🔲 Verify |
| Inference latency | µs / cycles | ✅ 52.16 µs / 5216 cy |
| Throughput | inf/s | ✅ ~19,200 |
| Fmax | MHz | 🔲 Sau synthesis |
| Resource: ALM/M10K/DSP18/MLAB/FF | đếm + % | 🔲 Sau synthesis |
| Dynamic power | mW | 🔲 PowerPlay |
| Static power | mW | 🔲 PowerPlay |
| Energy/inference | µJ | 🔲 Tính sau power |
| Cross-dataset accuracy (5 modes) | % | 🔲 Phase A |
| Quantization variants (A2-A4) accuracy | % | 🔲 Phase A' |
| DSP count per variant | # | 🔲 Phase A' + synth |
| Decomposition: quant drop vs distribution drop | % | 🔲 Phase A C6 |

### 6.2. Bảng so sánh SoTA (template, cần điền sau khi search literature)
| Ref | Year | Platform | Model | Quant | Dataset | Acc | Latency | Power | Energy/inf |
|---|---|---|---|---|---|---|---|---|---|
| [1] | 2022 | Zynq-7020 | 2D-CNN | INT16 | MIT-BIH | 97.x% | ~ms | ~W | ~mJ |
| [2] | 2023 | Cyclone V | LSTM | FP16 | MIT-BIH | ... | ... | ... | ... |
| [3] | 2024 | Cortex-M4 | TinyCNN | INT8 | Chapman | ... | ... | ... | ... |
| [4] | 2023 | Edge GPU | ResNet | FP16 | MIT-BIH | ... | ... | ... | ... |
| [5] | 2024 | Zynq UltraScale | 1D-CNN | INT8 | PTB-XL | ... | ... | ... | ... |
| [6] | 2025 | ASIC | CNN | INT8 | MIT-BIH | ... | ... | ... | ... |
| **Ours** | 2026 | **Cyclone V** | **1D-CNN** | **INT8 P2-QAT** | **Chapman+MIT** | **94.65%** | **52 µs** | **TBD** | **TBD** |

→ **Action**: 1 ngày search Google Scholar + IEEE Xplore, từ khoá: `"ECG" "FPGA" "CNN"`, `"arrhythmia" "quantization" "FPGA"`, filter 2021-2026.

---

## 7. Cấu trúc paper (8-12 trang, single-column MDPI hoặc double-column IEEE)

```
1. Introduction                            ~1.5 page
   1.1 Background & motivation
   1.2 Limitations of existing ECG-FPGA work
   1.3 Contributions (3 main + 2 enabling — Section 2.3)
   1.4 Paper organization

2. Related Work                            ~1 page
   2.1 ECG CNN classifiers (software)
   2.2 INT8 quantization schemes (power-of-2 vs general-scale)
   2.3 FPGA accelerators for ECG / 1D-CNN
   2.4 Cross-dataset / transfer learning trên embedded NN
   2.5 Gap analysis (Table 1)

3. CNN Model and Power-of-2 QAT Methodology  ~2 page  ⭐ contribution C1+C2
   3.1 Network topology (Fig. 1)
   3.2 Power-of-2 QAT formulation (eq. 1-4)
       - shift_bits selection rule
       - weight + activation + bias quantization
   3.3 Round-half-up rescale (eq. 5) — bit-exact với hardware shift
   3.4 Comparison with general-scale INT8 (Table 2 — hardware cost analysis)
   3.5 Bit-exact verification framework (Fig. 2 — 21 checkpoints)

4. Hardware Architecture                   ~2.5 page
   4.1 Top-level (Fig. 3)
   4.2 CP-block 5-stage pipeline (Fig. 4 + timing diagram)
   4.3 CP-engine: 8 PE + SRW + weight store
   4.4 Controller FSM (Fig. 5)
   4.5 GAP/FC/Argmax datapath
   4.6 Lightweight weight RAM + Avalon-MM loader (Fig. 6) — enabling C3
   4.7 Memory map (Table 3)

5. Implementation                          ~1 page
   5.1 Target device & tools
   5.2 Verification flow (21 checkpoint match)
   5.3 HPS driver for weight reload + inference

6. Results                                 ~3 page  ⭐ core data
   6.1 Quantization ablation (RQ1, RQ2) — Table 4 ⭐
       - A1 Float32 / A2 Power-of-2 / A3 General-scale / A4 Floor
       - Accuracy, F1, DSP, energy
   6.2 In-distribution accuracy on Chapman (5-fold) — Table 5
       - Per-class F1, confusion matrix, ROC/AUC
   6.3 Cross-dataset transfer study (RQ3) — Table 6 + Fig. 7 ⭐
       - 5 modes (C2-C6) × accuracy/F1
       - Decomposition: quant drop vs distribution shift
       - Confusion matrices MIT-BIH
   6.4 Hardware metrics (RQ4)
       - Resource & Fmax (V1 vs V2) — Table 7
       - Power & energy — Table 8
   6.5 SoTA comparison (RQ5) — Table 9 + Fig. 8 Pareto

7. Discussion                              ~0.7 page
   7.1 Why power-of-2 QAT loses < X% accuracy despite coarser scale
   7.2 What does MIT-BIH transfer tell us about ECG CNN generalization?
   7.3 Reconfiguration overhead — bounded and acceptable
   7.4 Limitations (single-lead, fixed topology, 2 datasets)

8. Conclusion & Future Work                ~0.3 page

References (40-60 entries)
```

### Figures cần vẽ
- **Fig. 1**: Network topology với tensor shape mỗi stage.
- **Fig. 2**: Bit-exact pipeline Python ↔ RTL (round-half-up flow) ⭐ C2.
- **Fig. 3**: Top-level block diagram.
- **Fig. 4**: CP-block 5-stage pipeline timing diagram.
- **Fig. 5**: Controller FSM state chart.
- **Fig. 6**: Weight RAM + Avalon loader (enabling C3).
- **Fig. 7**: Cross-dataset transfer results — bar/line chart 5 modes ⭐ C3.
- **Fig. 8**: Pareto: accuracy vs energy/inference cho ECG-FPGA literature.

### Tables
- **Table 1**: Gap analysis vs related work.
- **Table 2**: ⭐ Hardware cost — Power-of-2 vs General-scale rescale (DSP, gates, energy).
- **Table 3**: Avalon memory map.
- **Table 4**: ⭐ Quantization ablation (A1-A4) — accuracy, F1, DSP, energy.
- **Table 5**: Chapman 5-fold accuracy (mean ± std), per-class F1.
- **Table 6**: ⭐ Cross-dataset (C2-C6) — accuracy, F1, decomposition.
- **Table 7**: Resource & Fmax (V1 vs V2).
- **Table 8**: Power breakdown.
- **Table 9**: SoTA comparison.

---

## 8. Reproducibility & Artifact

- **GitHub public repo**: software/ + hardware/ + scripts để regenerate golden + sim.
- **Zenodo DOI** cho code release (MDPI khuyến khích).
- **README**: bước-bước reproduce từ dataset → golden → sim → synthesis.
- **Pretrained weights** + `flat_weights.hex` shipped.
- **Docker / conda env** cho software side.

→ Reproducibility là **bonus điểm lớn** cho Q3, đặc biệt MDPI.

---

## 9. Timeline chi tiết (3 tuần)

| Tuần | Ngày | Việc | Output |
|---|---|---|---|
| **W1** | 1-2 | Phase A' — QAT ablation A2/A3/A4 + k-fold | ablation_quant/*.json |
| | 3-4 | Phase A — MIT-BIH 5 modes | cross_eval/*.json + confusion |
| | 5-6 | Phase B — weight RAM RTL + Avalon | RTL pass 21/21 |
| | 7 | Buffer / catch-up | — |
| **W2** | 1-3 | Phase C — synthesis + PowerPlay | Fmax, resource, power |
| | 4-5 | Phase D — on-board DE10 | accuracy match |
| | 6-7 | Phase E — SoTA bảng + Pareto | Table 8, Fig. 8 |
| **W3** | 1-3 | Draft paper sections 1-4 | First half |
| | 4-5 | Draft sections 5-8 + figures | Full draft |
| | 6 | Internal review + revise | v2 |
| | 7 | Submit | Submission |

---

## 10. Rủi ro & giảm thiểu (Risks)

| Rủi ro | Khả năng | Mức ảnh hưởng | Giảm thiểu |
|---|---|---|---|
| Phase B (weight RAM) phá vỡ bit-exact 21/21 | Medium | High | Refactor incremental, test sau mỗi sub-step; giữ V1 làm fallback |
| Fmax sau Phase B < 100 MHz | Low | Medium | M10K read 1-cy match ROM hiện tại; SDC 100MHz có ~3ns slack |
| MIT-BIH zero-shot accuracy quá thấp | Medium | Low | Story vẫn ổn — chính là motivation cho reconfig + fine-tune |
| PowerPlay không có activity file đủ realistic | Medium | Medium | Dùng `.vcd` từ full-inference sim; nếu thiếu, dùng default toggle 12.5% |
| Reviewer hỏi multi-lead / streaming | High | Low | Section 7.3 thừa nhận limitation; future work |
| Reviewer đòi thêm dataset (PTB-XL, CPSC) | Medium | Medium | Mention trong future work; PTB-XL cần Phase C (topology programmable) — out-of-scope |
| Tạp chí reject vì topology nhỏ | Low | High | Argue: target wearable, params tối thiểu là feature; có ablation pruned vs dense |

---

## 11. Lựa chọn tạp chí — phân tích chi tiết

| Journal | Pros | Cons | Khuyến nghị |
|---|---|---|---|
| **Electronics (MDPI)** | Fit nhất (FPGA+embedded+biomed), OA, decision 3-5 tuần, IF ~2.9 | APC ~2000 CHF | ⭐ **Lựa chọn chính** |
| **Sensors (MDPI)** | Biomedical+sensor strong, IF ~3.4 | Cạnh tranh hơn, APC tương tự | ⭐ Backup |
| **IEEE Access** | Multi-disciplinary, IF ~3.4, dễ accept | Reputation thấp hơn | Backup |
| **Microprocessors and Microsystems (Elsevier)** | Embedded HW pure | Decision chậm 2-4 tháng, không OA | Nếu không gấp |
| **IEEE Embedded Systems Letters** | Letter ngắn 4p, decision nhanh | Phải cắt rất nhiều | Nếu muốn fast publish |

**Quyết định đề xuất**: Electronics MDPI, special issue về "AI Accelerators on FPGA" hoặc "Biomedical Signal Processing on Edge Devices" (check current open SIs).

---

## 12. Action checklist tóm tắt

### Bắt đầu ngay (W1 D1)
- [ ] Viết `software/python/quantization/qat_int8_general.py` (general-scale variant, A3).
- [ ] Fork A2 → A4 (floor version) bằng flag `--rescale-mode floor`.
- [ ] Setup 5-fold patient-independent split runner cho Chapman.
- [ ] Download MIT-BIH (wfdb) + preprocess script.
- [ ] Tạo nhánh git `feature/weight-ram` cho Phase B.

### Trước khi viết paper (deliverables tối thiểu cho contributions)
- **Cho C1**: Table 4 ablation A1-A4 đầy đủ với 5-fold mean±std.
- **Cho C2**: 21/21 bit-exact PASS cho cả A2 và sau khi load weight via Avalon (V2).
- **Cho C3**: Table 6 với 5 modes + confusion matrix + decomposition quant/distribution.
- **Cho C4**: Fmax, power, energy thực đo từ Quartus + PowerPlay.
- **Cho C5**: V1 vs V2 resource & Fmax overhead.
- **Common**: Bảng SoTA 6-8 papers, k-fold, confusion matrix, ROC/AUC.

### Trước khi submit
- [ ] GitHub repo public + Zenodo DOI.
- [ ] Cover letter highlight novelty (C1-C5).
- [ ] Check journal scope & special issue match.
- [ ] Proofread (Grammarly / native speaker).

---

## 13. Kết luận của proposal

Sau khi điều chỉnh theo Hướng 3 (a)+(c), novelty của paper tập trung vào hai trục **defendable trước reviewer Q3**:

- **(a) Power-of-2 QAT methodology** — không chỉ "we use power-of-2" mà có **ablation định lượng** vs general-scale INT8 và floor variant, chứng minh trade-off DSP/energy/accuracy.
- **(c) Empirical cross-dataset transfer study** trên FPGA INT8 — là **câu hỏi research thật**, không phải engineering claim. Decomposition quant-vs-distribution là điểm reviewer sẽ khen.

Runtime weight reload (Phase B) được pitch lại là **enabling mechanism**, minh bạch trong Section 4.6 và 6.4. Reviewer sẽ không thể critique "this is standard DPU technique" vì paper không claim nó là contribution chính.

Project hiện đã có nền tảng kỹ thuật vững (RTL bit-exact, QAT power-of-2, 94.65% accuracy). Khoảng cách tới Q3 là:
1. **Ablation quant variants** (A3 general-scale, A4 floor) — Phase A'.
2. **Cross-dataset systematic** (5 modes + decomposition) — Phase A.
3. **Số đo thật** (power, energy, Fmax, resource sau synthesis) — Phase C.
4. **Lightweight weight reload** để enable C3 trên hardware — Phase B.
5. **Bảng SoTA + Pareto** — Phase E.
6. **Viết bài + reproducibility artifact** — Phase F.

Tổng effort ước tính: **3 tuần làm việc tập trung**, deliverable cuối là 1 paper 8-12 trang nộp Electronics MDPI hoặc Sensors MDPI.
