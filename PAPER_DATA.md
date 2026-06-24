# PAPER_DATA.md — Số liệu thật, gom 1 chỗ để viết ICDV

> Nguồn duy nhất khi viết. MỌI số ở đây đã verify từ report/json/golden gốc (2026-06-15),
> KHÔNG chép từ doc cũ. Chỗ nào doc cũ lệch → ghi chú "⚠️ doc cũ ghi X, dùng Y".
> Nếu một số chưa verify được từ file → đánh dấu 🔲 NEEDS-SOURCE, phải chạy/tìm lại trước khi đưa vào bài.

---

## 0. Cấu hình chung (mọi synth/sim)
- **Device**: Intel Cyclone V `5CSXFC6D6F31C6` (DE10-Standard), speed grade C6.
- **Tool**: Quartus Prime 25.1std Lite, ModelSim/Questa FSE.
- **Clock target (SDC)**: 100 MHz. Fmax đọc ở Slow 1100mV 85C model.
- **Model**: ECG_1DCNN pruned (4,4,8,8), **654 params**, 4 class (AFIB/GSVT/SB/SR).
- **Input**: 2500 INT8 (lead II). **Quant**: power-of-2 round-half-up, nb={8,6,6,7,0},
  w_shift={6,6,6,7,8}, input_shift=2.

---

## 1. Software — Accuracy & Quantization (Chapman, patient-indep 70/15/15, seed=42)

### Table 4 — Quantization ablation (single-run) — `results/ablation_quant/TABLE4_FINAL.md`
| Variant | Scale | Train | Acc % | F1 | DSP rescale |
|---|---|---|---:|---:|---:|
| A1 Float32 baseline | — | — | 94.65 | 0.9402 | — |
| A0 PTQ power-of-2 | 2^nb | none | 94.08 | 0.9338 | **+0** |
| A0' PTQ general | absmax/127 | none | 94.46 | 0.9380 | +4 |
| **A2 QAT power-of-2 (ours)** | 2^nb | fake-quant | **94.37** | **0.9364** | **+0** |
| A3 QAT general | absmax/127 | fake-quant | 94.65 | 0.9398 | +4 |
| A4 QAT power-of-2 floor | 2^nb | fake-quant | 93.99 | 0.9328 | +0 |

Đọc: A2 vs A3 = −0.28% acc nhưng **−4 DSP18**. A2 vs A4 floor = **+0.38%** (round-half-up có lợi).

> ⚠️ **Số accuracy bit-exact "chốt"**: PROJECT.md/memory ghi **94.65% / F1 0.9396 / AFIB 0.9266**
> (re-train có FC bias 2026-06-08). Table 4 ghi A2=94.37 (bản 2026-06-02, trước FC bias).
> → **Quyết trước khi viết**: dùng MỘT con số nhất quán. Khuyến nghị **94.65% / F1 0.9396**
> (bản mới nhất, là số đã dùng cho golden RTL). Nếu dùng 94.65 thì Table 4 A2 cũng phải re-gen
> hoặc chú thích rõ "FC-bias version". 🔲 CẦN CHỐT.

### Bit-width: tại sao INT8 không INT4 — `TABLE4_FINAL.md` (bảng phụ 0)
| Variant | W/A | Acc % | F1 | AFIB F1 |
|---|---|---:|---:|---:|
| A2 INT8 (ours) | 8/8 | 94.37 | 0.9364 | — |
| QAT INT4 power-of-2 | 4/4 | 69.95 | 0.660 | — |
| QAT INT4 general (ceiling) | 4/4 | 75.59 | 0.704 | **0.42** |

→ INT4 mất ~19% kể cả trần general; AFIB sập → INT8 là sweet-spot (no-ReLU Conv1-3 giữ activation âm dải rộng).

### 5 test-fold robustness — `kfold/kfold_summary.json`
- A0 PTQ p2: acc 94.10±0.61%, F1 0.9345±0.66%
- A0' PTQ general: acc 94.19±0.43%, F1 0.9353±0.48%
- (các variant khác trong json) — std 0.4–0.9% ≥ chênh giữa variant → khác biệt acc nằm trong nhiễu;
  điểm chắc chắn = DSP cost (p2=0 vs general=4).
> Label trung thực: "5 test-fold robustness (re-quant per fold), KHÔNG phải leak-free CV".

### Confusion matrix / ROC — `results/figures/`
- Chapman CM + ROC: macro-AUC **0.967**. 🔲 xác nhận file PNG còn tồn tại trước khi cite.

---

## 2. Cross-dataset — Ningbo (chính) + PTB-XL (phụ)

### 2a. Ningbo (CROSS-DATASET CHÍNH) — `results/cross_eval/ningbo_c2_report.json`, `ningbo_cross_eval.json`
Ningbo (Chapman-Ningbo/Shaoxing), cùng họ Chapman (SNOMED-CT, WFDB 12-lead 500Hz/10s).
**⚠️ `data/ningba` GỘP Chapman-Shaoxing (JS00001–JS10646 = tập train của model) + Ningbo (JS10647+).** Đã loại nửa Chapman để tránh leakage → **33,143 record** Ningbo thuần.
Mapping 4-class khớp đúng Chapman `RHYTHM_TO_4CLASS` (Zheng 2020): AFIB{AFib,AFlutter} GSVT{ST,SVT,AT,AVNRT,AVRT,SAAWR} SB{SBrad} SR{SR, Sinus-Irregularity}.
500→250Hz, lead II, record-level split 70/15/15 (Ningbo .hea không có patient_id).

**C2 zero-shot — báo cáo đầy đủ (toàn bộ 33,143 record):** acc **0.9257**, macro-F1 0.9175, macro-AUC 0.9868, weighted-AUC 0.9880.

| Class | Precision | Recall | F1 | AUC | Support |
|---|---:|---:|---:|---:|---:|
| AFIB | 0.8971 | 0.8635 | 0.8800 | 0.9830 | 7,533 |
| GSVT | 0.8772 | 0.9112 | 0.8939 | 0.9886 | 5,788 |
| SB   | 0.9482 | 0.9839 | 0.9657 | 0.9970 | 11,937 |
| SR   | 0.9549 | 0.9074 | 0.9306 | 0.9787 | 7,885 |
| **macro** | **0.9194** | **0.9165** | **0.9175** | **0.9868** | 33,143 |
| **weighted** | **0.9258** | **0.9257** | **0.9253** | **0.9880** | — |

Confusion matrix (rows=true, cols=pred): AFIB[6505,471,341,216] GSVT[427,5274,30,57] SB[117,10,11745,65] SR[202,257,271,7155].
Figures: `results/figures/ningbo_c2_confusion_matrix.png`, `ningbo_c2_roc.png`.

**Key finding**: C2 (INT8 zero-shot) == C6 (float32 zero-shot) → quantization drop = **0%**; toàn bộ drop là distribution shift. Vì Ningbo cùng họ Chapman, shift nhỏ → zero-shot giữ 92.6% (mọi class AUC ≥ 0.98).

### 2b. PTB-XL (cross-dataset PHỤ) — `results/cross_eval/ptbxl_cross_eval.json`
PTB-XL: 19,952 records, 500→250Hz, lead II, patient-indep 70/15/15. SR chiếm 84% (imbalanced).

| Mode | Acc | F1-macro |
|---|---:|---:|
| C1 Chapman in-distribution | 0.9446 | 0.9379 |
| C2 zero-shot QAT-INT8 | 0.7714 | 0.6486 |
| C3 linear probe | 0.9263 | 0.7745 |
| C4 full fine-tune | 0.9336 | 0.7940 |
| C5 from-scratch PTB-XL | 0.9263 | 0.7686 |
| C6 float32 zero-shot | 0.7714 | 0.6486 |

**Key finding (PTB-XL)**: C2 == C6 → quantization drop = **0%**. Zero-shot thấp (77%) vì distribution shift lớn (dataset Đức, quy ước SCP khác). Cặp Ningbo (shift nhỏ, 92.6%) + PTB-XL (shift lớn, 77%) cho 2 đầu phổ generalization.
> ⚠️ PROJECT.md ghi C3=0.9249/C4=0.9329; JSON thật = C3 0.9263 / C4 0.9336. **Dùng số JSON.**

---

## 3. Hardware — Production (8-PE channel-parallel), `ecg_accelerator_top`
Nguồn: `hardware/fpga/output_files/ecg_accelerator_top.{fit.summary,sta.rpt,fit.rpt}` (compile 2026-06-15).

| Metric | Giá trị | Nguồn |
|---|---:|---|
| ALM | **2,201 / 41,910 (5%)** | fit.summary |
| DSP | **28 / 112 (25%)** | fit.summary |
| Registers | 3,177 | fit.summary |
| M10K (RAM blocks) | 20 / 553 (4%) | fit.summary |
| Block mem bits | 85,536 | fit.summary |
| **Fmax** | **104.85 MHz** (85C) | sta.rpt |
| **Latency** | **5,216 cy ≈ 52.16 µs @100MHz** | tb_top.v đo $time, 21/21 bit-exact |
| Throughput | ~19,200 inf/s | 1/(5216×10ns) |

Per-entity ALM: cp_engine 1,739 (24 DSP, 8×cp_block ~145) · gap_fc 344 · controller 74 · pingpong 25 · input_sram 0 (M10K).

> ⚠️ **Fmax**: memory/commit 32f7a11 ghi **137.6 MHz** (config bias-fold standalone khác).
> Compile production hiện tại = **104.85 MHz**. → DÙNG 104.85, nhất quán toàn bài. KHÔNG trộn 137.6.

### 3b. Production + Phase B01 weight reload (runtime-loadable weights)
Nguồn: `hardware/fpga/output_files/ecg_accelerator_top.{fit.summary,sta.rpt}` (compile 2026-06-16, branch `feature/weight-ram`, clk pinned PIN_AF14).
Conv weights chuyển từ FF-ROM ($readmemh, bake bitstream) → 8 per-oc M10K nạp runtime qua Avalon (+ bias/FC write port). Enabling mechanism cho C5/C3 (cùng .sof chạy Chapman hoặc PTB-XL weight).

| Metric | Production (baseline) | + Phase B01 weight-RAM | Δ |
|---|---:|---:|---:|
| ALM | 2,201 (5%) | **2,820 (7%)** | +619 (+1.5% device) |
| DSP | 28 (25%) | **28 (25%)** | 0 |
| Registers | 3,177 | **4,852** | +1,675 |
| M10K | 20 (4%) | **28 (5%)** | +8 (8 per-oc weight RAM) |
| Fmax (standalone) | 104.85 MHz | **108.94 MHz** | +4.1 MHz |
| Latency | 5,216 cy (52.16 µs) | **5,216 cy (52.16 µs)** | 0 (bit-exact) |
| Throughput | ~19,200 inf/s | ~19,200 inf/s | 0 |

> **+619 ALM KHÔNG do M10K** (altsyncram = 0 ALM, nằm trong M10K block). Overhead đến từ logic
> *runtime-loadable*: read-address adder (`layer_base+ic`) cho 8 RAM, 8-way write decode, 40-bit
> lo/hi assembly trong avalon_slave, + register cho các write port. Đây là chi phí cố hữu của
> tính nạp-được, không phải của bộ nhớ. Latency/accuracy **không đổi** (M10K sync-read thay đúng
> 1-1 stage `w_packed`, pipeline alignment giữ nguyên → tb_top 21/21 bit-exact, max|diff|=0 LSB).
>
> **Fmax board thật (jtag_top, Avalon internal, PLL 100MHz)**: setup slack **+2.202 ns @100MHz**,
> 0 violation mọi corner (Fmax core ~125 MHz). Standalone 108.94 MHz bao gồm I/O margin
> (`set_output_delay 1.5ns` trên `avs_*`) → số bảo thủ. **−5.767ns lần compile trước là artifact**
> clk chưa gán pin (route qua CLKENA +6.6ns); gán PIN_AF14 → hết, số ổn định lặp lại được.

---

## 4. Hardware — SIMD-20 (position-parallel), `ecg_simd_top` (worktree feature/simd-spec)
Nguồn: `D:/Thesis101-simd/hardware/fpga/simd_synth/ecg_simd_top.{fit.summary,sta.rpt,fit.rpt}` (2026-06-15, sau fix M10K).

| Metric | Giá trị | Nguồn |
|---|---:|---|
| ALM | **5,948 / 41,910 (14%)** | fit.summary |
| DSP | **64 / 112 (57%)** | fit.summary |
| Registers | 7,780 | fit.summary |
| M10K | 20 / 553 (4%) | fit.summary |
| Block mem bits | 85,536 | fit.summary |
| **Fmax** | **116.9 MHz** (85C) | sta.rpt |
| **Latency** | **2,755 cy ≈ 27.55 µs @100MHz** | tb_simd_top.v, 93384/93384 bit-exact |

Per-entity ALM: simd_lane_array 3,108 (60 DSP, 20 lane) · line_buffer_engine 1,991 (8×24 shift-reg) · gap_fc 422 · controller 253 · input_buffer 100 (4 M10K banks).

> ⚠️ ALM cũ 16,976 (41%) là artifact: input_buffer 625×32b với byte read-modify-write → 20k FF.
> Đã fix thành 4-bank M10K → 5,948. **DÙNG 5,948, KHÔNG dùng 16,976.**

---

## 5. DSE — bảng so sánh 2 dataflow (⭐ điểm nhấn ICDV)
Cùng device, cùng compile, cùng bit-exact contract, cùng model.

| Trục | Production (8-PE channel-par) | SIMD-20 (position-par) | Tỉ lệ |
|---|---:|---:|---:|
| Latency | 5,216 cy (52.16 µs) | **2,755 cy (27.55 µs)** | **1.89× nhanh** |
| Throughput @100MHz | 19,176 inf/s | **36,298 inf/s** | 1.89× |
| Throughput @Fmax | 20,107 inf/s (104.85MHz) | **42,427 inf/s** (116.9MHz) | 2.11× |
| ALM | 2,201 (5%) | 5,948 (14%) | 2.70× |
| DSP | 28 (25%) | 64 (57%) | 2.29× |
| Registers | 3,177 | 7,780 | 2.45× |
| M10K | 20 | 20 | 1.0× |
| Fmax | 104.85 MHz | 116.9 MHz | +11% |
| Control | FSM phẳng 8 state | 2 vòng lồng + pipeline decouple, 10 phase | phức tạp hơn |

→ Pareto area↔latency: production = low-area/simple; SIMD-20 = low-latency/high-area.

> Cả hai biến thể **cùng accuracy (94.65%), cùng bit-exact contract, cùng model, cùng device** → DSE
> thuần về dataflow, KHÔNG đánh đổi độ chính xác. Đây là điểm phân biệt với SoTA (thường so kiến trúc
> khác model/dataset).

### 5b. Đoạn so sánh hoàn chỉnh (prose cho paper — Section "Design-Space Exploration")

We evaluate two dataflow mappings of the *same* INT8 1D-CNN on the *same* Cyclone V device, both
verified bit-exact against the Python golden model (so the comparison isolates the dataflow, not the
numerics). The **channel-parallel** mapping (production, 8 PEs) computes one output position at a time
across all output channels in parallel, streaming the input through a shift-register window; its control
reduces to a flat 8-state FSM. The **position-parallel** mapping (SIMD-20) instead computes 20 output
positions per cycle through a 20-lane MAC array fed by a line-buffer, decoupling load and compute across
a 10-phase, doubly-nested controller.

The two occupy opposite corners of an **area–latency Pareto front**. Position-parallelism cuts latency
**1.89×** (5,216 → 2,755 cycles; 52.16 → 27.55 µs at 100 MHz) and, combined with an 11% higher Fmax,
raises sustained throughput **2.11×** (20.1k → 42.4k inferences/s). This speed-up is paid for in area:
**2.70× more logic** (2,201 → 5,948 ALMs, 5% → 14% of the device) and **2.29× more DSPs** (28 → 64, 25%
→ 57%), since 20 parallel lanes need 60 multipliers plus a wider line-buffer, against 8 PEs / 24
multipliers for the streaming design. On-chip memory is identical (20 M10K) because both store the same
inter-layer feature maps.

For the target use case — **wearable, continuous single-lead ECG monitoring** — the channel-parallel
design is the better operating point: at one inference per heartbeat (~1 Hz) a 52 µs latency is already
four orders of magnitude faster than required, so the 1.89× latency advantage of SIMD-20 buys nothing,
while its 2.3× DSP cost directly raises dynamic power (DSPs dominate switching energy in this design).
The channel-parallel core also leaves 95% of the device free for the HPS/JTAG bridge, PLL, and runtime
weight-reload logic, and is the variant we validated on-board (94.27% on DE10-Standard). SIMD-20 becomes
attractive only when throughput is the binding constraint — e.g. batch screening or multi-lead fusion —
where its higher inferences/s amortizes the area. The contribution is therefore not "which is better"
but a *quantified, accuracy-neutral* characterization of the dataflow trade-off on a deployed FPGA target.

---

## 6. On-board DE10-Standard (Phase D)
- JTAG-to-Avalon + System Console, weight $readmemh Chapman.
- Kết quả memory: **94.27% (1004/1065 test)** khớp Python 94.65%.
- 🔲 **NEEDS-SOURCE**: số 1004/1065 hiện CHỈ trong memory, chưa thấy trong file .log/.md committed.
  Trước khi đưa vào bài: tìm lại JTAG log hoặc **chạy lại on-board** để có log cite được.
- UART variant: RTL+pin+host script READY, chưa chạy (chờ module USB-TTL 3.3V).

---

## 7. Năng lượng (PowerPlay)

### 7a. Production trên Cyclone V (DE10-Standard) — 🔲 CẦN XÁC NHẬN TỪ REPORT
- Memory ghi production: Total 623mW / Dyn 198mW / Static 413mW → 10.3µJ dyn / 32.5µJ total per inf.
- 🔲 **NEEDS-SOURCE**: xác nhận từ PowerPlay report (.pow.rpt), không phải memory.
- 🔲 SIMD-20: CHƯA chạy PowerPlay. Nếu muốn DSE có trục energy đầy đủ → chạy thêm (1 ngày).
  ICDV KHÔNG bắt buộc energy; có thì mạnh hơn. Cân nhắc sau khi xong draft text.

### 7b. ⭐ Cùng core trên Cyclone IV E (DE0-Nano) — ✅ VERIFIED gate-level SDF
Nguồn: `hardware/fpga_de0/output_files/ecg_de0_100.pow.summary` + `.pow.rpt` (2026-06-21).
**Cùng `ecg_accelerator_top` 8-PE production core**, port sang DE0-Nano `EP4CE22F17C6`
(Cyclone IV E), topology Chapman mặc định. Power confidence cao hơn DE10 vì dùng VCD
**gate-level SDF** (back-annotated delay, slow 1200mV/85°C) thay vì VCD RTL.

| Metric | DE0-Nano (Cyclone IV E) | DE10 (Cyclone V) | Δ |
|---|---:|---:|---|
| **Total thermal power** | **247.3 mW** | 623 mW (memory) | −60% |
| Core dynamic | 135.4 mW | 198 mW | −32% |
| **Core static** | **79.7 mW** | 413 mW | **−81%** |
| I/O | 32.2 mW | ~12 mW | — |
| Energy/inference (total) | **12.9 µJ** | 32.5 µJ | −60% |
| Energy/inference (dynamic) | 7.07 µJ | 10.3 µJ | −31% |

- Energy = Power × latency = Power × 52.16 µs (cùng 5216 cy, bit-exact).
- **Static −81%** là driver chính: Cyclone V SoC die (có hard ARM) leak ~413mW dù
  fabric nhỏ; Cyclone IV E die nhỏ, không SoC → static 79.7mW. → bằng chứng low-power
  thật, **chọn device đúng quan trọng hơn tối ưu logic** cho continuous wearable.
- **Confidence "Medium"** (gate VCD, Unknown 0.0%, Toggle 80.9%, 86.9% signals có toggle
  rate từ sim). "High" cần 100% toggle coverage gồm glitch nội bộ — không khả thi với
  Quartus Lite + Questa FSE. Medium + 0% unknown = trần thực tế free-tier, defendable.
- Verify đường đi: `quartus_eda` → `.vo`+`.sdo` → RTL smoke 6/6 PASS → gate SDF
  (`result=3, cycles=5216` slow-corner) → VCD compute window 25.2–77.4µs → PowerPlay.

### 7c. Resource DE0-Nano (đối chiếu device nhỏ) — `ecg_de0_top.fit.summary`
| Metric | DE0-Nano | DE10 production | Ghi chú |
|---|---:|---:|---|
| Logic | 8,035 LE (36%) | 2,201 ALM (5%) | LE (Cyclone IV) ≠ ALM (Cyclone V), không so trực tiếp |
| Block RAM | 95,776 bit / 456 M9K seg (16%) | 85,536 bit / 20 M10K (4%) | RAM vào M9K đúng (forked ramstyle) |
| Multipliers | 44 / 132 (33%) | 28 DSP / 112 (25%) | 9×9 mult (CIV) vs DSP18 (CV) |
| Fmax @100MHz SDC | +0.44 ns slack, **104.6 MHz** | 104.85 MHz | gần như đồng nhất |
| Demo clock | 50 MHz (+7.14 ns slack) | 100 MHz | DE0 demo hạ 50MHz, vẫn dư timing |

> ⚠️ DE0-Nano là **port để đo low-power trên die nhỏ + chạy GLS timing-accurate** (Cyclone IV
> CÓ SDF, Cyclone V Lite không có). KHÔNG phải bản on-board demo (bus Avalon 83-wire ảo hóa).
> On-board demo thật vẫn là DE10 JTAG (94.27%). Dùng số DE0 cho **năng lượng & low-power story**,
> số DE10 cho **deployment & DSE**.

---

## 8. Việc phải dọn trước khi viết (checklist)
- [ ] 🔴 CHỐT một con số accuracy: 94.65 (mới, FC-bias) vs 94.37 (Table 4 cũ). Khuyến nghị 94.65.
- [ ] 🔴 Fmax production = 104.85 (baseline) / 108.94 (Phase B01 weight-RAM) / board thật jtag_top +2.202ns@100MHz (~125MHz). KHÔNG dùng 137.6. Chốt: DSE dùng 104.85 (so công bằng với SIMD baseline), năng lực weight-reload dùng 108.94, board dùng jtag_top.
- [ ] 🟠 Cross-dataset: dùng số JSON (C3 0.9263, C4 0.9336), không dùng số PROJECT.md.
- [ ] 🟠 On-board 1004/1065: tìm log hoặc chạy lại để cite được.
- [ ] 🟢 (tùy chọn) PowerPlay SIMD-20 nếu muốn energy 2 phía.
- [ ] 🟢 SoTA table 3-4 paper ECG-FPGA (ICDV không cần 10).
- [x] 🟢 Energy DE0-Nano: ✅ 247.3mW total / 12.9µJ-inf, gate-level SDF confidence Medium (§7b). Low-power story = static −81% vs DE10. Số DE10 production (§7a) vẫn 🔲 NEEDS-SOURCE.
