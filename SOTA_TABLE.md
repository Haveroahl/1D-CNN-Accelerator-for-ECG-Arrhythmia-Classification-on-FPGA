# SOTA_TABLE.md — So sánh SoTA ECG-FPGA (Phase E)

> Gom cho Section "Related Work / Comparison" + Pareto chart.
> Mục tiêu venue: ICDV cần 3-4 paper, MDPI cần 6-8. Bản này dựng **rộng 6-8** rồi cắt sau.
>
> **Quy tắc nguồn** (bắt buộc trước khi đưa vào bài):
> - Mỗi số phải truy được về paper gốc. Số lấy từ *search snippet* = 🔲 NEEDS-VERIFY,
>   phải mở paper gốc đọc lại con số + bảng trước khi cite (snippet có thể gộp/nhầm).
> - Số của Liu 2023 đã verify từ `Article.xml` (Table 3) + `Resoure_complare.txt` → ✅.
> - KHÔNG so % resource trực tiếp (khác die). Quy về số tuyệt đối khi có thể.
> - Khác dataset/class-count/lead → ghi rõ, KHÔNG xếp hạng accuracy thẳng hàng.

Ngày dựng: 2026-06-18.

> **2 bảng yêu cầu (2026-06-18):**
> - **Bảng A — Software/model trên Chapman**: so accuracy/F1 các model phân loại Chapman.
> - **Bảng B — Hardware/FPGA biomedical**: so accelerator FPGA cho tín hiệu y sinh (ECG/EEG/EMG).
> - Chỉ giữ entry **cite-được** (đã verify từ paper gốc); đã bỏ entry chỉ có số snippet/403.

---

## Bảng A — Software models trên Chapman dataset

> ⚠️ **Cảnh báo so sánh**: Chapman có 2 cách dùng phổ biến — (1) **4-superclass**
> AFIB/GSVT/SB/SR (giống ta), (2) **11-class / multi-label 12-lead**. Số 98-99% phần lớn
> là 12-lead multi-class → KHÔNG xếp thẳng hàng với 4-class single-lead của ta (94.65%).
> Cột "Lead / #class" là cột quyết định fairness — đọc kỹ trước khi cite.
>
> ⚠️ **Beat vs Rhythm**: Bảng A toàn bộ là **rhythm-level** (phân loại cả đoạn ghi 10s theo nhịp điệu,
> 1 nhãn/record) vì Chapman 4-superclass là bài toán rhythm. KHÁC HẲN phần lớn paper MIT-BIH ở Bảng B
> (beat-level: cắt từng heartbeat quanh đỉnh R, phân loại AAMI N/S/V/F/Q, 1 nhãn/beat). **Không so
> accuracy rhythm-level với beat-level** — khác đơn vị phân loại, khác độ khó, khác cách chia train/test
> (beat-level dễ leak inter-patient nếu chia theo beat). Đây là lý do accuracy beat-level MIT-BIH
> thường 98-99% còn rhythm-level Chapman ~94-96%.

| # | Ref (year) | Model | **Beat/Rhythm** | Lead / #class | **Params** | Acc % | macro-F1 | Ghi chú | Nguồn |
|---|---|---|---|---|---:|---:|---:|---|---|
| A1 | **Ours (2026)** | 1D-CNN pruned, INT8 P2-QAT | **Rhythm** (10s record) | **1-lead (II) / 4 (AFIB,GSVT,SB,SR)** | **654** | **94.65** | **0.9396** | bit-exact RTL, wearable | ✅ PAPER_DATA.md |
| A2 | Le et al. (LightX3ECG) ✅ | 1D-SEResNet (DSConv) + lead-attn, pruned 80% | **Rhythm** (record) | 3-lead (I,II,V1) / **4-superclass** | **5.31M** (6.52 MB) | 98.73 | 0.9718 | **cùng 4-superclass task**; nhưng 3-lead, 8100× params | ✅ arXiv 2207.12381v2 full-text |
| A3 | Bimodal CNN (PMC9941114) ✅ | 2D CNN (Inception-v3) grayscale + scalogram | **Rhythm** (record) | 12-lead / **4-superclass** | 🔲 (Inception-v3 ~23M) | 95.08 (Lead-II) / 95.74 (12-lead ens) | **0.944** (Lead-II) / 0.952 (ens) | cùng 4-superclass; image-based, model lớn | ✅ PMC9941114 full-text |
| A4 | CardioPatternFormer (2025) ✅ | Transformer (d=256, 8 heads, 4 enc-layers, d_ff~1024) | **Rhythm/diagnostic** (record, multi-label) | 12-lead / **6 multi-label** | **~3–4M** (ước từ 4·(4d²+2·d·d_ff)≈3.1M + tokenizer/head; paper KHÔNG báo) | 91.84 (Hamming) | 0.8019 | macro-AUC 0.9437; **multi-label 6-class khác task** | ✅ arXiv 2505.20481v1 (params=ước lượng) |
| A5 | **Zheng et al. (2020, gốc Chapman)** ✅ | **ML cổ điển (XGBoost gradient boosting)** — KHÔNG phải DL | **Rhythm** (record) | 12-lead / **4-superclass** | n/a (cây boosting, không tính params như NN) | — | **0.97** (XGBoost) | ✅ paper GỐC định nghĩa dataset + 4-superclass (Table 5); F1 0.97 là baseline ML. Citation = **PMC7016169** | ✅ PMC7016169 full-text |

**Đọc Bảng A** (cập nhật sau verify): **Cột Params là đòn bẩy mạnh nhất của ta.** Số đã verify từ
full-text:
- **A2 LightX3ECG = 5.31M params** (6.52 MB), cùng 4-superclass, 98.73%/F1 0.9718, nhưng 3-lead.
  → **Ta 654 params vs 5.31M ≈ 8,100× nhỏ hơn**, accuracy thấp hơn 4pp nhưng **single-lead + INT8 +
  deployable FPGA**. Đây là so sánh fair-task nhất (cùng AFIB/GSVT/SB/SR).
- **A3 Bimodal CNN = 95.08% Lead-II / F1 0.944** (cùng 4-superclass, Inception-v3 ~23M, image-based).
  → **Accuracy A3 (95.08) ≈ ta (94.65) ở Lead-II nhưng model ~35,000× lớn hơn** — luận điểm footprint cực mạnh.
- A4 CardioPatternFormer = 6-class **multi-label** (khác task, không xếp thẳng accuracy).
- A5 Zheng gốc = XGBoost baseline F1 0.97 (paper định nghĩa dataset + 4-superclass).

Luận điểm chốt: **ở cùng 4-superclass và dải accuracy ~94-95%, ta nhỏ hơn 3-4 bậc độ lớn về params
và là thiết kế single-lead INT8 deployable** — không đua accuracy đỉnh (model lớn hơn luôn thắng pp).

---

## Bảng B — FPGA accelerators cho biomedical signal classification

| # | Ref (year) | Device | Signal / Model | Dataset (#class) | **Beat/Rhythm** | Acc % | **Freq (MHz)** | Latency | **Throughput (inf/s)** | Power | Resource | Energy/inf | Nguồn |
|---|---|---|---|---|---|---:|---:|---|---:|---|---|---|---|
| B1 | **Ours (2026)** | Cyclone V 5CSXFC6D6F31C6 | ECG / 1D-CNN INT8 | Chapman (4) | **Rhythm** (10s) | **94.65** (board 94.27) | **100** (Fmax 104.85) | **52.16 µs** | **~19,200** (≈20,100 @Fmax) | 623 mW 🔲 | 2,201 ALM / 28 DSP / 20 M10K | 10.3 µJ dyn 🔲 | ✅ PAPER_DATA.md |
| B2 | **Liu et al. (2023)** | Cyclone V 5CSEBA6U23I7 | ECG / 1D-CNN + HR (fully-mapped) | Chapman (4) | **Rhythm** (10s) | **92.95** (INT8) / 93.24 (float) | **50** | 66 µs | **~15,150** ⊕ (1/66µs) | **66 mW** | ALM 51% / Reg 86% / DSP 39% (~44) / Mem 0.5% | 87.42 GOPS/W (63.48 khi +HR) | ✅ Article.xml Table 1+3 |
| B3 | Xing et al. ✅ | FPGA | ECG / **SNN** (spiking) | MIT-BIH (5: N,S,V,F,Q) | **Beat** (AAMI) | 98.26 / **92.07 inter-pat** | 🔲 | 1.37 ms/beat | **~730** ⊕ (1/1.37ms) | 🔲 | 🔲 | **346.33 µJ/beat** | ✅ review 2503.07276 [18] |
| B4 | Zhang et al. ✅ | Zynq-7000 SoC (ZC706) | ECG / lightweight NN | MIT-BIH (#class 🔲) | 🔲 (MIT-BIH→thường Beat) | 98.9 | 🔲 | 🔲 | 🔲 | 🔲 | 🔲 | **2.05 µJ/inf** | ✅ review 2503.07276 [29] |
| B5 | Rawal et al. (2023) ✅ | Zynq UltraScale | ECG / 1D-CNN (pruned PRCA) | **CinC 2017 (4)** | **Rhythm** (record) | **86.37 (HW)** / 90.80 (SW) | 🔲 | 🔲 | 🔲 | **628 mW** | 🔲 | 🔲 | ✅ BSPC 2023 DOI 10.1016/j.bspc.2023.104865 |
| B6 | Wei et al. (2021) ✅ | Zynq-7045 | ECG / 1-D CNN | **Chapman (4)** | **Rhythm** (10s) | **93.14 / F1 0.9242** | **200** | NA | 🔲 | 0.79 W | LUT 1.1% / DSP 10.67% | 33.67 GOPS/W | ✅ Article.xml Table 1+3 (rid B33) |
| B7 | Carreras et al. (2020) ✅ | Zynq-7020 | ECG / TCN | ⚠️ 🔲 (ECG5000? cần xác nhận paper FPGA-JETCAS) | 🔲 | 🔲 (~94 nếu ECG-TCN) | **120** | 17 ms | **~59** ⊕ | 3.3 W | LUT 80.76% / DSP 100% / Mem 91.4% | 33.8 GOPS/W | ✅ Article.xml Table 3 (rid B5); dataset 🔲 |
| B8 | Srivastava et al. (2022) ✅ | Artix-7 | ECG / PNN | **MIT-BIH (8: NSR,APB,PB,VT,VF,RBBB,LBBB,PVC)** | **Beat** | **98.27** | **100** | 17 s | **~0.06** ⊕ | **25 mW** | Reg 1.5% / IOB 89% | NA | ✅ Table 3 + Hindawi 2022/7564036 |
| B9 | Ran et al. (2022) ✅ | Zynq-7020 + ARM | ECG / CNN | MIT-BIH (🔲 #class) | 🔲 (MIT-BIH→thường Beat) | 🔲 | **100** | 2.895 s | **~0.35** ⊕ | 2.81 W | LUT 47.1% / DSP 68.6% / Mem 95% | NA | ✅ Table 3 (rid B28); acc 🔲 |
| B10 | Wess et al. (2017) ✅ | Zynq-7020 | ECG / **MLP + PCA** | **MIT-BIH** | **Beat** (per-beat) | **99.82** | **100** | 0.99 µs | **~1.0M** ⊕ | 0.124 W | LUT 3.6% / DSP 14.5% | 0.54 GOPS/W | ✅ Table 3 + ResearchGate 320091294 |

> ⊕ = **throughput suy ra** = 1/latency (1 inference/lần, không pipeline-overlap). KHÔNG phải số
> tác giả công bố — đánh dấu rõ trong bài. Nếu paper báo throughput riêng (vd có batching/pipeline)
> thì dùng số gốc, không dùng 1/latency. Throughput ta = 1/(5216 cy × 10ns) = 19,176, đã verify.
>
> **✅ = số verify từ paper gốc** (Liu Article.xml Table 1/3, hoặc abstract paper gốc của competitor).
> B6-B10 là 5 competitor mà Liu tổng hợp; resource để dạng % device (khác die, ghi % như gốc).
> **Dataset competitor (bổ sung từ abstract 2026-06-18):** B6 Wei = **Chapman** (cùng Liu Table 1);
> B8 Srivastava = **MIT-BIH 8-class** (acc 98.27, Hindawi 2022/7564036); B10 Wess = **MIT-BIH** + MLP/PCA
> (acc 99.82). B3 Xing = MIT-BIH 5-class; B4 Zhang = MIT-BIH; B5 Rawal = CinC-2017 4-class.
> **⚠️ B7 Carreras**: Liu cite "Optimizing TCN Inference on FPGA" (JETCAS 2020); search abstract trả về
> ECG-TCN (arXiv 2103.13740, dataset ECG5000, acc 94.2%, deploy MCU/RISC-V) — **có thể KHÁC paper** →
> dataset+acc B7 để 🔲 cho tới khi đọc đúng JETCAS 2020. B9 Ran accuracy chưa có (Liu Table 3 không liệt kê).
> (Đã bỏ 3 entry không fetch được full-text: CNN-BiLSTM, HLS Zynq 7Z020, PYNQ-Z2 — ScienceDirect 403.)

**Đọc Bảng B**: direct competitor = **B2 (Liu, cùng device + dataset + task)**. Số Liu đã verify
từ Article.xml: **INT8 Acc 92.95% / F1 0.9205** (float 93.24/0.9228), 50 MHz, 66 µs, **66 mW**,
87.42 GOPS/W. → **Ta accuracy CAO hơn Liu (94.65 vs 92.95, +1.7pp)** ở cùng dataset/task/device-family,
throughput cao hơn (~19.2k vs ~15.1k) ở freq gấp đôi. Đánh đổi: Liu power thấp hơn (66 vs 623 mW) vì
50 MHz + fully-mapped không truy cập memory runtime — đây là điểm Liu mạnh, phải thừa nhận; nhưng
note Liu 66 mW là **dual-function CNN+HR** đo gộp, và ALM/Reg của ta ít hơn nhiều (folded). B6-B10
(verified từ Liu Table 3) cho landscape: latency ta 52µs « TCN 17ms / PNN 17s / CNN 2.9s; chỉ MLP
Wess nhanh hơn (0.99µs) nhưng MLP+PCA trên MIT-BIH (per-beat, không per-record) nên không cùng dạng.
B6 Wei (Chapman, 93.14% < ta 94.65%) là so trực tiếp cùng dataset thứ 2 sau Liu. B3-B5 (MIT-BIH/CinC):
B4 Zhang (2.05 µJ/inf) là điểm energy đáng chú ý — vẫn cần xác nhận #class trước khi để cạnh trực tiếp.

**Beat vs Rhythm (cột mới)**: chỉ **3 entry cùng rhythm-level + Chapman** mới so accuracy trực tiếp
được với ta — **B1(ours)/B2(Liu)/B6(Wei)**, ta cao nhất (94.65 > 92.95 > 93.14). B5 Rawal cũng
rhythm nhưng CinC khác dataset. Các entry **beat-level MIT-BIH** (B3/B8/B10, acc 98-99%) KHÔNG so
accuracy thẳng — khác đơn vị phân loại (1 nhãn/beat vs 1 nhãn/record) + dễ leak inter-patient nếu
chia theo beat. Đây là lá chắn rebuttal: accuracy 99% beat-level ≠ bài toán rhythm-level của ta.

---

## 1. Quan sát để viết prose (defendable)

- **Direct competitor = Liu 2023 (B2)**: cùng model class (1D-CNN 4-layer), cùng Chapman 4-class,
  cùng họ Cyclone V → so sánh fair nhất. Điểm bán: (i) accuracy **+1.7pp** (94.65 vs INT8 92.95);
  (ii) **8-PE channel-parallel (folded)** vs Liu **fully-mapped** → ít ALM/Reg hơn nhiều (~9× ALM,
  ~23× Reg theo Resoure_complare.txt), DSP xấp xỉ (28 vs ~44); (iii) round-half-up vs floor (+0.38%, C1).
  Thừa nhận: Liu power thấp hơn (66 mW vs 623) vì 50 MHz + fully-mapped no-memory-traffic.
- **Params (Bảng A) là đòn bẩy mạnh nhất**: cùng 4-superclass, ta 654 vs LightX3ECG 5.31M (~8,100×),
  vs Bimodal CNN ~23M (~35,000×), ở cùng dải accuracy ~95% Lead-II. KHÔNG đua accuracy đỉnh.
- **Khác bài toán**: B3-B5 dùng MIT-BIH 5-class / CinC 4-class — ghi rõ "different dataset/format",
  KHÔNG xếp hạng accuracy thẳng hàng.
- **Latency/throughput**: ta 52 µs / ~19.2k inf/s « hầu hết thiết kế khác (TCN 17 ms, PNN 17 s,
  CNN 2.9 s). Đây là điểm Pareto mạnh cho compact single-lead model.
- **Energy/inference**: ít paper báo µJ/inf (đa số báo W + GOPS/W). Ta có µJ/inf (cần xác nhận
  PowerPlay 🔲); B4 Zhang 2.05 µJ/inf là ngoại lệ đáng chú ý — cần xác nhận #class.

---

## 2. Pareto chart — kế hoạch

- **Trục**: x = energy/inference (µJ, log) hoặc latency (log); y = accuracy (%).
- **Điểm**: mỗi paper 1 điểm; ours = 2 điểm (channel-par + SIMD-20) để show DSE front.
- **Vấn đề dữ liệu**: nhiều paper thiếu µJ/inf → fallback trục latency vs accuracy (đủ số hơn).
- **Cảnh báo**: chú thích dataset khác nhau bằng marker/màu; KHÔNG vẽ như cùng 1 bài toán.

---

## 3. Checklist Phase E còn lại

**✅ Đã verify (2026-06-18) — cite được:**
- [x] Liu 2023 accuracy: INT8 92.95% / F1 0.9205 (Article.xml Table 1). Ta +1.7pp.
- [x] 5 competitor FPGA-ECG từ Liu Table 3 (B8-B12): freq/latency/power/efficiency.
- [x] A2 LightX3ECG: 5.31M params, 4-superclass, 98.73%/0.9718 (3-lead).
- [x] A3 Bimodal CNN: 95.08% Lead-II / F1 0.944, 4-superclass.
- [x] A4 CardioPatternFormer: kiến trúc (d=256/8h/4L) → params ~3-4M (ước lượng, paper không báo).
- [x] A6 Zheng gốc: XGBoost F1 0.97, citation đúng = PMC7016169 (KHÔNG phải PMC7477611).
- [x] B3 Xing (SNN, MIT-BIH 5-class, 92.07 inter-pat), B4 Zhang (Zynq-7000, 98.9%, 2.05µJ),
      B5 Rawal (CinC 2017, 4-class, 86.37% HW).

**🔲 Còn kẹt (ScienceDirect/Springer 403 — không fetch được full-text):**
- [ ] A5 CNN-BiLSTM ensemble (S1746809424007614) — cân nhắc BỎ (không verify được task/params).
- [ ] B6 HLS Zynq 7Z020 (S0141933125000924), B7 PYNQ-Z2 (S1746809425005749), B8 AICSP 2026.
      → cần truy cập có quyền (thư viện trường / Sci-Hub / tải PDF tay) hoặc bỏ nếu không cite được.

**Còn lại:**
- [ ] 🟠 Chốt số paper "core comparison" (user: làm bản 6-8).
- [ ] 🟢 A4/A5 params: A4 là ước lượng (đánh dấu rõ "estimated"); xác nhận d_ff thật nếu muốn chính xác.
- [ ] 🟢 Vẽ Pareto (matplotlib) — **params vs accuracy** (Bảng A, đòn bẩy mạnh nhất) +
      latency vs accuracy (Bảng B). Energy thiếu số → để phụ.
- [ ] 🟢 DOI + BibTeX key cho từng ref khi chốt danh sách.

---

## Nguồn search (2026-06-18)

**Hardware (Bảng B):**
- Liu 2023 fully-mapped Cyclone V: https://www.frontiersin.org/articles/10.3389/fphys.2023.1079503/full
- Springer JRTIP 2025: https://link.springer.com/article/10.1007/s11554-025-01642-w
- Springer AICSP 2026: https://link.springer.com/article/10.1007/s10470-026-02560-y
- HLS PYNQ beat-image: https://www.sciencedirect.com/science/article/abs/pii/S1746809425005749
- HLS Zynq 7Z020 INQ: https://www.sciencedirect.com/science/article/abs/pii/S0141933125000924
- MDPI review (FPGA-accelerated ECG): https://www.mdpi.com/2079-9292/15/2/301
- arXiv systematic review (B3/B4/B5 source): https://arxiv.org/abs/2503.07276

**Software/Chapman (Bảng A):**
- Chapman 4-superclass gốc (Zheng 2020): https://pmc.ncbi.nlm.nih.gov/articles/PMC7477611/
- LightX3ECG (Le et al., 3-lead): https://arxiv.org/pdf/2207.12381
- Bimodal CNN grayscale+scalogram (4-superclass, Lead-II 96.13%): https://www.nature.com/articles/s41598-023-30208-8
- CardioPatternFormer (Transformer, multi-label): https://arxiv.org/pdf/2505.20481
- 1D-CNN-BiLSTM ensemble: https://www.sciencedirect.com/science/article/abs/pii/S1746809424007614

> **Verify trước camera-ready** (mọi ô 🔲): mở paper gốc đọc đúng con số + xác nhận
> **lead count + #class** (cột quyết định fairness Bảng A). Ưu tiên: B2 accuracy Liu (Article.xml),
> A6/A3 (cùng 4-superclass Chapman), B4 (2.05 µJ/inf — nếu đúng thì cần luận điểm).
