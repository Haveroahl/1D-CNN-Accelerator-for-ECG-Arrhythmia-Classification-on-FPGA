# Table 4 — Quantization Ablation (Contribution C1) — CHỐT

**Dataset**: Chapman ECG, lead II, 2500 samples, 4-class (AFIB/GSVT/SB/SR).
**Eval**: bit-exact INT8 (`int8_forward` == RTL path). **Model**: pruned (4,4,8,8), 654 params.
**Split**: patient-independent 70/15/15, seed=42. **Ngày chốt**: 2026-06-02.

Ma trận 2 trục: **train-method** (PTQ không train / QAT fine-tune) × **scale-type** (power-of-2 shift / general float).

---

## Bảng chính — single-run (seed=42)  ⭐ DÙNG CHO PAPER

| Variant | Scale | Train | Acc % | F1 | DSP rescale |
|---|---|---|---|---|---|
| **A1**  Float32 baseline | — | — | 94.65 | 0.9402 | — (upper bound) |
| **A0**  PTQ power-of-2 | `2^nb` | none (calibrate) | 94.08 | 0.9338 | **+0** |
| **A0'** PTQ general | `abs_max/127` | none (calibrate) | 94.46 | 0.9380 | +4 |
| **A2**  QAT power-of-2 (ours) | `2^nb` | fake-quant | **94.37** | **0.9364** | **+0** |
| **A3**  QAT general | `abs_max/127` | fake-quant | 94.65 | 0.9398 | +4 |
| **A4**  QAT power-of-2 floor | `2^nb` | fake-quant | 93.99 | 0.9328 | +0 |

Đọc bảng:
- **A2 (ours) vs A1 float**: −0.28% acc — gap lượng tử rất nhỏ.
- **A2 (ours) vs A3 general**: −0.28% acc, nhưng **tiết kiệm 4 DSP18** (rescale chỉ shift+add, 0 multiplier).
- **A2 vs A4 floor**: +0.38% — round-half-up có lợi so với floor truncation (RQ2).
- **A0 PTQ vs A2 QAT** (power-of-2): QAT +0.29% — QAT bù một phần thiệt hại do ép scale thô.

---

## Bảng phụ 1 — 5 test-fold robustness (mean ± std)

> **Scope (label trung thực)**: shared float pruned model, re-quant mỗi fold; đo variance của *quantization + test split*, **KHÔNG phải leak-free 5-fold CV**.

| Variant | Acc % (mean±std) | F1 (mean±std) | DSP |
|---|---|---|---|
| A0  PTQ p2 | 94.10 ± 0.61 | 0.9345 ± 0.0066 | +0 |
| A0' PTQ general | 94.19 ± 0.43 | 0.9353 ± 0.0048 | +4 |
| A2  QAT p2 (ours) | 93.60 ± 0.93 | 0.9287 ± 0.0108 | +0 |
| A3  QAT general | 94.30 ± 0.81 | 0.9365 ± 0.0089 | +4 |
| A4  QAT p2 floor | 93.59 ± 0.63 | 0.9287 ± 0.0072 | +0 |

→ std (0.4–0.9%) ≥ mọi khác biệt giữa các variant → khác biệt accuracy trong bảng chính phần lớn **trong nhiễu**; chỉ trục DSP (0 vs 4) là khác biệt chắc chắn.

## Bảng phụ 2 — reproducibility A2/A3 (cùng split, train lại)

| | Lần 1 | Lần 2 | Lần 3 | Spread |
|---|---|---|---|---|
| A2 QAT p2 | 94.55 | 94.37 | 94.18 | ~0.37% |
| A3 QAT general | 94.65 | 94.65 | 94.27 | ~0.38% |

(Số dùng cho bảng chính = "Lần trước" đại diện: A2 94.37, A3 94.65.)

---

## Kết luận C1 (defendable — đã chốt)

1. **Power-of-2 (ours, A2) chỉ kém float 0.28% và kém general-scale 0.28%, mà loại bỏ 4 DSP18** ở khâu rescale → lựa chọn **Pareto-ưu** cho wearable.
2. **Round-half-up cần thiết**: A2 hơn A4 floor +0.38% (RQ2).
3. **Khác biệt accuracy giữa các variant nằm trong nhiễu** (5-fold std 0.4–0.9%, reproducibility spread ~0.4%) → KHÔNG over-claim variant nào "tốt nhất" về accuracy; điểm chắc chắn duy nhất là **DSP cost** (power-of-2 = 0, general = 4).
4. **QAT KHÔNG bắt buộc**: PTQ power-of-2 cũng đạt 94.08% (A0). KHÔNG claim "QAT > PTQ" hay "QAT duy nhất".

**Pitch C1 cuối**: *"Power-of-2 INT8 đạt accuracy ngang general-scale (Δ < std) trong khi loại bỏ hoàn toàn multiplier ở rescale (−4 DSP18) — lựa chọn Pareto-ưu cho ECG accelerator năng lượng thấp."*

**Lưu ý hardware**: checkpoint deploy (`results/qat_int8/model_qat_int8.pth`, bit-exact 94.37%, 21/21 RTL match) KHÔNG bị thay đổi bởi ablation. Các rerun ghi vào thư mục tạm `_rerun_*/` (đã xoá).
