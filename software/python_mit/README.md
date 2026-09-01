# MIT-BIH Beat Classification — `software/python_mit/`

Pipeline PyTorch phân loại nhịp tim trên **MIT-BIH Arrhythmia Database** (`data/mitdb`, 48 records, 360 Hz, lead MLII). Độc lập với `software/python/` (Chapman). Mục tiêu: tìm model tốt nhất → pruning + INT8 → deploy FPGA.

Có **hai bài toán** trong thư mục này:

| Bài toán | Nhãn | Model chốt | Kết quả (test, intra) |
|---|---|---|---|
| **A. 5-symbol multi-task** | N,L,R,A,V + binary | `ECG_TinyMultiTask` (727 params) | 5-class F1 **0.958**, A F1 **0.856** |
| **B. 5-class AAMI** | N,S,V,F,Q | `ECG_BeatDualBranch` (14,493 params) | 5-class F1 **0.932** |

---

## Cấu trúc

```
python_mit/
├── utils/
│   ├── dataset.py        # AAMI 5-class loader (N/S/V/F/Q) + RR features
│   └── dataset_hier.py   # 5-symbol loader (N,L,R,A,V) + binary + RR features
├── model/
│   ├── model.py          # AAMI models: baseline/rr8/inception/thesis/deep/incep15/tcn/dualbranch
│   └── model_hier.py     # ECG_TinyMultiTask (2-head, ~500-800 params)
├── train.py              # train AAMI 5-class
├── train_hier.py         # train 5-symbol multi-task
└── results/
    ├── hier_5sym/        # ★ MODEL CHỐT bài A (5-symbol)
    ├── aami_dualbranch/  # ★ MODEL CHỐT bài B (5-class AAMI, intra)
    └── aami_inter/        #   AAMI inter-patient (de Chazal, số trung thực)
```

Chạy trong venv: `d:\Thesis101\.venv\Scripts\python.exe`.

---

## Tiền xử lý chung (cả hai bài)

- **Beat segmentation**: cửa sổ quanh R-peak (từ annotation `.atr`).
  - Bài A (5-symbol): **192 samples**, PRE=100 / POST=92 (giữ P-wave, cắt T-wave — quan trọng cho lớp A).
  - Bài B (AAMI): 256 samples, PRE=100 / POST=156.
  - *(PRE/POST khai báo trong `utils/dataset.py`; cả hai loader dùng chung.)*
- **Z-score** per-beat.
- **RR features** (heart-rate-invariant, chuẩn hóa theo median RR của record): 12 features gồm pre/post-RR, ±2-beat context, và **4 feature prematurity** (`prematurity`, `compensation`, `rr_irregular`, `premature_pause`) nhắm lớp A (ngoại tâm thu nhĩ — giống N về hình thái, chỉ khác nhịp đến sớm).
- **Split**: intra-patient random 80/10/10 (bài A và B chính); inter-patient AAMI DS1/DS2 (chỉ bài B, không leak bệnh nhân).
- **Mất cân bằng**: focal loss (γ=2) + class-weight tempered (T=0.5, nén spread); oversample minority có kiểm soát + jitter on-the-fly (KHÔNG nhân bản tĩnh — tránh overfit). Augment giữ hình thái: amplitude/noise/baseline-wander/time-warp ±5%.

---

## Bài A — 5-symbol multi-task (model chốt cho FPGA)

5 ký hiệu MIT-BIH phổ biến nhất: **N, L (LBBB), R (RBBB), A (APB), V (PVC)** (~100k beat; loại các ký hiệu khác).

`ECG_TinyMultiTask` — 1 backbone CNN + **2 head độc lập**:
- **Dense1 (binary)**: bình thường `{N,L,R}` (nhịp xoang dẫn) vs bất thường `{A,V}` (ngoại tâm thu).
- **Dense2 (5-class)**: N / L / R / A / V.

```
Input (192) → Conv(1→4,k5)BN→pool/4 → Conv(4→8,k5)BN→pool/4 → Conv(8→8,k3)BN→pool/4
            → GAP(8) ⊕ RR-MLP(12→8→8) → [Dense1: 16→2] [Dense2: 16→5]
```
**727 params** (~0.7 KB INT8). Loss = `Focal(bin) + 2·Focal(5class)`.

### Kết quả (`results/hier_5sym/`)

| Head | acc | F1-macro |
|---|---|---|
| Dense1 binary | 98.91% | 0.968 |
| Dense2 5-class | 98.69% | **0.958** |

Per-class (5-class): N 0.993 · L 0.984 · R 0.988 · **A 0.856** · V 0.970.

### Chạy lại
```powershell
cd d:\Thesis101\software\python_mit
..\..\.venv\Scripts\python.exe train_hier.py --epochs 100 `
    --n_rr 12 --rr_hidden 8 --lambda5 2.0 `
    --oversample_A 6 --a_weight_mult 1.5 --aug_minority A --aug_p 0.7 `
    --lr 1e-3 --lr_schedule step --lr_drop 50 `
    --output_dir results/hier_5sym
```
> **LR schedule** (1e-3 cho 50 epoch đầu → 1e-4) là yếu tố quyết định để đạt mục tiêu.

---

## Bài B — 5-class AAMI (N/S/V/F/Q)

Nhóm AAMI de Chazal. Model chốt `ECG_BeatDualBranch` (CNN hình thái + RR-MLP, 14,493 params).

### Kết quả

| Split | model | acc | F1-macro | Ghi chú |
|---|---|---|---|---|
| intra | dualbranch (`aami_dualbranch/`) | 98.90% | **0.932** | có leak bệnh nhân (lạc quan) |
| inter (DS1/DS2) | rr8 (`aami_inter/`) | 90.03% | 0.489* | trung thực, không leak; *4-class N/S/V/F (Q=paced loại theo de Chazal), S/F sập do generalization |

Trần bài B là **F** (fusion, chỉ 802 beat) và **S** (giống N) — giới hạn bản chất dữ liệu, ~0.93 là tốt nhất cho đơn-model intra.

### Chạy lại (intra)
```powershell
..\..\.venv\Scripts\python.exe train.py --scheme intra --epochs 100 `
    --loss focal --weight_temp 0.5 --model dualbranch `
    --output_dir results/aami_dualbranch
```
`--model`: `baseline|rr8|inception|thesis|deep|incep15|tcn|dualbranch`.

---

## Kết luận thiết kế (đã chứng minh bằng thực nghiệm)

- **Lớp khó luôn là lớp giống N về hình thái** (A ở bài A; S ở bài B): cần **RR features** (nhịp), không phải morphology. RR-MLP (RR phi tuyến) > nối RR tuyến tính.
- **Đổi kiến trúc CNN không vượt được trần thông tin** — deep residual ≈ inception ≈ TCN khi cùng RR; thắng/thua nằm ở **cách dùng RR + LR schedule + cửa sổ beat**.
- **Cửa sổ 192 > 256** cho beat MIT-BIH: cắt T-wave/đuôi giảm nhiễu beat lân cận → tăng precision lớp A.
- **Cân bằng**: focal + class-weight tempered + oversample-có-kiểm-soát + jitter on-the-fly. **Tránh** oversample tĩnh 20k (gây overfit lớp hiếm).
