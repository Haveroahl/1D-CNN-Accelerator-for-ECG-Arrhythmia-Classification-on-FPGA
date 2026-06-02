# Software — ECG 1D-CNN Pipeline (PyTorch)

> Đây là tài liệu vận hành software duy nhất. Kiến trúc model, quantization spec, kết quả: xem [../PROJECT.md](../PROJECT.md).

## Môi trường (Windows, chạy trong d:\Thesis101)
- Python 3.x, venv tại `d:\Thesis101\.venv`
- `cd d:\Thesis101\software\python`
- Dataset: `d:\Thesis101\data\Chapman` (cross-dataset: `d:\Thesis101\data\ptbxl`)
- Activate venv: `.\.venv\Scripts\Activate.ps1` (PowerShell)
- Mọi script chạy từ `software/python/`. Default `--data_dir` là path tương đối `../../data/Chapman`, không cần truyền tay.

Packages: `torch numpy scipy scikit-learn wfdb tqdm pandas matplotlib`

---

## Scripts hiện có

| Script | Chức năng |
|---|---|
| `train.py` | Train float32 baseline → `results/best_model.pth` |
| `prune_finetune.py` | Structured channel pruning (4,4,8,8) → `results/best_model_pruned.pth` |
| `quantization/qat_int8.py` | QAT-INT8 power-of-2 round-half-up (A2, method chính) → `results/qat_int8/model_qat_int8.pth` |
| `quantization/qat_int8_general.py` | A3 general-scale INT8 (ablation, `--rescale-mode round/floor`) |
| `quantization/qat_int8_floor.py` | A4 power-of-2 floor truncation (ablation) |
| `run_ablation_quant.py` | Orchestrator chạy A1–A4 → `results/ablation_quant/table4.txt` |
| `export_weights_int8.py` | Export `flat_weights.hex` (580 INT8, không comment) cho RTL |
| `generate_golden.py` | Sinh golden `.mem` (21 checkpoints/sample) cho RTL verification |
| `interpretability.py` | Layer ablation / calibration / noise robustness |
| `cross_eval/` | PTB-XL cross-dataset eval (preprocess + 6 modes C1–C6) |

---

## Pipeline chính (Chapman → hardware weights)

```powershell
cd d:\Thesis101\software\python

# 1. Train float32 baseline (✅ checkpoint đã có)
python train.py

# 2. Prune + finetune → channels (4,4,8,8) (✅ đã có)
python prune_finetune.py --checkpoint .\results\best_model.pth

# 3. QAT-INT8 power-of-2 (✅ đã có — 94.65% acc, F1=0.9404)
python quantization\qat_int8.py `
    --checkpoint .\results\best_model_pruned.pth `
    --output_dir .\results\qat_int8

# 4. Export weights → flat_weights.hex (✅ đã có — 580 INT8)
python export_weights_int8.py `
    --checkpoint .\results\qat_int8\model_qat_int8.pth `
    --output_dir .\results\weights_qat_int8
# → copy results/weights_qat_int8/flat_weights.hex sang hardware/RTL/

# 5. Golden files cho RTL verify (✅ đã có — 3 samples, 21/21 bit-exact)
python generate_golden.py `
    --checkpoint .\results\qat_int8\model_qat_int8.pth `
    --output_dir .\results\golden --sample_idx 0   # lặp 0,1,2
```

---

## Phase A' — QAT ablation (contribution C1, Table 4)

```powershell
# Chạy cả 4 variant (A2 đã train sẵn thì truyền --a2_checkpoint để bỏ qua re-train)
python run_ablation_quant.py `
    --pruned_checkpoint .\results\best_model_pruned.pth `
    --a2_checkpoint     .\results\qat_int8\model_qat_int8.pth
# → results/ablation_quant/{a1_float32,a2_p2_round,a3_general_round,a4_p2_floor}/results.json
# → results/ablation_quant/table4.txt
```

| Variant | Scale | Rescale | DSP extra |
|---|---|---|---|
| A1 Float32 | — | — | — |
| A2 Power-of-2 (ours) | `2^nb` | `(acc+2^(nb-1))>>nb` | 0 |
| A3 General-scale | `abs_max/127` | float multiply | +4 DSP18 |
| A4 Power-of-2 + floor | `2^nb` | `acc>>nb` | 0 |

---

## Phase A — Cross-dataset PTB-XL (contribution C3) — ✅ DONE

```powershell
python cross_eval\ptbxl_preprocess.py   # → data/ptbxl_processed/ptbxl_dataset.npz
python cross_eval\ptbxl_eval.py `
    --chapman_ckpt .\results\qat_int8\model_qat_int8.pth `
    --output_dir   .\results\cross_eval
# → results/cross_eval/ptbxl_cross_eval.json (6 modes C1–C6 + U0)
```

Kết quả: C1 in-dist 0.9446 / C2 zero-shot 0.7714 / C3 linear-probe 0.9249 / C4 finetune 0.9329.
Key finding: C2==C6 → drop 100% do distribution shift, QAT không gây thêm loss.

---

## Lưu ý quan trọng
- `flat_weights.hex`: KHÔNG có comment lines — `$readmemh` đọc từ byte 0.
- `nb` per layer: Conv1=8, Conv2=6, Conv3=6, Conv4=7, FC=0 (hardcoded trong RTL).
- ReLU **chỉ** sau Conv4 — Conv1-3 không có (preserve negative ECG features).
- GAP: integer floor `sum >> 2` (không phải float average).
- FC: nb=0, raw INT32 logits → argmax.
- Power-of-2 INT8 robust: PTQ (calibrate, no fine-tune) đạt 94.08%; QAT cải thiện thêm ~0.3% (94.37%). Dùng QAT cho checkpoint chính (gap với float chỉ 0.28%), PTQ là baseline A0 trong Table 4.
