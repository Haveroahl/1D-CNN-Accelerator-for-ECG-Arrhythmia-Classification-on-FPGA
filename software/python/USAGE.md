# ECG_1DCNN - Usage Guide

## Mục Lục

1. [Chuẩn Bị](#chuẩn-bị)
2. [Training](#training)
3. [Pruning (tùy chọn)](#pruning-tùy-chọn)
4. [Quantization](#quantization)
   - [Q8.8 Fixed-Point (16-bit)](#q88-fixed-point-16-bit)
   - [INT8 fphys (8-bit)](#int8-fphys-8-bit)
5. [Evaluation](#evaluation)
6. [Export Weights cho Hardware](#export-weights-cho-hardware)
7. [Quick Reference](#quick-reference)
8. [Model Properties](#model-properties)

---

## Chuẩn Bị

```bash
cd /home/duc/Thesis/software/python
source /home/duc/Thesis/.venv/bin/activate
```

Cấu trúc thư mục:

```
software/python/
├── model/model.py              # ECG_1DCNN + ECG_1DCNN_Q88
├── utils/dataset.py            # Chapman ECG dataset loader
├── utils/evaluate.py           # Metrics
├── quantization/
│   ├── quantize_q88.py         # Q8.8 quantization
│   ├── quantize_int8.py        # INT8 weight quantization + nb calibration
│   └── evaluate_quantized.py   # So sánh float32 vs Q8.8 vs INT8
├── train.py                    # Training
├── prune_finetune.py           # Channel pruning + fine-tuning
├── export_weights_q88.py       # Export Q8.8 .mem files cho Verilog
└── export_weights_int8.py      # Export INT8 .mem files cho Verilog
```

---

## Training

```bash
./.venv/bin/python train.py \
    --data_dir /home/duc/Thesis/data/Chapman \
    --epochs 100 \
    --batch_size 128 \
    --learning_rate 1e-3
```

**Output:**
- `./results/best_model.pth` — float32 baseline
- `./results/train_log.csv`

**Expected:** ~94.84% test accuracy

---

## Pruning (tùy chọn)

Structured channel pruning, giảm 49.4% parameters.

```bash
./.venv/bin/python prune_finetune.py \
    --checkpoint ./results/best_model.pth \
    --data_dir /home/duc/Thesis/data/Chapman \
    --output_dir ./results
```

| Layer | Full → Pruned | Params |
|-------|--------------|--------|
| Conv1 | 1→4 → 1→3   | 20→15  |
| Conv2 | 4→8 → 3→6   | 160→90 |
| Conv3 | 8→8 → 6→6   | 160→90 |
| Conv4 | 8→16 → 6→10 | 320→150|
| FC    | 16→4 → 10→4 | 64→44  |
| **Total** | | **1244→654 (-49.4%)** |

**Fine-tuning:** Phase 1 (30 epochs, lr=1e-3) → Phase 2 (20 epochs, lr=1e-4)

**Output:**
- `./results/best_model_pruned.pth` — ~92% accuracy

---

## Quantization

### Q8.8 Fixed-Point (16-bit)

Format chuẩn fixed-point: 8 bit nguyên + 8 bit thập phân, scale cố định = 1/256.

```bash
# Full model
./.venv/bin/python quantization/quantize_q88.py \
    --checkpoint ./results/best_model.pth \
    --output_dir ./results/q88

# Pruned model
./.venv/bin/python quantization/quantize_q88.py \
    --checkpoint ./results/best_model_pruned.pth \
    --output_dir ./results/q88_pruned
```

**Phương pháp:**
```
q = round(w × 256),  clip [-32768, 32767]   (int16)
w' = q / 256.0                               (dequantize về float để evaluate)
```

**Đặc điểm:**
- Scale **cố định** = 1/256, không cần calibration
- Range: [-128.0, +127.996]  → phù hợp với weights ECG model nhỏ
- Hardware: `acc = Σ w_q88 × x_q88` (int32) → `acc >> 8` → Q8.8 output
- Memory: 16-bit/weight → 2.5 KB (full), 1.3 KB (pruned)

**Output:** `./results/q88/model_q88.pth`

---

### INT8 fphys (8-bit)

Theo Liu et al. (2023) — fully end-to-end INT8, không cần float multiply.

```bash
# Full model
./.venv/bin/python quantization/quantize_int8.py \
    --checkpoint ./results/best_model.pth \
    --output_dir ./results/int8 \
    --data_dir   /home/duc/Thesis/data/Chapman

# Pruned model
./.venv/bin/python quantization/quantize_int8.py \
    --checkpoint ./results/best_model_pruned.pth \
    --output_dir ./results/int8_pruned \
    --data_dir   /home/duc/Thesis/data/Chapman
```

**Phương pháp (fphys Eq.10-12):**

**Bước 1 — Weight & Input quantization (Eq.10):**
```
w_int8       = floor(w / |w|_max × 127 + 0.5),  range [-127, 127]
input_int8   = floor(x / |x|_max × 127 + 0.5),  range [-127, 127]
input_scale  = max_dataset|x| / 127
```

**Bước 2 — Activation shift bits (Eq.11):**
```
nb_l = ceil(log2(max_dataset |O_l| / 127))
O_l  = raw output của conv_l (trước ReLU/pool)
Hardware: acc_int32 >>> nb_l  (arithmetic right shift, không cần multiplier)
```

**Pipeline hardware (end-to-end INT8):**
```
Input(INT8) → Conv(INT8×INT8→INT32) → [ReLU] → >>>nb → clip[-127,127] → MaxPool(INT8) → ...
              └── conv1-3: không ReLU ──────────────────────────────────────────────────┘
              └── conv4: có ReLU ────────────────────────────────────────────────────────┘
```

**Tại sao dùng power-of-2 scale?**
```
nb = power-of-2 → chia cho 2^nb = arithmetic right shift = wire routing trên FPGA
→ 0 logic gates, không cần multiplier cho rescaling
```

**Arguments:**

| Argument | Default | Mô tả |
|----------|---------|-------|
| `--checkpoint` | required | Model .pth |
| `--output_dir` | `./results/int8` | Output directory |
| `--data_dir` | `/home/duc/Thesis/data/Chapman` | Dataset cho calibration |
| `--n_cal_batches` | `20` | Số batch cho calibration |

**Output:** `./results/int8/model_int8.pth` — INT8 checkpoint với `w_scales`, `nb`, `input_scale`

---

## Evaluation

So sánh accuracy trước/sau quantization:

```bash
# Float32 vs Q8.8 vs INT8 (full model)
./.venv/bin/python quantization/evaluate_quantized.py \
    --float_ckpt ./results/best_model.pth \
    --q88_ckpt   ./results/q88/model_q88.pth \
    --int8_ckpt  ./results/int8/model_int8.pth

# Float32 vs Pruned Q8.8 vs Pruned INT8
./.venv/bin/python quantization/evaluate_quantized.py \
    --float_ckpt       ./results/best_model.pth \
    --pruned_ckpt      ./results/best_model_pruned.pth \
    --q88_ckpt         ./results/q88/model_q88.pth \
    --int8_ckpt        ./results/int8/model_int8.pth \
    --pruned_int8_ckpt ./results/int8_pruned/model_int8.pth
```

**Output mẫu:**
```
  Method                     Accuracy    ΔAcc    Params    Memory  Status
  ────────────────────────────────────────────────────────────────────────
  Float32 (Baseline)          94.84%    +0.00%     1244    5.0 KB  — baseline
  Q8.8 (16-bit fixed-point)   94.84%    +0.00%     1244    2.5 KB  ✓ no loss
  INT8 (PTQ)                  94.65%    -0.19%     1244    1.2 KB  ✓ acceptable
  Pruned Float32              ~92.00%   -2.84%      654    2.6 KB  ✓ acceptable
  Pruned INT8 (PTQ)           ~92.00%   -2.84%      654    0.6 KB  ✓ acceptable
```

---

## Export Weights cho Hardware

### Q8.8 Export

```bash
# Full model
./.venv/bin/python export_weights_q88.py \
    --checkpoint ./results/best_model.pth \
    --output_dir ./results/weights_q88

# Pruned model
./.venv/bin/python export_weights_q88.py \
    --checkpoint ./results/best_model_pruned.pth \
    --output_dir ./results/weights_q88_pruned
```

**Output files:**
```
./results/weights_q88/
├── conv1_weight.mem    # Q8.8 weights (4-char hex, int16)
├── conv1_bias.mem
├── conv2_weight.mem  ...
├── fc_weight.mem
├── fc_bias.mem
├── flat_weights.mem    # All weights concatenated
└── weights_summary.json
```

**Verilog usage (Q8.8):**
```verilog
// 16-bit per weight, scale fixed = 1/256
$readmemh("conv1_weight.mem", conv1_rom);  // int16 values

// Pipeline:
// x_q88     = round(x_float * 256) as int16
// acc_int32  = conv(x_q88, conv1_rom) + bias_q88
// out_q88   = acc_int32 >> 8  (fixed shift, no per-layer param needed)
```

---

### INT8 Export

```bash
# Full model
./.venv/bin/python export_weights_int8.py \
    --checkpoint ./results/int8/model_int8.pth \
    --output_dir ./results/weights_int8

# Pruned model
./.venv/bin/python export_weights_int8.py \
    --checkpoint ./results/int8_pruned/model_int8.pth \
    --output_dir ./results/weights_int8_pruned
```

**Output files:**
```
./results/weights_int8/
├── conv1_weight.mem    # INT8 weights, range [-127,127] (2-char hex)
├── conv1_bias.mem
├── conv2_weight.mem  ...
├── fc_weight.mem
├── fc_bias.mem
├── nb_shifts.mem       # Per-layer shift bits, conv1-4+fc (5 entries)
├── input_scale.txt     # Input quantization scale (reference)
├── flat_weights.mem    # All weights concatenated
└── weights_summary.json
```

**nb_shifts.mem** (ví dụ):
```
// Per-layer activation right-shift bits (fphys Eq.11)
// Hardware: acc_int32 >>> nb_lut[layer_idx]
03   // conv1
04   // conv2
04   // conv3
03   // conv4
02   // fc
```

**Verilog usage (INT8):**
```verilog
// Weights: $readmemh("conv1_weight.mem", conv1_rom);
// nb hardcoded as parameters (from nb_shifts.mem):
parameter integer NB_CONV1 = 3;
parameter integer NB_CONV2 = 4;
parameter integer NB_CONV3 = 4;
parameter integer NB_CONV4 = 3;

// Pipeline:
// input_int8 = clamp(floor(x / input_scale + 0.5), -127, 127)
// acc_int32  = conv(input_int8, weight_int8) + bias_int
// out_int8   = clamp(acc_int32 >>> NB_CONV1, -127, 127)
```

---

## Quick Reference

### So Sánh Q8.8 vs INT8

| Tiêu chí | Q8.8 (16-bit) | INT8 (fphys) |
|----------|---------------|--------------|
| Bit width | 16-bit | 8-bit |
| Memory (full) | 2.5 KB | 1.2 KB |
| Memory (pruned) | 1.3 KB | 0.6 KB |
| Calibration | Không cần | Cần (dataset) |
| Scale | Cố định 1/256 | Per-layer (calibrated) |
| Accumulator | int32 → >>8 | int32 → >>>nb |
| Accuracy loss | ~0% | ~-0.19% |
| Hardware đơn giản hơn | Shift cố định | nb hardcoded as param |

---

### Pipeline A: Full model — Q8.8

```bash
cd /home/duc/Thesis/software/python

# 1. Train
./.venv/bin/python train.py

# 2. Quantize Q8.8
./.venv/bin/python quantization/quantize_q88.py \
    --checkpoint ./results/best_model.pth \
    --output_dir ./results/q88

# 3. Evaluate
./.venv/bin/python quantization/evaluate_quantized.py \
    --float_ckpt ./results/best_model.pth \
    --q88_ckpt   ./results/q88/model_q88.pth

# 4. Export
./.venv/bin/python export_weights_q88.py \
    --checkpoint ./results/best_model.pth \
    --output_dir ./results/weights_q88
```

**Kết quả:** ~94.84% | 2.5 KB | 2× compression, zero accuracy loss

---

### Pipeline B: Full model — INT8 (fphys)

```bash
cd /home/duc/Thesis/software/python

# 1. Train
./.venv/bin/python train.py

# 2. Quantize + calibrate INT8
./.venv/bin/python quantization/quantize_int8.py \
    --checkpoint ./results/best_model.pth \
    --output_dir ./results/int8

# 3. Evaluate
./.venv/bin/python quantization/evaluate_quantized.py \
    --float_ckpt ./results/best_model.pth \
    --int8_ckpt  ./results/int8/model_int8.pth

# 4. Export
./.venv/bin/python export_weights_int8.py \
    --checkpoint ./results/int8/model_int8.pth \
    --output_dir ./results/weights_int8
```

**Kết quả:** ~94.65% | 1.2 KB | 4.2× compression

---

### Pipeline C: Prune → INT8 (fphys)

```bash
cd /home/duc/Thesis/software/python

# 1. Train
./.venv/bin/python train.py

# 2. Prune + fine-tune
./.venv/bin/python prune_finetune.py \
    --checkpoint ./results/best_model.pth \
    --output_dir ./results

# 3. Quantize pruned model + calibrate
./.venv/bin/python quantization/quantize_int8.py \
    --checkpoint ./results/best_model_pruned.pth \
    --output_dir ./results/int8_pruned

# 4. Evaluate
./.venv/bin/python quantization/evaluate_quantized.py \
    --float_ckpt       ./results/best_model.pth \
    --pruned_ckpt      ./results/best_model_pruned.pth \
    --pruned_int8_ckpt ./results/int8_pruned/model_int8.pth

# 5. Export
./.venv/bin/python export_weights_int8.py \
    --checkpoint ./results/int8_pruned/model_int8.pth \
    --output_dir ./results/weights_int8_pruned
```

**Kết quả:** ~92% | ~0.6 KB | ~8.4× compression

| Stage | Params | Memory | Accuracy |
|-------|--------|--------|----------|
| Float32 baseline | 1244 | 5.0 KB | 94.84% |
| After pruning | 654 | 2.6 KB | ~92% |
| After INT8 quantize | 654 | 0.6 KB | ~92% |

---

## Model Properties

| Property | Value |
|----------|-------|
| Input | 2500-sample ECG (Lead II @ 250 Hz) |
| Output | 4 classes: AFIB / GSVT / SB / SR |
| Layers | Conv1(1→4) → Conv2(4→8) → Conv3(8→8) → Conv4(8→16) → FC(16→4) |
| ReLU | After Conv4 only |
| Parameters | 1,244 |
| Baseline Accuracy | 94.84% |

---

## Thời Gian Thực Thi

| Task | GPU | CPU |
|------|-----|-----|
| Training (100 epochs) | 30 min | 3 hours |
| Q8.8 quantize | < 1 min | < 1 min |
| INT8 quantize + calibrate | < 2 min | 5 min |
| Pruning + fine-tuning | 1 hour | 5 hours |
| Evaluation | 20 sec | 2 min |
| Export | < 1 min | < 1 min |

---

**Last Updated:** 2026-04-27
