# Software Pipeline - Trạng Thái Hoàn Thành

**Cập nhật: 12 tháng 5, 2026**

---

## 📊 Tóm Tắt Tổng Quát

| Thành Phần | Trạng Thái | Ghi Chú |
|-----------|-----------|--------|
| **Training Pipeline** | ✅ Hoàn thành | `train.py` - trainable |
| **Pruning & Fine-tune** | ✅ Hoàn thành | `prune_finetune.py` - pruned model (640 params) |
| **QAT-INT8 Quantization** | ✅ Hoàn thành | `quantization/qat_int8.py` - INT8 model trained |
| **Weight Export** | ✅ Hoàn thành | `export_weights_int8.py` - .hex & .mem files |
| **Golden Reference Files** | ✅ Hoàn thành | `generate_golden.py` - RTL verification files |
| **Model Definition** | ✅ Hoàn thành | `model/model.py` - 306 lines, multi-variant |
| **Dataset Utils** | ✅ Hoàn thành | `utils/dataset.py` - Chapman loader (238 lines) |
| **Evaluation Metrics** | ✅ Hoàn thành | `utils/evaluate.py` - 106 lines |

---

## 📁 Cấu Trúc Thư Mục

```
software/python/
├── ✅ train.py                    # Float32 baseline training
├── ✅ prune_finetune.py           # Structured channel pruning
├── ✅ quantization/
│   ├── qat_int8.py               # QAT INT8 training
│   ├── quantize_int8.py          # PTQ INT8
│   ├── fake_quantize.py          # STE fake quant ops
│   ├── base_quantizer.py         # Base class
│   ├── qat_training.py           # QAT training loop
│   └── evaluate_quantized.py     # Eval for quantized models
├── ✅ export_weights_int8.py      # Export to .mem/.hex
├── ✅ generate_golden.py          # Golden reference generation
├── ✅ model/
│   ├── model.py                  # ECG_1DCNN variants
│   ├── model_2fc.py              # 2-FC version
│   └── __init__.py
├── ✅ utils/
│   ├── dataset.py                # Chapman ECG loader
│   ├── evaluate.py               # Metrics
│   └── __init__.py
├── 📦 results/                    # Output checkpoints
│   ├── best_model.pth            # ✅ Float32 baseline
│   ├── best_model_pruned.pth     # ✅ Pruned model (640 params)
│   ├── qat_int8/
│   │   ├── model_qat_int8.pth    # ✅ Best QAT model
│   │   ├── qat_history.json
│   │   └── QAT_SUMMARY.md
│   ├── weights_qat_int8/         # ✅ Exported weights
│   │   ├── flat_weights.hex      # ✅ ROM file for Verilog
│   │   ├── conv{1-4}_weight.mem
│   │   ├── fc_weight.mem
│   │   ├── nb_shifts.mem
│   │   └── weights_summary.json
│   └── golden/                   # ✅ Golden reference files
│       ├── input_int8.mem
│       ├── after_conv1.mem
│       ├── after_conv2.mem
│       ├── after_conv3.mem
│       ├── after_conv4_relu.mem
│       ├── after_pool*.mem
│       ├── after_gap.mem
│       ├── logits_fc.mem
│       └── golden_meta.json
```

---

## 🎯 Chi Tiết Từng Thành Phần

### 1️⃣ Training Pipeline (`train.py`) ✅
- **Trạng thái**: Hoàn thành & Chạy được
- **Chức năng**:
  - Đào tạo mô hình float32 cơ sở
  - Hỗ trợ 3 biến thể mô hình (base/v2/v3)
  - Adam optimizer với MultiStepLR scheduler
  - 100 epochs (mặc định)
- **Checkpoint đầu ra**: `best_model.pth` (27 KB) - được tạo vào 2026-05-09

### 2️⃣ Pruning & Fine-tuning (`prune_finetune.py`) ✅
- **Trạng thái**: Hoàn thành & Chạy được
- **Phương pháp**: L1-norm structured channel pruning
- **Kết quả**:
  - Giảm từ 1244 → **640 params** (48.6% giảm)
  - Conv channels: (4,4,8,8) như target
  - Accuracy after fine-tune: **93.8% (float)** → **94.08% (Q8.8)** → **93.89% (INT8)**
  - 2-phase fine-tuning: 30 epochs @ 1e-3, then 20 @ 1e-4
- **Checkpoint đầu ra**: `best_model_pruned.pth` (6.6 KB) - được cập nhật 2026-05-11

### 3️⃣ QAT-INT8 Quantization (`quantization/qat_int8.py`) ✅
- **Trạng thái**: Hoàn thành & Chạy được
- **Phương pháp**: Quantization-Aware Training với STE
- **INT8 Pipeline**:
  - Fake-quantize weights + activations during training
  - Power-of-2 scales: `shift_bits = floor(log2(127/abs_max))`
  - Rounding: round-half-up (KHÔNG floor)
  - nb per layer: [8, 7, 6, 8, 0] for [conv1, conv2, conv3, conv4, fc]
  - w_shift per layer: [6, 7, 6, 8, 7]
  - input_shift_bits = 2
- **Kết quả**:
  - **Accuracy (INT8 eval): 94.65%** ⭐ (Best result)
  - **F1-macro: 0.9404**
  - AFIB Recall: 0.9404
- **Checkpoint đầu ra**: 
  - `model_qat_int8.pth` (9.8 KB)
  - `qat_history.json` - training history

### 4️⃣ Weight Export (`export_weights_int8.py`) ✅
- **Trạng thái**: Hoàn thành & Chạy được
- **Chức năng**: Chuyển đổi INT8 checkpoint sang .mem/.hex
- **Định dạng**:
  - **conv{1-4}_weight.mem**: INT8 weights
  - **conv{1-4}_bias.mem**: INT32 biases
  - **fc_weight.mem**: FC layer weights
  - **fc_bias.mem**: FC layer biases
  - **flat_weights.hex**: Concatenated weights (2.3 KB) - **NO COMMENTS**
  - **nb_shifts.mem**: Per-layer nb values
  - **rom_address_map.txt**: RAM layout guide
  - **weights_summary.json**: Metadata
- **Đầu ra**: Thư mục `results/weights_qat_int8/` (15 files)

### 5️⃣ Golden Reference Generation (`generate_golden.py`) ✅
- **Trạng thái**: Hoàn thành & Chạy được
- **Chức năng**: Tạo tham chiếu INT8 cho RTL verification
- **Capture các stage**:
  - ✅ input_int8 (2500 samples)
  - ✅ after_conv{1,2,3,4}
  - ✅ after_pool{1,2,3,4}
  - ✅ after_conv4_relu (trước >>nb)
  - ✅ after_gap
  - ✅ logits_fc (INT32)
  - ✅ predicted_class
- **Đầu ra**: Thư mục `results/golden/` (13 files)
- **Metadata**: `golden_meta.json` - sample info, expected class, etc.

### 6️⃣ Model Definition (`model/model.py`) ✅
- **Trạng thái**: Hoàn thành
- **Dòng code**: 306 lines
- **Chứa các class**:
  - `ECG_1DCNN` - Base float32 model
  - `ECG_1DCNN_Q88` - Q8.8 simulation variant
  - `ECG_1DCNN_INT8` - INT8 variant
  - `build_model()` - Factory function
- **Tính năng**: ReLU chỉ sau Conv4 (preserve negative ECG features)

### 7️⃣ Dataset Utils (`utils/dataset.py`) ✅
- **Trạng thái**: Hoàn thành
- **Dòng code**: 238 lines
- **Chức năng**:
  - Chapman ECG dataset loader
  - `get_dataloaders()` → (train, val, test)
  - Batch format: `(ecg, label, hr)`
  - Class mapping: AFIB(0), GSVT(1), SB(2), SR(3)

### 8️⃣ Evaluation Metrics (`utils/evaluate.py`) ✅
- **Trạng thái**: Hoàn thành
- **Dòng code**: 106 lines
- **Chức năng**:
  - `evaluate_model()` - Inference loop
  - `compute_metrics()` - Accuracy, F1-macro, recall per class
  - `print_classification_report()` - Console output

---

## 📊 Kết Quả Cuối Cùng

### Model Performance Summary
```
┌─────────────────────────────────────┬──────────┬───────────┐
│ Model                               │ Accuracy │ F1-macro  │
├─────────────────────────────────────┼──────────┼───────────┤
│ Float32 Baseline                    │ 95.21%   │ 0.9462    │
│ Pruned (float, no fine-tune)        │ 37.75%   │ 0.3144    │
│ Pruned + Fine-tuned (float)         │ 93.80%   │ 0.9309    │
│ Pruned + Fine-tuned (Q8.8)          │ 94.08%   │ 0.9340    │
│ Pruned + Fine-tuned (INT8)          │ 93.89%   │ 0.9314    │
│ QAT INT8 (Float eval)               │ 94.84%   │ —         │
│ **QAT INT8 (Round-half-up)** ⭐     │ **94.65%** │ **0.9404** │
└─────────────────────────────────────┴──────────┴───────────┘
```

### Architecture Details
```
Input: 2500 samples (INT8)
  ↓
Conv1 (1→4, K=5) + MaxPool(5,5) → 500×4
  ↓
Conv2 (4→4, K=5) + MaxPool(5,5) → 100×4
  ↓
Conv3 (4→8, K=5) + MaxPool(5,5) → 20×8
  ↓
Conv4 (8→8, K=5, ReLU) + MaxPool(5,5) → 4×8
  ↓
GlobalAveragePool → 8
  ↓
FC (8→4)
  ↓
Argmax → Class (0-3)

Parameters: 640 (after pruning from 1244)
```

---

## ⚠️ Lưu Ý Quan Trọng

1. **flat_weights.hex**: ⚠️ **KHÔNG CÓ COMMENT LINES** - $readmemh đọc từ byte 0
2. **QAT là pipeline duy nhất cho hardware** - PTQ trên pruned model bị lỗi (~22% acc)
3. **Rounding**: round-half-up (`acc + 2^(nb-1)) >> nb`), **KHÔNG phải floor**
4. **ReLU positioning**: Chỉ sau Conv4 - bảo tồn các feature âm của ECG
5. **Output channels**: 4,4,8,8 (power-of-2) ✅ Đã match với target

---

## 🔧 Các Lệnh Chạy

```bash
cd /home/duc/Thesis/software/python

# 1. Training
python3 train.py --data_dir /home/duc/Thesis/data/Chapman

# 2. Pruning
python3 prune_finetune.py --checkpoint ./results/best_model.pth \
    --data_dir /home/duc/Thesis/data/Chapman

# 3. QAT INT8
python3 quantization/qat_int8.py --checkpoint ./results/best_model_pruned.pth \
    --output_dir ./results/qat_int8 --data_dir /home/duc/Thesis/data/Chapman

# 4. Export weights
python3 export_weights_int8.py \
    --checkpoint ./results/qat_int8/model_qat_int8.pth \
    --output_dir ./results/weights_qat_int8

# 5. Generate golden files
python3 generate_golden.py \
    --checkpoint ./results/qat_int8/model_qat_int8.pth \
    --data_dir /home/duc/Thesis/data/Chapman \
    --output_dir ./results/golden
```

---

## 📝 Trạng Thái TODO

### ✅ Đã Hoàn Thành
- [x] Training pipeline
- [x] Pruning & fine-tuning
- [x] QAT-INT8 quantization
- [x] Weight export to .hex/.mem
- [x] Golden reference files
- [x] Model architecture (4,4,8,8 channels)
- [x] Dataset loader
- [x] Evaluation metrics

### ⏳ Cần Làm Tiếp Theo (Hardware Stage)
- [ ] Update hardware/RTL/ cho channels mới (4,4,8,8)
- [ ] Verify simulation với golden files
- [ ] Synthesis trên Quartus
- [ ] Timing closure
- [ ] End-to-end validation trên DE10-Nano

---

## 📦 Checkpoint Locations

| Checkpoint | Path | Size | Updated |
|-----------|------|------|---------|
| Float32 Baseline | `results/best_model.pth` | 27 KB | 2026-05-09 |
| Pruned Model | `results/best_model_pruned.pth` | 6.6 KB | 2026-05-11 |
| QAT INT8 | `results/qat_int8/model_qat_int8.pth` | 9.8 KB | 2026-05-02 |
| Weights | `results/weights_qat_int8/flat_weights.hex` | 2.3 KB | 2026-04-29 |
| Golden Files | `results/golden/` | ~72 KB | 2026-04-29 |

---

## 🎓 Kết Luận

**Software pipeline hoàn toàn hoàn thành và sẵn sàng cho stage hardware.** Tất cả các lệnh chạy được, checkpoint được lưu, và golden reference files đã được sinh để kiểm chứng RTL.

Bước tiếp theo: **Cập nhật hardware RTL để match với architecture (4,4,8,8) và sử dụng weights từ `weights_qat_int8/` để verify simulation.**
