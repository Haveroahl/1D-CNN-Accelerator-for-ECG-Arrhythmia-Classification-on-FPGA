# Software Architecture: 1D-CNN ECG Classification

## 1. Project Context & Objectives
- **Target Application:** Real-time ECG rhythm classification (4 classes: AFIB, GSVT, SB, SR)
- **Target Platform:** Intel Cyclone V FPGA (HCMUS Electronics & Telecommunications project)
- **Input:** 2500-sample ECG data (sequence classification)
- **Output:** 4-class probability distribution / hard class decision

---

## 2. Model Architecture

### 2.1 Neural Network Design
**Model Name:** `ECG_1DCNN`

**Layers:**
| Layer | Type | Config | Output Shape | Purpose |
|-------|------|--------|--------------|---------|
| INPUT | - | 2500 samples | (2500,) | Raw ECG signal |
| CONV1 | Conv1d | kernel=5, out_channels=32, padding=2 | (32, 2500) | Feature extraction (low-level) |
| POOL1 | MaxPool1d | kernel=5, stride=5 | (32, 500) | Temporal downsampling |
| CONV2 | Conv1d | kernel=5, out_channels=64, padding=2 | (64, 500) | Feature extraction (mid-level) |
| POOL2 | MaxPool1d | kernel=5, stride=5 | (64, 100) | Temporal downsampling |
| CONV3 | Conv1d | kernel=5, out_channels=128, padding=2 | (128, 100) | Feature extraction (higher-level) |
| POOL3 | MaxPool1d | kernel=5, stride=5 | (128, 20) | Temporal downsampling |
| CONV4 | Conv1d | kernel=5, out_channels=256, padding=2 | (256, 20) | Feature extraction (deep) |
| POOL4 | MaxPool1d | kernel=5, stride=5 | (256, 4) | Temporal downsampling |
| GAP | GlobalAvgPool1d | - | (256,) | Temporal aggregation (divider-free: `>> 2` for 4 samples) |
| FC | Linear | in_features=256, out_features=4 | (4,) | Class logits |

**Total Parameters:** 1,244 (post-pruning: ~47.4% reduction)

### 2.2 Activation Functions
- **Conv1-3:** No ReLU (preserves negative clinical features: Q, S, T waves)
- **Conv4:** ReLU fused after quantization
- **FC:** No activation (logits output)

### 2.3 Key Design Decisions
1. **Kernel Size = 5:** Matches ECG physiological feature wavelengths
2. **Stride = 5 Pooling:** Explicit downsampling (hardware: on-the-fly max pooling)
3. **No Early ReLU:** Clinical ECG waveforms require negative values
4. **Global Average Pooling:** Reduces spatial dimension to single value per channel

---

## 3. Optimization Pipeline

### 3.1 Structured Pruning
- **Method:** Channel-level pruning with fine-tuning
- **Target:** 47.4% parameter reduction
- **Output:** Checkpoint at `./results/pruned_model.pth`
- **Script:** `v3_prune_finetune.py`

### 3.2 Quantization Schemes

#### Q8.8 (16-bit Fixed-Point)
- **Format:** 8 integer bits + 8 fractional bits (scale factor: 256)
- **Use Case:** High precision, intermediate testing
- **Bit Width:** 16-bit signed
- **Output:** `model_q88.pth`
- **Script:** `quantization/quantize_q88.py`

#### Q4.4 (8-bit Fixed-Point)
- **Format:** 4 integer bits + 4 fractional bits (scale factor: 16)
- **Use Case:** Mid-range precision, embedded systems
- **Bit Width:** 8-bit signed
- **Output:** `model_q44.pth`
- **Script:** `quantization/quantize_q44.py`

#### INT8 (Integer-Only)
- **Format:** 8-bit signed integer (no fractional bits, scale factor: 1)
- **Use Case:** Hardware deployment (FPGA)
- **Bit Width:** 8-bit signed
- **Accumulator:** INT32 (during computation)
- **Output:** `model_int8.pth`
- **Script:** `quantization/quantize_int8.py`
- **Special Feature:** Quantization-Aware Training (QAT) calibration

### 3.3 Workflow Pipeline
```
Original Model
     ↓
Structured Pruning (v3_prune_finetune.py)
     ↓
Pruned Model (pruned_model.pth)
     ↓
     ├─→ Q8.8 Quantization (quantize_q88.py)
     ├─→ Q4.4 Quantization (quantize_q44.py)
     └─→ INT8 Quantization (quantize_int8.py)
         ↓
      Comparison & Evaluation (evaluate_quantized.py)
```

---

## 4. Quantization-Hardware Interface

### 4.1 INT8 Quantization (Hardware-Relevant)
**Quantization Formula:**
```
quantized_value = round(float_value × scale) 
clipped_quantized_value = clip(quantized_value, -128, 127)
```

**For Weights (per-channel):**
- Scale per output channel determined during training
- Stored as INT8 in parameter ROM
- Dynamic range: [-128, 127] → ~2 bits per channel for scale

**For Activations (per-layer):**
- Scale per activation tensor (global)
- Used for input/output quantization
- Dynamic range: [-128, 127]

**Accumulation (32-bit):**
```
accumulator = 0 (INT32)
for i in range(kernel_size):
    partial_sum = weight[i] * input[i]  // Both INT8, result INT16
    accumulator += partial_sum          // INT32
// Requantize to INT8
output = (accumulator * scale_shift) >> shift_amount
output = clip(output, -128, 127)
```

### 4.2 Memory Layout for Hardware
- **Weights:** INT8 per-channel storage in ROM
- **Biases:** INT32 (pre-computed bias values)
- **Scales:** INT32 scaling factors per layer
- **Activations:** INT8 per-layer in SRAM

---

## 5. Directory Structure
```
software/
├── python/
│   ├── model/
│   │   ├── __init__.py
│   │   └── model.py                    # ECG_1DCNN class definition
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── dataset.py                  # Data loading & preprocessing
│   │   └── evaluate.py                 # Metrics computation
│   ├── quantization/
│   │   ├── __init__.py
│   │   ├── base_quantizer.py           # Abstract base class
│   │   ├── quantize_q88.py             # Q8.8 quantization
│   │   ├── quantize_q44.py             # Q4.4 quantization
│   │   ├── quantize_int8.py            # INT8 quantization (QAT)
│   │   └── evaluate_quantized.py       # Comparison tool
│   ├── train.py                        # Training script
│   ├── v3_prune_finetune.py            # Pruning script
│   ├── interpretability_v3.py           # Feature visualization
│   ├── export_weights_q88.py            # Weight export utility
│   ├── USAGE.md                        # Usage guide
│   └── SUMMARY.md                      # Implementation summary
├── results_v3/                         # Pre-computed results
├── plots/                              # Visualization outputs
└── ARCHITECTURE.md                     # This file
```

---

## 6. Usage Examples

### 6.1 Training
```bash
python train.py \
    --data_dir /home/duc/Thesis/data/Chapman \
    --output_dir ./results \
    --epochs 50 \
    --batch_size 32
```

### 6.2 Pruning
```bash
python v3_prune_finetune.py \
    --checkpoint ./results/best_model.pth \
    --output_dir ./results/pruned
```

### 6.3 Quantization (INT8 for Hardware)
```bash
python quantization/quantize_int8.py \
    --checkpoint ./results/pruned/model.pth \
    --output_dir ./results/int8_pruned \
    --calibration_data /home/duc/Thesis/data/Chapman/train
```

### 6.4 Export for Hardware
```bash
python export_weights_int8.py \
    --checkpoint ./results/int8_pruned/model_int8.pth \
    --output_dir ./results/weights_int8_pruned
```

---

## 7. Interface Specifications (Software → Hardware)

### 7.1 Input Interface
- **Format:** 2500 INT8 samples (signed)
- **Range:** [-128, 127]
- **Protocol:** Avalon-ST streaming with valid/ready handshake
- **Timing:** Samples streamed one per cycle

### 7.2 Output Interface
- **Format:** Single INT8 value (class ID)
- **Range:** [0, 3]
  - 0: Normal Sinus Rhythm (SR)
  - 1: Atrial Fibrillation (AFIB)
  - 2: Supraventricular Tachycardia (GSVT)
  - 3: Slow Bradycardia (SB)
- **Protocol:** Avalon-ST with valid signal
- **Timing:** One output per 2500 sample batch

### 7.3 Parameter Format
All parameters stored in hardware ROM:

**Weights:**
- Format: INT8 per-channel
- Size: ~1,244 × 1 byte = ~1.2 KB
- Organization: CONV1 weights, CONV2 weights, ..., FC weights

**Biases:**
- Format: INT32 per-channel
- Size: ~(32 + 64 + 128 + 256 + 4) × 4 bytes = ~1.9 KB

**Scales:**
- Format: INT32 per-layer quantization scale
- Size: ~10 layers × 4 bytes = ~40 bytes

**Total ROM Size:** ~3.1 KB + metadata

### 7.4 Internal SRAM Requirements
- **Ping-Pong Banks:** For intermediate feature maps
- **CONV1 output:** 32 × 2500 = 80 KB (reduced by pooling)
- **CONV2 output:** 64 × 500 = 32 KB
- **CONV3 output:** 128 × 100 = 12.8 KB
- **CONV4 output:** 256 × 4 = 1 KB
- **Total active:** ~126 KB (dual-bank = 252 KB SRAM)

---

## 8. Testing & Verification

### 8.1 Bit-Exact Testing
Before deploying to hardware:
```bash
python quantization/evaluate_quantized.py \
    --float_ckpt ./results/best_model.pth \
    --int8_ckpt ./results/int8_pruned/model_int8.pth \
    --test_data /home/duc/Thesis/data/Chapman/test
```

Expected output: Per-sample INT8 values match hardware simulation.

### 8.2 Accuracy Benchmarks
- **Float32:** Baseline accuracy
- **Q8.8:** Should be ~99.5% of float32
- **Q4.4:** Should be ~95-98% of float32
- **INT8:** Should be ~95-97% of float32

---

## 9. Performance Summary

| Metric | Value |
|--------|-------|
| Input Samples | 2500 |
| Model Parameters (pruned) | ~1,244 |
| Quantization Format | INT8 weights, INT32 accum |
| Est. Cycles @ 50 MHz | ~4,747 |
| Est. Latency @ 50 MHz | ~95 µs |
| Est. Latency @ 100 MHz | ~47.5 µs |
| FPGA Platform | Intel Cyclone V |

---

## 10. Version Control & Documentation
- **SUMMARY.md:** Quick reference for code changes
- **USAGE.md:** Step-by-step execution guide
- **ARCHITECTURE.md:** This file—design specifications
