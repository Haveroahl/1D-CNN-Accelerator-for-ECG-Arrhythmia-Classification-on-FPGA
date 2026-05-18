# Quantization-Aware Training (QAT) for INT8 - Summary Report

## Overview

Quantization-Aware Training (QAT) was implemented to fine-tune the baseline ECG_1DCNN model with simulated INT8 quantization during training. The goal was to improve model accuracy when quantized to INT8 by allowing weights to learn representations that fit better within the INT8 range [-128, 127].

## QAT Training Configuration

| Parameter | Value |
|---|---|
| Baseline Checkpoint | `./results/best_model.pth` |
| Fine-tuning Epochs | 20 |
| Learning Rate (initial) | 1e-4 |
| Learning Rate (after epoch 14) | 1e-5 (scheduler at 70% epochs) |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Quantization Scheme | INT8 (8-bit signed, per-layer dynamic scaling) |
| Quantization Method | Straight-Through Estimator (STE) |

## Per-Layer Quantization Scales

Computed from baseline model weights at initialization:

| Layer | Scale | Bit Width | Range |
|---|---|---|---|
| Conv1 (1→4) | 0.007002 | 8-bit | [-128, 127] quantized to [-0.896, 0.889] |
| Conv2 (4→8) | 0.007637 | 8-bit | [-128, 127] quantized to [-0.977, 0.970] |
| Conv3 (8→8) | 0.007179 | 8-bit | [-128, 127] quantized to [-0.919, 0.912] |
| Conv4 (8→16) | 0.004393 | 8-bit | [-128, 127] quantized to [-0.562, 0.556] |
| FC (16→4) | 0.004353 | 8-bit | [-128, 127] quantized to [-0.557, 0.551] |

## Training Progress

The training showed stable convergence with fake quantization enabled:

```
[  1/20] ✓ TrLoss=0.1324 TrAcc=95.75% ValLoss=0.1826 ValAcc=93.24%
[ 13/20] ✓ TrLoss=0.1306 TrAcc=95.66% ValLoss=0.1814 ValAcc=93.24% ← BEST EPOCH
[ 20/20]   TrLoss=0.1282 TrAcc=95.85% ValLoss=0.1817 ValAcc=93.33%
```

**Best Epoch**: 13 with validation loss = 0.1814

## Test Set Accuracy Results

### QAT-INT8 Performance

```
Accuracy  : 0.9465 (94.65%)
F1-macro  : 0.9402

Per-Class Metrics:
┌──────┬───────────┬────────┬────────┬─────────┐
│Class │ Precision │ Recall │   F1   │ Support │
├──────┼───────────┼────────┼────────┼─────────┤
│AFIB  │   0.8712  │ 0.9312 │ 0.9002 │   218   │
│GSVT  │   0.9322  │ 0.9091 │ 0.9205 │   242   │
│SB    │   0.9817  │ 0.9895 │ 0.9856 │   380   │
│SR    │   0.9812  │ 0.9289 │ 0.9543 │   225   │
└──────┴───────────┴────────┴────────┴─────────┘
```

## Comparison with Other Quantization Methods

| Method | Bit Width | Test Accuracy | Δ from Baseline | Memory (1244 params) |
|---|---|---|---|---|
| **Float32 (Baseline)** | 32 | **94.84%** | - | ~5.0 KB |
| **Q8.8 (Fixed-point)** | 16 | **94.84%** | 0.00% | ~2.5 KB |
| **Q4.4 (Fixed-point)** | 8 | 94.65% | -0.19% | ~1.2 KB |
| **INT8 (Post-training)** | 8 | 94.65% | -0.19% | ~1.2 KB |
| **INT8 (QAT)** | 8 | 94.65% | -0.19% | ~1.2 KB |

## Key Findings

### 1. QAT Performance
- QAT achieved **94.65% test accuracy**, maintaining competitive performance with post-training INT8 quantization
- The accuracy drop from float32 (94.84%) is only **0.19%**, which is clinically acceptable for ECG classification
- Training was stable with 20 epochs and achieved convergence at epoch 13

### 2. Per-Layer Dynamic Scaling Effectiveness
- Per-layer scales were computed from baseline weights and kept fixed during QAT
- Fixed scales (rather than learnable) ensure hardware-friendly quantization with predictable ranges
- Weights learned to fit well within these fixed quantization bins

### 3. INT8 vs QAT Comparison
- Post-training INT8: 94.65%
- QAT-INT8: 94.65%
- Both methods achieved identical accuracy, suggesting that:
  - The per-layer dynamic scaling approach is already optimal for this model
  - Weights don't need further adjustment via QAT for INT8
  - Simple post-training quantization is sufficient for INT8 (cost-effective)

### 4. Q8.8 Superiority
- Q8.8 (16-bit) achieved **94.84% accuracy** (same as float32)
- This is the best quantization result, achieving **zero accuracy loss**
- For medical-critical applications, Q8.8 is the recommended quantization scheme
- Cost: ~2.5 KB per model vs 1.2 KB for INT8

## Hardware Implementation Implications

### For INT8 (QAT or Post-training)
```
Input Data (float32) → Quantize to INT8 → [INT8 MACs] → Dequantize → Output (float32)
Memory: ~1.2 KB (vs 5 KB float32) → 4.2× compression
Inference: INT8 multiply-accumulate operations (lower power, faster)
Accuracy: 94.65% (acceptable for ECG classification)
```

### For Q8.8 (Recommended)
```
Input Data (float32) → Quantize to Q8.8 → [Q8.8 MACs] → Dequantize → Output (float32)
Memory: ~2.5 KB (vs 5 KB float32) → 2× compression
Inference: Fixed-point MACs with scale=256
Accuracy: 94.84% (clinical-grade, zero loss from float32)
```

## Quantization Scheme Selection Recommendation

| Use Case | Recommended Scheme | Rationale |
|---|---|---|
| **Critical ECG analysis** | Q8.8 | Zero accuracy loss, hardware-efficient |
| **Medical IoT devices** | INT8 | Maximum compression (4.2×), acceptable accuracy |
| **Embedded systems** | INT8 | Minimal memory footprint, fast inference |
| **Research/validation** | Float32 → Q8.8 → INT8 | Compare trade-offs |

## Files Generated

1. **Model Checkpoint**: `./results/qat_int8/model_qat_int8.pth`
   - Contains QAT-trained weights and per-layer scales
   - Compatible with `evaluate_quantized.py` evaluation pipeline
   
2. **Training Log**: `./results/qat_int8/qat_log.csv`
   - Per-epoch metrics: train_loss, train_acc, val_loss, val_acc, learning_rate
   
3. **Training History**: `./results/qat_int8/qat_history.json`
   - Full training history in JSON format for analysis

## Computational Cost

- **QAT Fine-tuning Time**: ~2 minutes (20 epochs on GPU)
- **Evaluation Time**: ~5 seconds per model
- **Total Quantization Pipeline**: < 5 minutes end-to-end

## Conclusion

Quantization-Aware Training (QAT) for INT8 successfully trained the ECG_1DCNN model with fake quantization simulation. The trained model maintains 94.65% test accuracy with INT8 quantization, achieving 4.2× memory compression compared to float32.

However, post-training INT8 quantization achieves the same accuracy at significantly lower computational cost (no training). For this model and dataset, **post-training quantization to INT8 is recommended** for practical deployment.

For clinical-critical applications requiring zero accuracy loss, **Q8.8 quantization** should be used instead, which achieves 94.84% accuracy (same as float32) with 2× memory compression.

---

**Generated**: 2026-04-27  
**Model**: ECG_1DCNN (1,244 parameters)  
**Dataset**: Chapman ECG Database (10,646 recordings)  
**Hardware Target**: FPGA with INT8/Q8.8 MAC support
