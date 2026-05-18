# Quantization-Aware Training (QAT) Quick Reference

## What is QAT?

QAT (Quantization-Aware Training) fine-tunes a trained model with simulated quantization during forward passes. This allows weights to learn representations that fit better within the INT8 range [-128, 127].

### Key Concepts

1. **Fake Quantization**: During forward pass, weights are quantized (rounded/clipped) then dequantized back to float
2. **Straight-Through Estimator (STE)**: Gradients flow through as-is during backprop (no quantization gradient)
3. **Per-Layer Dynamic Scaling**: Each layer gets its own quantization scale factor computed from weight statistics
4. **Fixed Scales**: Scales are computed once from baseline weights and kept fixed during training (hardware-friendly)

## Usage

### Basic Command

```bash
cd /home/duc/Thesis/software/python

# Run QAT with default settings (20 epochs, LR=1e-4)
./.venv/bin/python quantization/qat_training.py \
    --checkpoint ./results/best_model.pth \
    --output_dir ./results/qat_int8

# OR with custom parameters
./.venv/bin/python quantization/qat_training.py \
    --checkpoint ./results/best_model.pth \
    --output_dir ./results/qat_int8 \
    --num_epochs 30 \
    --learning_rate 5e-5 \
    --batch_size 128 \
    --num_workers 2
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | **required** | Path to trained baseline model checkpoint |
| `--output_dir` | `./results/qat_int8` | Directory to save QAT outputs |
| `--num_epochs` | `20` | Number of fine-tuning epochs |
| `--learning_rate` | `1e-4` | Learning rate (Adam optimizer) |
| `--data_dir` | `/home/duc/Thesis/data/Chapman` | Dataset directory |
| `--batch_size` | `128` | Training batch size |
| `--num_workers` | `2` | Dataset loading workers |

## Output Files

After QAT training completes, the following files are generated:

```
./results/qat_int8/
├── model_qat_int8.pth          # QAT-trained checkpoint (weights + scales)
├── qat_log.csv                 # Per-epoch metrics (CSV format)
├── qat_history.json            # Full training history (JSON)
└── QAT_SUMMARY.md              # This summary report
```

### Checkpoint Structure

```python
checkpoint = {
    'model_state_dict': {...},      # Learned float32 weights (fit INT8 range)
    'scales': {'conv1': 0.007002, ...},  # Per-layer quantization scales
    'quantization': 'INT8-QAT',
    'bit_width': 8,
    'test_acc': 0.9465,
    'val_acc_best': 0.9324,
    'val_loss_best': 0.1814,
}
```

## Evaluation

### Evaluate QAT Model Alone

```bash
# Use standard model evaluation
./.venv/bin/python -c "
import torch
from model.model import ECG_1DCNN
from utils.dataset import get_dataloaders
from utils.evaluate import evaluate_model, compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('./results/qat_int8/model_qat_int8.pth', map_location=device, weights_only=False)
model = ECG_1DCNN(num_classes=4).to(device)
model.load_state_dict(ckpt['model_state_dict'])

_, _, test_loader = get_dataloaders('/home/duc/Thesis/data/Chapman')
preds, labels = evaluate_model(model, test_loader, device)
metrics = compute_metrics(preds, labels, ['AFIB', 'GSVT', 'SB', 'SR'])
print(f'Test Accuracy: {metrics[\"accuracy\"]:.4f}')
print(f'F1-macro: {metrics[\"f1_macro\"]:.4f}')
"
```

### Compare All Quantization Methods

```bash
# Compare float32, Q8.8, Q4.4, INT8, and QAT
./.venv/bin/python quantization/evaluate_quantized.py \
    --float_ckpt ./results/best_model.pth \
    --q88_ckpt ./results/q88/model_q88.pth \
    --q44_ckpt ./results/q44/model_q44.pth \
    --int8_ckpt ./results/int8/model_int8.pth \
    --qat_ckpt ./results/qat_int8/model_qat_int8.pth \
    --data_dir /home/duc/Thesis/data/Chapman
```

## Results Summary (Current QAT Run)

```
Method         Bit Width    Test Acc    Δ from Float32    Memory
──────────────────────────────────────────────────────────────────
Float32        32           94.84%      baseline          ~5.0 KB
Q8.8           16           94.84%      0.00%             ~2.5 KB
INT8 (QAT)     8            94.65%      -0.19%            ~1.2 KB
```

## Training Dynamics

### Fake Quantization Process

For each forward pass:
1. **Quantize**: `q = round(w / scale)` → clamp to [-128, 127]
2. **Dequantize**: `w_q = q * scale`
3. **STE**: Output `w + (w_q - w).detach()` (gradient flows from original weight)

### Learning Rate Schedule

Default schedule drops learning rate by 10× at 70% of epochs:
- Epochs 1-14: LR = 1e-4
- Epochs 15-20: LR = 1e-5

This is implemented via `torch.optim.lr_scheduler.MultiStepLR`.

### Convergence Monitoring

The script monitors validation loss and saves the best-epoch model:

```
[  1/20] ✓ TrLoss=0.1324 ValLoss=0.1826 ← IMPROVED
[ 13/20] ✓ TrLoss=0.1306 ValLoss=0.1814 ← BEST (saved)
[ 20/20]   TrLoss=0.1282 ValLoss=0.1817 ← FINAL
```

## Customization

### Use Fixed Scale for All Layers (Simpler Hardware)

Edit `qat_training.py` line ~130:
```python
# Current: compute per-layer scales
scales_dict = compute_per_layer_scales(float_model, num_bits=8)

# Alternative: use fixed scale = 256 (Q8.8-like)
scales_dict = {name: 256.0 for name, _ in float_model.named_parameters()}
```

### Train Longer

Increase `--num_epochs`:
```bash
./.venv/bin/python quantization/qat_training.py \
    --checkpoint ./results/best_model.pth \
    --output_dir ./results/qat_int8_longer \
    --num_epochs 50  # Instead of default 20
```

### Use Different Learning Rate Schedule

Edit `qat_training.py` around line ~180 to modify the scheduler:
```python
# Current: drop LR at 70% of epochs
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[int(num_epochs * 0.7)], gamma=0.1
)

# Alternative: constant LR
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)

# Alternative: cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
```

## Troubleshooting

### QAT makes accuracy worse

- Reduce learning rate: `--learning_rate 5e-5`
- Increase number of epochs: `--num_epochs 50`
- Check that baseline model is trained well (>93% validation accuracy)

### Out of memory during QAT

- Reduce batch size: `--batch_size 64`
- Use CPU if GPU is full: Set `device = torch.device('cpu')` in script

### QAT model not compatible with evaluate_quantized.py

- Ensure the saved checkpoint includes `model_state_dict` with keys matching `ECG_1DCNN`
- The extraction code at line ~195 handles this automatically

## Hardware Integration

### Extract Quantization Scales for Hardware

```python
import torch
ckpt = torch.load('./results/qat_int8/model_qat_int8.pth')
scales = ckpt['scales']

# Print in hardware-friendly format
for layer, scale in scales.items():
    print(f"{layer}: scale_q8 = {int(scale * 256)} / 256")
    # scale_q8 is the fixed-point representation
```

### Implement INT8 Inference

For FPGA/hardware implementation:
1. Load learned float32 weights from QAT model
2. Apply per-layer quantization using saved scales
3. Run INT8 MACs
4. Dequantize output using same scales

See `quantization/fake_quantize.py` for reference implementation.

## References

- QAT Paper: Jacob et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (CVPR 2018)
- Straight-Through Estimator: Bengio et al. "Estimating or Propagating Gradients Through Stochastic Neurons" (NIPS 2013)
- PyTorch QAT Guide: https://pytorch.org/docs/stable/quantization.html

---

**Last Updated**: 2026-04-27  
**Tested On**: ECG_1DCNN (1,244 params), Chapman ECG Dataset  
**Status**: ✓ Production Ready
