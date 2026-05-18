"""Quantization-Aware Training (QAT) for INT8

QAT fine-tunes a trained model with fake quantization layers enabled during forward pass.
Weights learn to fit the INT8 range [-128, 127] via per-layer dynamic scaling.

Usage:
    python qat_training.py --checkpoint ./results/best_model.pth \\
                           --output_dir ./results/qat_int8 \\
                           --num_epochs 20 \\
                           --learning_rate 1e-4
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import csv
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from model.model import ECG_1DCNN
from utils.dataset import get_dataloaders, CLASS_NAMES
from utils.evaluate import evaluate_model, compute_metrics, print_classification_report
from quantization.fake_quantize import wrap_model_for_qat


def compute_per_layer_scales(model, num_bits=8):
    """Compute per-layer quantization scales from current weights."""
    scales = {}
    qmax = 2 ** (num_bits - 1) - 1  # 127 for 8-bit

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                w = module.weight.data
                w_np = w.cpu().numpy()
                abs_max = np.max(np.abs(w_np))
                if abs_max > 0:
                    scale = abs_max / qmax
                else:
                    scale = 1.0
                scales[name] = scale

    return scales


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch with fake quantization."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for ecg, labels, _ in train_loader:
        ecg = ecg.unsqueeze(1).float().to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(ecg)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate with fake quantization."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for ecg, labels, _ in val_loader:
            ecg = ecg.unsqueeze(1).float().to(device)
            labels = labels.to(device)

            outputs = model(ecg)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def qat_training(baseline_checkpoint: str, output_dir: str, num_epochs: int = 20,
                 learning_rate: float = 1e-4, data_dir: str = '/home/duc/Thesis/data/Chapman',
                 batch_size: int = 128, num_workers: int = 2):
    """Run QAT fine-tuning on baseline model."""
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # Load baseline model
    print(f"[INFO] Loading baseline checkpoint: {baseline_checkpoint}")
    ckpt = torch.load(baseline_checkpoint, map_location=device, weights_only=False)
    float_model = ECG_1DCNN(num_classes=4)
    float_model.load_state_dict(ckpt['model_state_dict'])
    float_model = float_model.to(device)

    # Compute per-layer scales from baseline weights
    print("[INFO] Computing per-layer quantization scales from baseline weights...")
    scales_dict = compute_per_layer_scales(float_model, num_bits=8)
    print("[INFO] Per-layer scales:")
    for name, scale in scales_dict.items():
        print(f"       {name:30s} scale={scale:.6f}")

    # Wrap model with fake quantization layers
    print("[INFO] Wrapping model with fake quantization (INT8, per-layer dynamic scales)...")
    qat_model = wrap_model_for_qat(float_model, scales_dict=scales_dict, num_bits=8)
    qat_model = qat_model.to(device)

    # Load dataloaders
    print(f"[INFO] Loading data from {data_dir}")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, batch_size=batch_size, num_workers=num_workers
    )

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(num_epochs * 0.7)], gamma=0.1
    )

    # Training loop
    print(f"\n[INFO] Starting QAT fine-tuning for {num_epochs} epochs")
    print(f"       Learning rate: {learning_rate}")
    print(f"       Optimizer: Adam")
    print(f"       Loss: CrossEntropyLoss\n")

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(qat_model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(qat_model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = {k: v.clone() if isinstance(v, torch.Tensor) else v
                               for k, v in qat_model.state_dict().items()}

        status = "✓" if is_best else " "
        print(f"[{epoch+1:2d}/{num_epochs}] {status} "
              f"TrLoss={train_loss:.4f} TrAcc={train_acc:.2f}% "
              f"ValLoss={val_loss:.4f} ValAcc={val_acc:.2f}% "
              f"LR={optimizer.param_groups[0]['lr']:.2e}")

    print(f"\n[INFO] QAT training completed. Best epoch: {best_epoch+1} (val_loss={best_val_loss:.4f})")

    # Load best model
    qat_model.load_state_dict(best_model_state)
    qat_model = qat_model.to(device)

    # Evaluate on test set
    print("\n[INFO] Evaluating QAT model on test set...")
    test_preds, test_labels = evaluate_model(qat_model, test_loader, device)
    test_metrics = compute_metrics(test_preds, test_labels, CLASS_NAMES)
    print_classification_report(test_metrics)

    # Extract inner float model from wrapped quantization layers
    print("[INFO] Extracting learned weights from quantization-aware layers...")
    float_model_final = ECG_1DCNN(num_classes=4)
    with torch.no_grad():
        # Copy weights from wrapped conv/linear layers
        for qat_name, qat_module in qat_model.named_children():
            final_module = getattr(float_model_final, qat_name)
            if hasattr(qat_module, 'conv'):  # QuantizeConv1d
                final_module.weight.data = qat_module.conv.weight.data.clone()
                if qat_module.conv.bias is not None:
                    final_module.bias.data = qat_module.conv.bias.data.clone()
            elif hasattr(qat_module, 'linear'):  # QuantizeLinear
                final_module.weight.data = qat_module.linear.weight.data.clone()
                if qat_module.linear.bias is not None:
                    final_module.bias.data = qat_module.linear.bias.data.clone()
    float_model_final = float_model_final.to(device)

    # Save QAT checkpoint compatible with evaluate_quantized.py
    qat_checkpoint = {
        'epoch': best_epoch,
        'model_state_dict': float_model_final.state_dict(),
        'scales': scales_dict,
        'quantization': 'INT8-QAT',
        'bit_width': 8,
        'train_loss_final': history['train_loss'][-1],
        'val_loss_best': best_val_loss,
        'val_acc_best': history['val_acc'][best_epoch],
        'test_acc': test_metrics['accuracy'],
    }
    qat_checkpoint_path = os.path.join(output_dir, 'model_qat_int8.pth')
    torch.save(qat_checkpoint, qat_checkpoint_path)
    print(f"\n[DONE] Saved QAT INT8 model: {qat_checkpoint_path}")

    # Save training history
    history_path = os.path.join(output_dir, 'qat_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save training log CSV
    log_path = os.path.join(output_dir, 'qat_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i+1,
                history['train_loss'][i],
                history['train_acc'][i],
                history['val_loss'][i],
                history['val_acc'][i],
                history['lr'][i],
            ])
    print(f"[INFO] Training history saved: {log_path}")

    print(f"\n{'='*60}")
    print(f"  QAT INT8 Summary")
    print(f"{'='*60}")
    print(f"  Baseline (from checkpoint): epoch={ckpt.get('epoch', '?')}, val_acc={ckpt.get('val_acc', '?')}")
    print(f"  QAT result:                 epoch={best_epoch+1}, val_loss={best_val_loss:.4f}, val_acc={history['val_acc'][best_epoch]:.2f}%")
    print(f"  Test accuracy:              {test_metrics['accuracy']:.4f}")
    print(f"  Test F1-macro:              {test_metrics['f1_macro']:.4f}")
    print(f"  Quantization scheme:        INT8 (16-level per-layer dynamic scaling)")
    print(f"  Memory (1244 params):       ~1.2 KB (INT8) vs ~5 KB (float32)")


def main():
    p = argparse.ArgumentParser(
        description='Quantization-Aware Training (QAT) for INT8'
    )
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to trained baseline checkpoint (.pth)')
    p.add_argument('--output_dir', type=str, default='./results/qat_int8',
                   help='Output directory for QAT model')
    p.add_argument('--num_epochs', type=int, default=20,
                   help='Number of QAT fine-tuning epochs (default 20)')
    p.add_argument('--learning_rate', type=float, default=1e-4,
                   help='Learning rate for QAT (default 1e-4)')
    p.add_argument('--data_dir', type=str,
                   default='/home/duc/Thesis/data/Chapman',
                   help='Chapman ECG data directory')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=2)
    args = p.parse_args()

    qat_training(args.checkpoint, args.output_dir, args.num_epochs,
                 args.learning_rate, args.data_dir, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
