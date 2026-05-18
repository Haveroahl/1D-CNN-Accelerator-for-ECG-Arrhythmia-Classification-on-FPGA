"""
Training Pipeline for 1-D CNN ECG Classifier
=============================================
Dataset  : Chapman ECG (4 classes: AFIB / GSVT / SB / SR)
Model    : ECG_1DCNN variants (base / v2 / v3)
Optimizer: Adam  lr=1e-3 → 1e-4 (MultiStepLR, drops at epoch 50)
Loss     : CrossEntropyLoss
Epochs   : 100 (default), best checkpoint by lowest val loss
Batch    : 128

After training three accuracy figures are reported:
  1. Float model (fp32)
  2. Q8.8 model  — simulates exact FPGA weight quantization (×256 scale)

Usage:
  python train.py --data_dir /home/duc/Thesis/data/Chapman
                  --output_dir ./results
                  [--model base|v2|v3]
                  [--epochs 100] [--batch_size 128] [--lr 1e-3]
"""

import os
import sys
import argparse
import copy
import csv
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))

from utils.dataset  import get_dataloaders, CLASS_NAMES
from utils.evaluate import evaluate_model, compute_metrics, print_classification_report
from model.model import build_model, ECG_1DCNN_Q88, ECG_1DCNN_INT8

MODEL_REGISTRY = {
    'ecg': (build_model, ECG_1DCNN_Q88),
}


# ============================================================
#  Epoch helpers
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for ecg, labels, _ in loader:
        ecg    = ecg.unsqueeze(1).float().to(device)   # (B, 1, 2500)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(ecg)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * ecg.size(0)
        correct    += outputs.detach().argmax(1).eq(labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for ecg, labels, _ in loader:
            ecg    = ecg.unsqueeze(1).to(device)
            labels = labels.to(device)

            out  = model(ecg)
            loss = criterion(out, labels)

            total_loss += loss.item() * ecg.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


# ============================================================
#  Main training pipeline
# ============================================================

def train(args):
    # ── Set random seed for reproducibility ──────────────────
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Random seed: {seed}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────
    print("\n[1/4] Loading dataset ...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── 2. Model ─────────────────────────────────────────────
    print(f"\n[2/4] Building model ({args.model}) ...")
    build_fn, Q88Class = MODEL_REGISTRY[args.model]
    model = build_fn(num_classes=4, input_length=2500).to(device)
    print(model.get_layer_info())
    print(f"\nTotal parameters: {model.count_parameters()}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # LR schedule: ×0.1 at the halfway point (epoch 50 for 100 epochs)
    half = max(1, args.epochs // 2)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[half], gamma=0.1
    )

    # ── 3. Training loop ──────────────────────────────────────
    print(f"\n[3/4] Training for {args.epochs} epochs ...")

    history = {k: [] for k in
               ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1')}

    log_csv = os.path.join(args.output_dir, 'train_log.csv')
    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'train_loss', 'train_acc',
             'val_loss', 'val_acc', 'val_f1',
             'lr', 'time_s', 'is_best']
        )

    best_val_loss = float('inf')
    best_epoch    = 0
    ckpt_path     = os.path.join(args.output_dir, 'best_model.pth')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        va_loss, va_acc = validate(model, val_loader, criterion, device)

        # F1-macro on validation (one extra pass)
        vp, vl   = evaluate_model(model, val_loader, device)
        val_f1   = compute_metrics(vp, vl, CLASS_NAMES)['f1_macro']
        elapsed  = time.time() - t0
        cur_lr   = optimizer.param_groups[0]['lr']

        for k, v in zip(
            ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1'),
            (tr_loss, tr_acc, va_loss, va_acc, val_f1)
        ):
            history[k].append(v)

        is_best = va_loss < best_val_loss
        if is_best:
            best_val_loss = va_loss
            best_epoch    = epoch
            torch.save({
                'epoch':              epoch,
                'model_state_dict':   model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':           va_loss,
                'val_acc':            va_acc,
            }, ckpt_path)

        star = " *" if is_best else ""
        print(f"  Epoch {epoch:3d}/{args.epochs}"
              f"  Train {tr_loss:.4f}/{tr_acc:.4f}"
              f"  Val {va_loss:.4f}/{va_acc:.4f}"
              f"  F1 {val_f1:.4f}"
              f"  LR {cur_lr:.1e}"
              f"  {elapsed:.1f}s{star}")

        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch,
                f"{tr_loss:.6f}", f"{tr_acc:.6f}",
                f"{va_loss:.6f}", f"{va_acc:.6f}", f"{val_f1:.6f}",
                f"{cur_lr:.1e}", f"{elapsed:.1f}", int(is_best),
            ])

        scheduler.step()

    with open(os.path.join(args.output_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best checkpoint: epoch {best_epoch}  "
          f"val_loss={best_val_loss:.4f}")

    # ── 4. Evaluation ─────────────────────────────────────────
    print("\n[4/4] Evaluating on test set ...")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    # ---- Float model ----
    fp, fl   = evaluate_model(model, test_loader, device)
    m_float  = compute_metrics(fp, fl, CLASS_NAMES)
    print_classification_report(m_float, title="Float model (fp32)")

    # ---- Q8.8 model (simulates FPGA weight ROM, scale=256) ----
    q88_model = Q88Class(model).to(device)
    qp, ql    = evaluate_model(q88_model, test_loader, device)
    m_q88     = compute_metrics(qp, ql, CLASS_NAMES)
    print_classification_report(m_q88, title="Q8.8 model (FPGA weight ROM, ×256)")

    # ---- INT8 model (hardware integer simulation, power-of-2 scale) ----
    int8_model = ECG_1DCNN_INT8(model, calib_loader=train_loader, device=device)
    ip, il     = evaluate_model(int8_model, test_loader, device)
    m_int8     = compute_metrics(ip, il, CLASS_NAMES)
    print_classification_report(m_int8, title="INT8 model (hardware simulation)")

    # Save results
    results = {
        'model':      args.model,
        'best_epoch': best_epoch,
        'float': {
            'accuracy': m_float['accuracy'],
            'f1_macro': m_float['f1_macro'],
            'confusion_matrix': m_float['confusion_matrix'].tolist(),
        },
        'q88': {
            'accuracy': m_q88['accuracy'],
            'f1_macro': m_q88['f1_macro'],
            'confusion_matrix': m_q88['confusion_matrix'].tolist(),
        },
        'int8': {
            'accuracy': m_int8['accuracy'],
            'f1_macro': m_int8['f1_macro'],
            'confusion_matrix': m_int8['confusion_matrix'].tolist(),
        },
    }
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Saved to {args.output_dir}/")
    print(f"       Float: acc={m_float['accuracy']:.4f}  "
          f"F1={m_float['f1_macro']:.4f}")
    print(f"       Q8.8:  acc={m_q88['accuracy']:.4f}  "
          f"F1={m_q88['f1_macro']:.4f}")
    print(f"       INT8:  acc={m_int8['accuracy']:.4f}  "
          f"F1={m_int8['f1_macro']:.4f}")

    return history, results


# ============================================================
#  CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Train 1-D CNN (no bias, Q8.8) on Chapman ECG'
    )
    p.add_argument('--data_dir',    type=str,
                   default='/home/duc/Thesis/data/Chapman',
                   help='Path to Chapman ECG database directory')
    p.add_argument('--output_dir',  type=str, default='./results',
                   help='Directory for checkpoints and logs')
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=128)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--num_workers', type=int,   default=2)
    p.add_argument('--model',       type=str,   default='ecg',
                   choices=['ecg'],
                   help='Model variant (currently ECG_1DCNN)')
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
