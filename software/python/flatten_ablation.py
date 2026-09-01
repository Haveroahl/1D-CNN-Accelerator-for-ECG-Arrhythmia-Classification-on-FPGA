"""
Flatten-head ablation (design-progression upstream baseline)
============================================================
Standalone experiment — does NOT modify any existing file.

Goal: establish the FIRST link of the design progression
    Flatten (this file)  ->  GAP  ->  prune  ->  INT8
so the paper can quantify what swapping the heavy Flatten head for
Global Average Pooling costs/saves (accuracy vs FC params).

Backbone is identical to model.model.ECG_1DCNN (unpruned, 4-8-8-16),
only the classifier head differs:

    Pool4 (B, 16, 4)
        GAP head     :  GAP -> (B, 16) -> FC(16->4)     # existing ECG_1DCNN
        Flatten head :  flatten -> (B, 64) -> FC(64->4) # this file

Float32 only. Not exported to hardware (no QAT, no bit-exact).

Usage (from software/python, venv active):
    python flatten_ablation.py
    python flatten_ablation.py --epochs 100 --output_dir ./results/flatten_ablation
"""

import os
import sys
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))

from utils.dataset import get_dataloaders, CLASS_NAMES
from utils.evaluate import evaluate_model, compute_metrics, print_classification_report
from model.model import ECG_1DCNN


# ============================================================
#  Flatten-head model (subclass — existing model.py untouched)
# ============================================================

class ECG_1DCNN_Flatten(ECG_1DCNN):
    """
    Same 4-8-8-16 backbone as ECG_1DCNN, but the head keeps the full
    Pool4 feature map (16 ch x 4 samples = 64) via flatten instead of
    averaging each channel with GAP.

    FC params: 64*4 + 4 = 260   (vs GAP head 16*4 + 4 = 68).
    """

    def __init__(self, num_classes=4, input_length=2500):
        super().__init__(num_classes=num_classes, input_length=input_length)
        self.gap = None                              # drop GAP head
        self.fc = nn.Linear(16 * 4, num_classes, bias=True)   # 64 -> 4

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))                # (B,  4, 500)  no ReLU
        x = self.pool2(self.conv2(x))                # (B,  8, 100)  no ReLU
        x = self.pool3(self.conv3(x))                # (B,  8,  20)  no ReLU
        x = self.pool4(F.relu(self.conv4(x)))        # (B, 16,   4)  +ReLU
        x = x.flatten(1)                             # (B, 64)
        return self.fc(x)                            # (B,  4)


# ============================================================
#  Train / eval (mirrors train.py loop, float32 only)
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for ecg, labels, _ in loader:
        ecg    = ecg.unsqueeze(1).float().to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out  = model(ecg)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * ecg.size(0)
        correct    += out.detach().argmax(1).eq(labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
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


def main(args):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Device: {device}  seed: {seed}")

    print("\n[1/3] Loading Chapman ...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    print("\n[2/3] Training Flatten-head model (4-8-8-16, FC 64->4) ...")
    model = ECG_1DCNN_Flatten(num_classes=4, input_length=2500).to(device)
    n_params = model.count_parameters()
    print(f"  Total parameters: {n_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    half = max(1, args.epochs // 2)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[half], gamma=0.1)

    best_val_loss = float('inf')
    best_epoch    = 0
    ckpt_path     = os.path.join(args.output_dir, 'flatten_best.pth')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        is_best = va_loss < best_val_loss
        if is_best:
            best_val_loss = va_loss
            best_epoch    = epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': va_loss, 'val_acc': va_acc}, ckpt_path)

        print(f"  Epoch {epoch:3d}/{args.epochs}"
              f"  Train {tr_loss:.4f}/{tr_acc:.4f}"
              f"  Val {va_loss:.4f}/{va_acc:.4f}"
              f"  {elapsed:.1f}s{' *' if is_best else ''}")
        scheduler.step()

    print(f"\n  Best checkpoint: epoch {best_epoch}  val_loss={best_val_loss:.4f}")

    print("\n[3/3] Evaluating best checkpoint on test set (float32) ...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    preds, labels = evaluate_model(model, test_loader, device)
    m = compute_metrics(preds, labels, CLASS_NAMES)
    print_classification_report(m, title="Flatten head (float32, 4-8-8-16, FC 64->4)")

    results = {
        'variant':    'flatten_head',
        'backbone':   '4-8-8-16 (unpruned)',
        'head':       'flatten -> FC(64->4)',
        'params':     int(n_params),
        'best_epoch': best_epoch,
        'float': {
            'accuracy': m['accuracy'],
            'f1_macro': m['f1_macro'],
            'confusion_matrix': m['confusion_matrix'].tolist(),
        },
    }
    out_path = os.path.join(args.output_dir, 'flatten_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] params={n_params}  "
          f"acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}")
    print(f"       Saved: {out_path}")
    print("\n  Compare against GAP head (existing, from PROJECT.md):")
    print("    GAP float32 unpruned (4-8-8-16, FC 16->4) ~94.8% acc, ~1244 params")
    print("  -> Flatten vs GAP delta is the first link of the design progression.")


def parse_args():
    p = argparse.ArgumentParser(description='Flatten-head ablation (float32, Chapman)')
    p.add_argument('--data_dir',    type=str, default='../../data/Chapman')
    p.add_argument('--output_dir',  type=str, default='./results/flatten_ablation')
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=128)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--num_workers', type=int,   default=2)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
