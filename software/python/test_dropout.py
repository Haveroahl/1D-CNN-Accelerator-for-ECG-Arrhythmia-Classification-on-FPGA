"""
Test: Does adding dropout help the FC head?
============================================
Builds on test_hidden_layer.py finding (hidden head overfits slightly).

Configs compared (same seed / split / recipe):
  base                         : GAP(->16) -> FC(16->4)
  base + dropout(p)            : GAP(->16) -> Dropout(p) -> FC(16->4)
  hidden                       : GAP(->16) -> FC(16->32) -> ReLU -> FC(32->4)
  hidden + dropout(p)          : GAP(->16) -> FC(16->32) -> ReLU -> Dropout(p) -> FC(32->4)

Dropout swept over p in {0.2, 0.3, 0.5}.
Float32 only. Does NOT touch model.py or the pruned hardware model.

Usage:
  python test_dropout.py [--epochs 100] [--hidden 32]
"""

import os
import sys
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))

from utils.dataset import get_dataloaders, CLASS_NAMES
from utils.evaluate import evaluate_model, compute_metrics
from model.model import ECG_1DCNN


# ============================================================
#  Variants — identical conv feature extractor, different heads
# ============================================================

class ECG_1DCNN_Head(ECG_1DCNN):
    """ECG_1DCNN with a configurable FC head (optional hidden layer + dropout)."""

    def __init__(self, num_classes=4, input_length=2500, hidden=0, dropout=0.0):
        super().__init__(num_classes=num_classes, input_length=input_length)
        layers = []
        if hidden > 0:
            layers += [nn.Linear(16, hidden, bias=True), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, num_classes, bias=True))
        else:
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(16, num_classes, bias=True))
        self.fc = nn.Sequential(*layers)

    # forward() inherited: ...gap(x).squeeze(-1) -> self.fc(x)


# ============================================================
#  Training (mirrors train.py: Adam, MultiStepLR @ half)
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for ecg, labels, _ in loader:
        ecg    = ecg.unsqueeze(1).float().to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(ecg), labels)
        loss.backward()
        optimizer.step()


def validate_loss(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for ecg, labels, _ in loader:
            ecg    = ecg.unsqueeze(1).float().to(device)
            labels = labels.to(device)
            loss = criterion(model(ecg), labels)
            total_loss += loss.item() * ecg.size(0)
            total      += labels.size(0)
    return total_loss / total


def fit(model, train_loader, val_loader, test_loader, device, epochs, lr, tag):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    half = max(1, epochs // 2)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[half], gamma=0.1)

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss = validate_loss(model, val_loader, criterion, device)
        scheduler.step()
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    preds, labels = evaluate_model(model, test_loader, device)
    m = compute_metrics(preds, labels, CLASS_NAMES)
    print(f"  [{tag:<22}] best_val_loss={best_val_loss:.4f}  "
          f"acc={m['accuracy']*100:.2f}  f1={m['f1_macro']:.4f}")
    return {'tag': tag, 'params': model.count_parameters(),
            'accuracy': m['accuracy'], 'f1_macro': m['f1_macro']}


def reseed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=r'd:\Thesis101\data\Chapman')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num_workers', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device} | epochs={args.epochs} | hidden={args.hidden}\n")

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    H = args.hidden
    DROPOUTS = [0.2, 0.3, 0.5]

    # (tag, hidden, dropout)
    configs = [('base', 0, 0.0)]
    configs += [(f'base+drop{p}', 0, p) for p in DROPOUTS]
    configs += [(f'hidden{H}', H, 0.0)]
    configs += [(f'hidden{H}+drop{p}', H, p) for p in DROPOUTS]

    results = []
    for tag, hid, drop in configs:
        reseed(42)
        model = ECG_1DCNN_Head(num_classes=4, input_length=2500,
                               hidden=hid, dropout=drop).to(device)
        results.append(fit(model, train_loader, val_loader, test_loader,
                           device, args.epochs, args.lr, tag))

    # ---- Report ----
    base_acc = results[0]['accuracy']
    base_f1  = results[0]['f1_macro']
    print("\n" + "=" * 64)
    print(f"{'Config':<22}{'Params':<9}{'Accuracy':<11}{'F1-macro':<11}{'dAcc%':<8}")
    print("-" * 64)
    for r in results:
        d_acc = (r['accuracy'] - base_acc) * 100
        print(f"{r['tag']:<22}{r['params']:<9}{r['accuracy']*100:<11.2f}"
              f"{r['f1_macro']:<11.4f}{d_acc:<+8.2f}")
    print("=" * 64)
    best = max(results, key=lambda r: r['accuracy'])
    print(f"\nBest config: {best['tag']} "
          f"(acc={best['accuracy']*100:.2f}, f1={best['f1_macro']:.4f})")
    if best['tag'] == 'base':
        print("=> Dropout / hidden layer do NOT beat the plain FC(16->4) baseline.")


if __name__ == "__main__":
    main()
