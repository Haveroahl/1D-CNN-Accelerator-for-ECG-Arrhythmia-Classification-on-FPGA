"""
Test: Does adding a hidden FC layer to the classifier improve results?
=======================================================================
Baseline classifier  : GAP(->16) -> FC(16->4)
Hidden-layer variant : GAP(->16) -> FC(16->H) -> ReLU -> FC(H->4)

Everything else (conv feature extractor, training recipe, data split, seed)
is identical, so the only difference is the extra hidden FC layer.

Float32 only - answers "does a hidden layer change accuracy/F1?".
Does NOT touch model.py, the pruned hardware model, or quantization.

Usage:
  python test_hidden_layer.py [--epochs 100] [--hidden 32] [--batch_size 128]
"""

import os
import sys
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))

from utils.dataset import get_dataloaders, CLASS_NAMES
from utils.evaluate import evaluate_model, compute_metrics
from model.model import ECG_1DCNN


# ============================================================
#  Hidden-FC variant — identical feature extractor, 2-layer head
# ============================================================

class ECG_1DCNN_Hidden(ECG_1DCNN):
    """Base ECG_1DCNN with the single FC(16->4) replaced by FC(16->H)->ReLU->FC(H->4)."""

    def __init__(self, num_classes=4, input_length=2500, hidden=32):
        super().__init__(num_classes=num_classes, input_length=input_length)
        self.hidden = hidden
        # Replace the single-layer head with a 2-layer MLP head.
        self.fc = nn.Sequential(
            nn.Linear(16, hidden, bias=True),
            nn.ReLU(),
            nn.Linear(hidden, num_classes, bias=True),
        )

    # forward() inherited unchanged: ...self.gap(x).squeeze(-1) -> self.fc(x)
    # nn.Sequential head accepts the (B,16) tensor directly.


# ============================================================
#  Training (mirrors train.py recipe: Adam, MultiStepLR @ half)
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
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  [{tag}] epoch {epoch:3d}/{epochs}  val_loss={va_loss:.4f}  best={best_val_loss:.4f}")

    # Evaluate best checkpoint on the test set.
    model.load_state_dict(best_state)
    preds, labels = evaluate_model(model, test_loader, device)
    m = compute_metrics(preds, labels, CLASS_NAMES)
    return {
        'params':   model.count_parameters(),
        'accuracy': m['accuracy'],
        'f1_macro': m['f1_macro'],
        'per_class': m['per_class'],
    }


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

    # Same loaders for both models (split is fixed by the seed inside get_dataloaders).
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    print("[1/2] Baseline  GAP->FC(16->4) ...")
    reseed(42)
    base = ECG_1DCNN(num_classes=4, input_length=2500).to(device)
    base_res = fit(base, train_loader, val_loader, test_loader, device,
                   args.epochs, args.lr, tag='base')

    print(f"\n[2/2] Hidden    GAP->FC(16->{args.hidden})->ReLU->FC({args.hidden}->4) ...")
    reseed(42)
    hid = ECG_1DCNN_Hidden(num_classes=4, input_length=2500, hidden=args.hidden).to(device)
    hid_res = fit(hid, train_loader, val_loader, test_loader, device,
                  args.epochs, args.lr, tag='hidden')

    # ---- Report ----
    print("\n" + "=" * 60)
    print(f"{'Model':<28}{'Params':<10}{'Accuracy':<12}{'F1-macro':<10}")
    print("-" * 60)
    print(f"{'Baseline FC(16->4)':<28}{base_res['params']:<10}{base_res['accuracy']*100:<12.2f}{base_res['f1_macro']:<10.4f}")
    print(f"{f'Hidden FC(16->{args.hidden}->4)':<28}{hid_res['params']:<10}{hid_res['accuracy']*100:<12.2f}{hid_res['f1_macro']:<10.4f}")
    print("-" * 60)
    d_acc = (hid_res['accuracy'] - base_res['accuracy']) * 100
    d_f1  = hid_res['f1_macro'] - base_res['f1_macro']
    print(f"{'Delta (hidden - base)':<28}{hid_res['params']-base_res['params']:<+10}{d_acc:<+12.2f}{d_f1:<+10.4f}")
    print("=" * 60)
    verdict = "improves" if d_acc > 0.2 else ("hurts" if d_acc < -0.2 else "no meaningful change (<0.2%)")
    print(f"\nVerdict: adding a hidden FC layer {verdict}.")


if __name__ == "__main__":
    main()
