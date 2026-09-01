"""
Residual-block ablation (float, from scratch)
=============================================
Question: does adding a double-conv residual block at Stage 2 and Stage 4
improve 4-class Chapman accuracy over the plain 4-conv baseline (94.65%)?

Baseline (deployed pruned):   channels 1->4->4->8->8
  S1: Conv1(1->4)                       -> Pool1
  S2: Conv2(4->4)                       -> Pool2
  S3: Conv3(4->8)                       -> Pool3
  S4: Conv4(8->8) +ReLU                 -> Pool4

Residual variant (this script):
  S1: Conv1(1->4)                       -> Pool1            [unchanged]
  S2: Conv2a(4->4) -> ReLU -> Conv2b(4->4) -> (+x) -> Pool2
  S3: Conv3(4->8)                       -> Pool3            [unchanged]
  S4: Conv4a(8->8) -> ReLU -> Conv4b(8->8) -> (+x) -> ReLU -> Pool4

Both models trained from scratch under identical conditions (same split,
same 2-phase LR schedule, same seed) for a fair comparison.

Usage:
  python run_residual_ablation.py
  python run_residual_ablation.py --epochs1 30 --epochs2 20
"""

import os
import sys
import copy
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset  import get_dataloaders, CLASS_NAMES
from utils.evaluate import evaluate_model, compute_metrics


# ============================================================
#  Models
# ============================================================

class ECG_Baseline(nn.Module):
    """Plain 4-conv pruned baseline (1->4->4->8->8), ReLU only after Conv4."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, 5, padding=2, bias=True)
        self.conv2 = nn.Conv1d(4, 4, 5, padding=2, bias=True)
        self.conv3 = nn.Conv1d(4, 8, 5, padding=2, bias=True)
        self.conv4 = nn.Conv1d(8, 8, 5, padding=2, bias=True)
        self.pool  = nn.MaxPool1d(5)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(8, num_classes, bias=True)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ECG_Res24(nn.Module):
    """
    Residual block at Stage 2 and Stage 4.
    S2: Conv2a -> ReLU -> Conv2b -> (+input) -> Pool2   (channels 4->4->4)
    S4: Conv4a -> ReLU -> Conv4b -> (+input) -> ReLU -> Pool4  (channels 8->8->8)
    Stages 1 and 3 are plain (channel-changing, no skip).
    """

    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1  = nn.Conv1d(1, 4, 5, padding=2, bias=True)
        # Stage 2 residual block (4->4->4)
        self.conv2a = nn.Conv1d(4, 4, 5, padding=2, bias=True)
        self.conv2b = nn.Conv1d(4, 4, 5, padding=2, bias=True)
        self.conv3  = nn.Conv1d(4, 8, 5, padding=2, bias=True)
        # Stage 4 residual block (8->8->8)
        self.conv4a = nn.Conv1d(8, 8, 5, padding=2, bias=True)
        self.conv4b = nn.Conv1d(8, 8, 5, padding=2, bias=True)
        self.pool   = nn.MaxPool1d(5)
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(8, num_classes, bias=True)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool(self.conv1(x))                       # S1 plain
        # S2 residual
        identity = x
        h = F.relu(self.conv2a(x))
        h = self.conv2b(h)
        x = self.pool(h + identity)                        # +x, no ReLU (no-ReLU stage)
        x = self.pool(self.conv3(x))                       # S3 plain
        # S4 residual (with ReLU, matching baseline's post-conv4 ReLU)
        identity = x
        h = F.relu(self.conv4a(x))
        h = self.conv4b(h)
        x = self.pool(F.relu(h + identity))                # +x then ReLU
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
#  Train / eval
# ============================================================

def _validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for ecg, labels, _ in loader:
            ecg    = ecg.unsqueeze(1).float().to(device)
            labels = labels.to(device)
            out    = model(ecg)
            total_loss += criterion(out, labels).item() * ecg.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            total      += labels.size(0)
    return total_loss / total, correct / total


def train_from_scratch(model, train_loader, val_loader, device, e1, e2, tag):
    criterion     = nn.CrossEntropyLoss()
    phases        = [(e1, 1e-3), (e2, 1e-4)]
    best_val_loss = float('inf')
    best_state    = copy.deepcopy(model.state_dict())

    print(f"\n[{tag}] Training from scratch ({e1}@1e-3 + {e2}@1e-4) ...")
    print(f"  {'Ep':>4}  {'LR':>8}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VaLoss':>8}  {'VaAcc':>7}")
    print("  " + "-" * 56)

    ge = 0
    for pe, lr in phases:
        opt = optim.Adam(model.parameters(), lr=lr)
        for _ in range(pe):
            ge += 1
            model.train()
            run_loss, correct, total = 0.0, 0, 0
            for ecg, labels, _ in train_loader:
                ecg    = ecg.unsqueeze(1).float().to(device)
                labels = labels.to(device)
                opt.zero_grad()
                out  = model(ecg)
                loss = criterion(out, labels)
                loss.backward()
                opt.step()
                run_loss += loss.item() * ecg.size(0)
                correct  += out.argmax(1).eq(labels).sum().item()
                total    += labels.size(0)
            va_loss, va_acc = _validate(model, val_loader, criterion, device)
            star = ""
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_state    = copy.deepcopy(model.state_dict())
                star = " *"
            print(f"  {ge:>4}  {lr:>8.0e}  {run_loss/total:>8.4f}  "
                  f"{correct/total:>7.4f}  {va_loss:>8.4f}  {va_acc:>7.4f}{star}")
    model.load_state_dict(best_state)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   type=str, default='../../data/Chapman')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--epochs1',    type=int, default=30)
    p.add_argument('--epochs2',    type=int, default=20)
    p.add_argument('--seed',       type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}  seed={args.seed}")

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    results = []
    for tag, ctor in [("baseline", ECG_Baseline), ("res24", ECG_Res24)]:
        torch.manual_seed(args.seed)   # identical init seed per model
        model = ctor().to(device)
        print(f"\n{'='*60}\n  {tag}: {model.count_parameters()} params\n{'='*60}")
        model = train_from_scratch(model, train_loader, val_loader, device,
                                   args.epochs1, args.epochs2, tag)
        model.eval()
        pred, lab = evaluate_model(model, test_loader, device)
        m = compute_metrics(pred, lab, CLASS_NAMES)
        results.append((tag, model.count_parameters(),
                        m['accuracy'], m['f1_macro']))

    print("\n" + "=" * 60)
    print("  RESIDUAL ABLATION SUMMARY (float, from scratch)")
    print("=" * 60)
    print(f"  {'Model':<14} {'Params':>8} {'Acc':>10} {'F1-macro':>10}")
    print("  " + "-" * 44)
    for tag, np_, acc, f1 in results:
        print(f"  {tag:<14} {np_:>8} {acc:>10.4f} {f1:>10.4f}")
    print("=" * 60)
    b = results[0]; r = results[1]
    print(f"\n  Delta acc (res24 - baseline): {(r[2]-b[2])*100:+.2f} pp")
    print(f"  Delta F1  (res24 - baseline): {(r[3]-b[3])*100:+.2f} pp")
    print(f"  5-fold std reference ~0.4-0.9pp -> needs >~1pp to be meaningful.")


if __name__ == '__main__':
    main()