"""
Test: hidden layer + dropout on the PRUNED hardware model (4-4-8-8, FC 8->4)
===========================================================================
Loads best_model_pruned.pth (conv layers already trained), swaps the FC head,
then fine-tunes the whole model with the same 2-phase recipe as
prune_finetune.py (30 ep @ 1e-3 + 20 ep @ 1e-4).

Configs (same seed / split / recipe):
  base                : GAP(->8) -> FC(8->4)                       [matches deployed model]
  hidden32            : GAP(->8) -> FC(8->32) -> ReLU -> FC(32->4)
  hidden32+drop0.2    : GAP(->8) -> FC(8->32) -> ReLU -> Dropout(0.2) -> FC(32->4)
  hidden32+drop0.3    : ... Dropout(0.3) ...

Float32 only. Does NOT modify model.py, prune_finetune.py, or the deployed ckpt.

Usage:
  python test_pruned_head.py [--hidden 32]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import get_dataloaders, CLASS_NAMES
from utils.evaluate import evaluate_model, compute_metrics
from prune_finetune import ECG_1DCNN_Pruned


# ============================================================
#  Pruned model with configurable FC head
# ============================================================

class ECG_1DCNN_PrunedHead(ECG_1DCNN_Pruned):
    """Pruned model (4-4-8-8) with an optional hidden FC layer + dropout in the head."""

    def __init__(self, hidden=0, dropout=0.0, num_classes=4):
        super().__init__(c1_out=4, c2_out=4, c3_out=8, c4_out=8,
                         num_classes=num_classes)
        c_in = self.c4_out  # = 8 (GAP output width)
        layers = []
        if hidden > 0:
            layers += [nn.Linear(c_in, hidden, bias=True), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, num_classes, bias=True))
        else:
            layers.append(nn.Linear(c_in, num_classes, bias=True))
        self.fc = nn.Sequential(*layers)

    # forward() inherited from ECG_1DCNN_Pruned: ...gap(x).squeeze(-1) -> self.fc(x)


def load_conv_from_ckpt(model, ckpt_path, device):
    """Copy trained conv1-4 weights from the pruned checkpoint; leave the new head random."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    conv_keys = {k: v for k, v in state.items() if k.startswith('conv')}
    missing, unexpected = model.load_state_dict(conv_keys, strict=False)
    # missing should be only the new fc.* params; conv keys must all load.
    assert not any(k.startswith('conv') for k in missing), \
        f"conv weights failed to load: {missing}"
    return model


# ============================================================
#  2-phase fine-tune (mirrors prune_finetune.finetune)
# ============================================================

def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for ecg, labels, _ in loader:
            ecg    = ecg.unsqueeze(1).float().to(device)
            labels = labels.to(device)
            total_loss += criterion(model(ecg), labels).item() * ecg.size(0)
            total      += labels.size(0)
    return total_loss / total


def finetune(model, train_loader, val_loader, test_loader, device, tag):
    criterion = nn.CrossEntropyLoss()
    phases = [(30, 1e-3), (20, 1e-4)]
    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())

    for phase_epochs, phase_lr in phases:
        optimizer = optim.Adam(model.parameters(), lr=phase_lr)
        for _ in range(phase_epochs):
            model.train()
            for ecg, labels, _ in train_loader:
                ecg    = ecg.unsqueeze(1).float().to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(ecg), labels)
                loss.backward()
                optimizer.step()
            va_loss = evaluate_loss(model, val_loader, criterion, device)
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    preds, labels = evaluate_model(model, test_loader, device)
    m = compute_metrics(preds, labels, CLASS_NAMES)
    print(f"  [{tag:<18}] best_val_loss={best_val_loss:.4f}  "
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
    ap.add_argument('--checkpoint', default=r'd:\Thesis101\software\python\results\best_model_pruned.pth')
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device} | hidden={args.hidden}")
    print(f"[INFO] Conv weights from: {args.checkpoint}\n")

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    H = args.hidden
    # (tag, hidden, dropout)
    configs = [
        ('base', 0, 0.0),
        (f'hidden{H}', H, 0.0),
        (f'hidden{H}+drop0.2', H, 0.2),
        (f'hidden{H}+drop0.3', H, 0.3),
    ]

    results = []
    for tag, hid, drop in configs:
        reseed(42)
        model = ECG_1DCNN_PrunedHead(hidden=hid, dropout=drop).to(device)
        load_conv_from_ckpt(model, args.checkpoint, device)
        results.append(finetune(model, train_loader, val_loader, test_loader,
                                device, tag))

    # ---- Report ----
    base_acc = results[0]['accuracy']
    print("\n" + "=" * 60)
    print(f"{'Config':<20}{'Params':<9}{'Accuracy':<11}{'F1-macro':<11}{'dAcc%':<8}")
    print("-" * 60)
    for r in results:
        d_acc = (r['accuracy'] - base_acc) * 100
        print(f"{r['tag']:<20}{r['params']:<9}{r['accuracy']*100:<11.2f}"
              f"{r['f1_macro']:<11.4f}{d_acc:<+8.2f}")
    print("=" * 60)
    best = max(results, key=lambda r: r['accuracy'])
    print(f"\nBest config: {best['tag']} "
          f"(acc={best['accuracy']*100:.2f}, f1={best['f1_macro']:.4f})")
    print("\nNOTE: base here = current deployed pruned model (FC 8->4).")
    print("      Hidden/dropout exist only at train time; an extra FC would need")
    print("      adding to the RTL head if adopted (current RTL has single FC).")


if __name__ == "__main__":
    main()
