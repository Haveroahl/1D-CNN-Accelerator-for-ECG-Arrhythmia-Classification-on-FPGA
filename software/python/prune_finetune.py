"""
Structured Channel Pruning + Fine-tuning for ECG_1DCNN
======================================================
Pipeline:
  best_model.pth
    → L1-norm structured channel pruning (selected Conv layers)
    → Fine-tune (2-phase LR)
    → best_model_pruned.pth

Pruning strategy: L1-norm filter ranking on all Conv layers
  Conv1: 1→4   → 1→4   (kept all 4 filters)
  Conv2: 4→8   → 4→4   (remove 4 weakest filters, input = kept Conv1 out)
  Conv3: 8→8   → 8→8   (kept all 8 filters)
  Conv4: 8→16  → 8→8   (remove 8 weakest filters, input = kept Conv3 out)
  FC:   16→4   → 8→4  (input shrinks with Conv4 pruning)

Parameter count:
  Full  : (1×4×5+4)+(4×8×5+8)+(8×8×5+8)+(8×16×5+16)+(16×4+4) = 1244
  Pruned: (1×4×5+4)+(4×4×5+4)+(4×8×5+8)+(8×8×5+8)+(8×4+4) = 640
  Reduction: 48.6%

Fine-tune schedule:
  Phase 1: 30 epochs @ lr=1e-3  (recovery from pruning)
  Phase 2: 20 epochs @ lr=1e-4  (fine refinement)

Usage:
  python prune_finetune.py \\
      --checkpoint ./results/best_model.pth \\
      --data_dir   D:/Thesis101/data/Chapman \\
      --output_dir ./results
"""

import os
import sys
import copy
import json
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset  import get_dataloaders, CLASS_NAMES
from utils.evaluate import evaluate_model, compute_metrics, print_classification_report
from model.model import ECG_1DCNN, ECG_1DCNN_Q88, ECG_1DCNN_INT8


# ============================================================
#  Pruned model definition
# ============================================================

class ECG_1DCNN_Pruned(nn.Module):
    """
    ECG_1DCNN with reduced channel counts after L1-norm channel pruning.
    All Conv layers pruned; ReLU only after Conv4.

    Args:
        c1_out: Conv1 output channels (default 4)
        c2_out: Conv2 output channels (default 4)
        c3_out: Conv3 output channels (default 8)
        c4_out: Conv4 output channels (default 8)
    """

    def __init__(self, c1_out=4, c2_out=4, c3_out=8, c4_out=8, num_classes=4):
        super().__init__()
        self.c1_out      = c1_out
        self.c2_out      = c2_out
        self.c3_out      = c3_out
        self.c4_out      = c4_out
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(1,      c1_out, kernel_size=5, padding=2, bias=True)
        self.conv2 = nn.Conv1d(c1_out, c2_out, kernel_size=5, padding=2, bias=True)
        self.conv3 = nn.Conv1d(c2_out, c3_out, kernel_size=5, padding=2, bias=True)
        self.conv4 = nn.Conv1d(c3_out, c4_out, kernel_size=5, padding=2, bias=True)

        self.pool1 = nn.MaxPool1d(kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=5)
        self.pool3 = nn.MaxPool1d(kernel_size=5)
        self.pool4 = nn.MaxPool1d(kernel_size=5)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(c4_out, num_classes, bias=True)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))             # (B, c1_out, 500) [NO ReLU]
        x = self.pool2(self.conv2(x))             # (B, c2_out, 100) [NO ReLU]
        x = self.pool3(self.conv3(x))             # (B, c3_out,  20) [NO ReLU]
        x = self.pool4(F.relu(self.conv4(x)))     # (B, c4_out,   4) [+ReLU]
        x = self.gap(x).squeeze(-1)               # (B, c4_out)
        return self.fc(x)                          # (B, num_classes)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
#  Channel importance ranking
# ============================================================

def _l1_rank(weight: torch.Tensor) -> torch.Tensor:
    """Rank output channels by L1-norm descending. Returns sorted indices."""
    norms = weight.abs().sum(dim=tuple(range(1, weight.dim())))
    return torch.argsort(norms, descending=True)


def _taylor_rank(model: ECG_1DCNN, loader, layer_names: list,
                 device, n_batches: int = 20) -> dict:
    """
    First-order Taylor expansion importance scoring (Molchanov et al. 2017).

    For each channel i in each conv layer:
        I_i = | E[ (∂L/∂h_i) · h_i ] |

    where h_i is the activation output and ∂L/∂h_i its gradient.
    Channels with high I_i contribute more to the loss → keep them.

    Returns: {layer_name: importance tensor of shape (C,)}
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    scores    = {name: None for name in layer_names}
    n_accum   = 0

    for batch_idx, (ecg, labels, _) in enumerate(loader):
        if batch_idx >= n_batches:
            break

        ecg    = ecg.unsqueeze(1).float().to(device)
        labels = labels.to(device)

        # Forward — register hooks to capture activations with grad
        batch_acts = {}
        hooks = []
        for name in layer_names:
            def make_hook(n):
                def hook(module, inp, out):
                    batch_acts[n] = out
                    out.retain_grad()
                return hook
            hooks.append(getattr(model, name).register_forward_hook(make_hook(name)))

        model.zero_grad()
        loss = criterion(model(ecg), labels)
        loss.backward()

        for h in hooks:
            h.remove()

        # Accumulate |mean_batch_spatial(grad * act)| per channel
        for name in layer_names:
            act  = batch_acts[name]          # (B, C, L)
            grad = act.grad                  # (B, C, L)
            if grad is None:
                continue
            # Average over batch and spatial dims → (C,)
            taylor = (grad * act).abs().mean(dim=(0, 2)).detach()
            scores[name] = taylor if scores[name] is None else scores[name] + taylor

        n_accum += 1

    # Normalise to average over batches
    for name in layer_names:
        if scores[name] is not None and n_accum > 0:
            scores[name] /= n_accum

    model.eval()
    return scores


def prune_model(full_model: ECG_1DCNN, train_loader=None, device='cpu'):
    """
    Structured channel pruning. Target: Conv1(4→4), Conv2(8→4),
    Conv3(8→8), Conv4(16→8).

    Ranking strategy:
      - train_loader provided → Taylor expansion + gradient sensitivity
      - train_loader is None  → L1-norm fallback (weight-only, no data needed)

    Returns:
        pruned_model: ECG_1DCNN_Pruned with transferred weights
        kept:         dict of kept channel indices per Conv layer
    """
    targets    = {'conv1': 4, 'conv2': 4, 'conv3': 8, 'conv4': 8}
    layer_names = list(targets.keys())

    # ── Choose ranking method ──────────────────────────────────
    if train_loader is not None:
        print("  [Pruning] Taylor expansion + gradient sensitivity scoring "
              f"({20} batches) ...")
        taylor_scores = _taylor_rank(full_model, train_loader,
                                     layer_names, device, n_batches=20)
        kept = {}
        for name, keep_n in targets.items():
            if taylor_scores[name] is not None:
                idx = torch.argsort(taylor_scores[name], descending=True)[:keep_n]
            else:
                w   = getattr(full_model, name).weight.data
                idx = _l1_rank(w)[:keep_n]
            kept[name] = torch.sort(idx)[0]
        print("  [Pruning] Taylor scores per layer:")
        for name in layer_names:
            s = taylor_scores[name]
            if s is not None:
                print(f"    {name}: {s.cpu().numpy().round(4)}")
    else:
        print("  [Pruning] L1-norm ranking (no data loader provided) ...")
        kept = {}
        for name, keep_n in targets.items():
            w   = getattr(full_model, name).weight.data
            idx = torch.sort(_l1_rank(w)[:keep_n])[0]
            kept[name] = idx

    with torch.no_grad():
        kept = {k: v.to('cpu') for k, v in kept.items()}

        pruned = ECG_1DCNN_Pruned(c1_out=4, c2_out=4, c3_out=8, c4_out=8)

        # Conv1: prune output channels only (input=1, always full)
        pruned.conv1.weight.data.copy_(
            full_model.conv1.weight.data[kept['conv1']]
        )
        pruned.conv1.bias.data.copy_(
            full_model.conv1.bias.data[kept['conv1']]
        )

        # Conv2: prune input (= kept Conv1 outputs) + output channels
        pruned.conv2.weight.data.copy_(
            full_model.conv2.weight.data[kept['conv2']][:, kept['conv1'], :]
        )
        pruned.conv2.bias.data.copy_(
            full_model.conv2.bias.data[kept['conv2']]
        )

        # Conv3: prune input (= kept Conv2 outputs) + output channels
        pruned.conv3.weight.data.copy_(
            full_model.conv3.weight.data[kept['conv3']][:, kept['conv2'], :]
        )
        pruned.conv3.bias.data.copy_(
            full_model.conv3.bias.data[kept['conv3']]
        )

        # Conv4: prune input (= kept Conv3 outputs) + output channels
        pruned.conv4.weight.data.copy_(
            full_model.conv4.weight.data[kept['conv4']][:, kept['conv3'], :]
        )
        pruned.conv4.bias.data.copy_(
            full_model.conv4.bias.data[kept['conv4']]
        )

        # FC: prune input features (= kept Conv4 outputs)
        pruned.fc.weight.data.copy_(
            full_model.fc.weight.data[:, kept['conv4']]
        )
        pruned.fc.bias.data.copy_(full_model.fc.bias.data)

    return pruned, kept


# ============================================================
#  Fine-tuning
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


def finetune(model, train_loader, val_loader, device):
    """
    Fine-tune the pruned model with a 2-phase LR schedule:
      Phase 1: 30 epochs @ lr=1e-3
      Phase 2: 20 epochs @ lr=1e-4
    Best val_loss checkpoint is restored at the end.
    """
    criterion     = nn.CrossEntropyLoss()
    phases        = [(30, 1e-3), (20, 1e-4)]
    best_val_loss = float('inf')
    best_state    = copy.deepcopy(model.state_dict())

    print(f"\n  Fine-tuning 50 epochs — 2-phase LR:")
    print(f"    Phase 1: 30 epochs @ lr=1e-3")
    print(f"    Phase 2: 20 epochs @ lr=1e-4")
    print(f"  {'Ep':>4}  {'LR':>8}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VaLoss':>8}  {'VaAcc':>7}  {'Time':>6}")
    print("  " + "-" * 65)

    global_epoch = 0
    for phase_epochs, phase_lr in phases:
        optimizer = optim.Adam(model.parameters(), lr=phase_lr)

        for _ in range(phase_epochs):
            global_epoch += 1
            t0 = time.time()
            model.train()
            run_loss, correct, total = 0.0, 0, 0

            for ecg, labels, _ in train_loader:
                ecg    = ecg.unsqueeze(1).float().to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                out  = model(ecg)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                run_loss += loss.item() * ecg.size(0)
                correct  += out.argmax(1).eq(labels).sum().item()
                total    += labels.size(0)

            tr_loss = run_loss / total
            tr_acc  = correct  / total
            va_loss, va_acc = _validate(model, val_loader, criterion, device)

            star = ""
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_state    = copy.deepcopy(model.state_dict())
                star = " ★"

            print(f"  {global_epoch:>4}  {phase_lr:>8.0e}  {tr_loss:>8.4f}  "
                  f"{tr_acc:>7.4f}  {va_loss:>8.4f}  {va_acc:>7.4f}  "
                  f"{time.time()-t0:>5.1f}s{star}")

    model.load_state_dict(best_state)
    print(f"\n  Best fine-tuned val_loss = {best_val_loss:.4f}")
    return model


# ============================================================
#  Main pipeline
# ============================================================

def prune_and_finetune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────
    print("\n[1/5] Loading dataset ...")
    if args.npz:
        from utils.npz_dataset import get_npz_dataloaders
        train_loader, val_loader, test_loader = get_npz_dataloaders(
            args.npz, batch_size=args.batch_size, num_workers=args.num_workers,
        )
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            args.data_dir, batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # ── 2. Load full trained v3 model ───────────────────────
    print(f"\n[2/5] Loading checkpoint: {args.checkpoint}")
    full_model = ECG_1DCNN(num_classes=4).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    full_model.load_state_dict(ckpt['model_state_dict'])
    full_model.eval()

    params_before = full_model.count_parameters()
    print(f"  Parameters (full): {params_before}")

    # ── 3. Baseline accuracy ─────────────────────────────────
    print("\n[3/5] Baseline accuracy ...")

    fp, fl  = evaluate_model(full_model, test_loader, device)
    m_float = compute_metrics(fp, fl, CLASS_NAMES)
    print_classification_report(m_float, title="Full Model — Float (fp32)")

    q88_full = ECG_1DCNN_Q88(full_model).to(device)
    qp, ql   = evaluate_model(q88_full, test_loader, device)
    m_q88    = compute_metrics(qp, ql, CLASS_NAMES)
    print_classification_report(m_q88, title="Full Model — Q8.8 (FPGA simulation)")

    int8_full = ECG_1DCNN_INT8(full_model, calib_loader=train_loader, device=device)
    ip, il    = evaluate_model(int8_full, test_loader, device)
    m_int8    = compute_metrics(ip, il, CLASS_NAMES)
    print_classification_report(m_int8, title="Full Model — INT8 (hardware simulation)")

    # ── 4. Structured pruning ────────────────────────────────
    print(f"\n[4/5] Pruning (Taylor expansion, target channels 4,4,8,8) ...")
    pruned_model, kept = prune_model(full_model, train_loader=train_loader,
                                     device=device)
    pruned_model = pruned_model.to(device)

    params_after = pruned_model.count_parameters()
    reduction    = (1 - params_after / params_before) * 100

    for lname, idx in kept.items():
        orig = getattr(full_model, lname).out_channels
        print(f"  {lname}: kept {idx.tolist()}  ({len(idx)}/{orig})")
    print(f"  Parameters: {params_before} → {params_after}  (↓{reduction:.1f}%)")

    pruned_model.eval()
    pp, pl   = evaluate_model(pruned_model, test_loader, device)
    m_pruned = compute_metrics(pp, pl, CLASS_NAMES)
    print_classification_report(m_pruned, title="Pruned (before fine-tune) — Float")

    # ── 5. Fine-tuning ───────────────────────────────────────
    print(f"\n[5/5] Fine-tuning ...")
    pruned_model = finetune(pruned_model, train_loader, val_loader, device)

    pruned_model.eval()
    fp2, fl2  = evaluate_model(pruned_model, test_loader, device)
    m_ft      = compute_metrics(fp2, fl2, CLASS_NAMES)
    print_classification_report(m_ft, title="Pruned + fine-tuned — Float")

    q88_pruned = ECG_1DCNN_Q88(pruned_model).to(device)
    qp2, ql2   = evaluate_model(q88_pruned, test_loader, device)
    m_ft_q88   = compute_metrics(qp2, ql2, CLASS_NAMES)
    print_classification_report(m_ft_q88, title="Pruned + fine-tuned — Q8.8 (FPGA)")

    int8_pruned = ECG_1DCNN_INT8(pruned_model, calib_loader=train_loader, device=device)
    ip2, il2    = evaluate_model(int8_pruned, test_loader, device)
    m_ft_int8   = compute_metrics(ip2, il2, CLASS_NAMES)
    print_classification_report(m_ft_int8, title="Pruned + fine-tuned — INT8 (hardware simulation)")

    # ── Summary ──────────────────────────────────────────────
    rows = [
        ("Full — Float",              m_float['accuracy'],   m_float['f1_macro']),
        ("Full — Q8.8",               m_q88['accuracy'],     m_q88['f1_macro']),
        ("Full — INT8",               m_int8['accuracy'],    m_int8['f1_macro']),
        ("Pruned (no fine-tune)",     m_pruned['accuracy'],  m_pruned['f1_macro']),
        ("Pruned + fine-tuned",       m_ft['accuracy'],      m_ft['f1_macro']),
        ("Pruned + fine-tuned Q8.8",  m_ft_q88['accuracy'],  m_ft_q88['f1_macro']),
        ("Pruned + fine-tuned INT8",  m_ft_int8['accuracy'], m_ft_int8['f1_macro']),
    ]
    print("\n" + "=" * 60)
    print("  PRUNING SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<30} {'Accuracy':>10}  {'F1-macro':>10}")
    print("  " + "-" * 54)
    for name, acc, f1 in rows:
        print(f"  {name:<30} {acc:>10.4f}  {f1:>10.4f}")
    print("=" * 60)
    print(f"\n  Parameter reduction: {params_before} → {params_after} "
          f"(↓{reduction:.1f}%)")

    # ── Save ─────────────────────────────────────────────────
    out_ckpt = os.path.join(args.output_dir, 'best_model_pruned.pth')
    torch.save({
        'model_state_dict': pruned_model.state_dict(),
        'c1_out':           pruned_model.c1_out,
        'c2_out':           pruned_model.c2_out,
        'c3_out':           pruned_model.c3_out,
        'c4_out':           pruned_model.c4_out,
        'kept_channels':    {k: v.tolist() for k, v in kept.items()},
        'params_before':    params_before,
        'params_after':     params_after,
    }, out_ckpt)
    print(f"\n  Checkpoint → {out_ckpt}")

    results = {
        'architecture': {
            'full':   {'conv1': '1→4', 'conv2': '4→8', 'conv3': '8→8',
                       'conv4': '8→16', 'fc_in': 16, 'params': params_before},
            'pruned': {'conv1': '1→4', 'conv2': '4→4', 'conv3': '8→8',
                       'conv4': '8→8', 'fc_in': 8, 'params': params_after,
                       'reduction_pct': round(reduction, 2)},
        },
        'kept_channels': {k: v.tolist() for k, v in kept.items()},
        'metrics': {name: {'acc': acc, 'f1_macro': f1} for name, acc, f1 in rows},
        'confusion_matrix_pruned_float': m_ft['confusion_matrix'].tolist(),
        'confusion_matrix_pruned_q88':   m_ft_q88['confusion_matrix'].tolist(),
        'confusion_matrix_pruned_int8':  m_ft_int8['confusion_matrix'].tolist(),
    }
    out_json = os.path.join(args.output_dir, 'v3_results_pruned.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON → {out_json}")

    return pruned_model


# ============================================================
#  CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Structured channel pruning + fine-tuning for ECG_1DCNN'
    )
    p.add_argument('--checkpoint',  type=str,
                   default='./results/best_model.pth')
    p.add_argument('--data_dir',    type=str,
                   default='../../data/Chapman')
    p.add_argument('--npz',         type=str, default=None,
                   help='Pre-split .npz path; overrides --data_dir when set')
    p.add_argument('--output_dir',  type=str,   default='./results')
    p.add_argument('--batch_size',  type=int,   default=128)
    p.add_argument('--num_workers', type=int,   default=2)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("=" * 60)
    print("  ECG_1DCNN — Structured Pruning + Fine-tuning")
    print("=" * 60)
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Pruning     : Channel reduction to 4,4,8,8 (1244 → 640 params)")
    print(f"  Fine-tune   : 50 epochs  (30 @ 1e-3 + 20 @ 1e-4)")
    print(f"  Output      : {args.output_dir}")
    print("=" * 60)
    prune_and_finetune(args)
