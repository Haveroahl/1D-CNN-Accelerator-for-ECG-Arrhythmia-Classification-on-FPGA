"""
Training Pipeline — MIT-BIH beat classifier (5-class AAMI)
=========================================================
Model    : ECG_BeatCNN (compact residual 1D-CNN, 2181 params)
Optimizer: Adam, lr=1e-4 (per request), cosine-annealed
Loss     : CrossEntropyLoss with inverse-frequency class weights
Best ckpt: lowest val loss (then re-evaluated on test)

Two split schemes (see utils/dataset.py):
  --scheme intra   random 80/10/10 over all beats (optimistic, ~98% target)
  --scheme inter   AAMI DS1/DS2 inter-patient (honest, no patient leakage)

Anti-overfitting: weighted CE + minority oversampling+jitter (train only),
BatchNorm, cosine LR, early-stop on val loss, weight decay.

Usage:
  python train.py --data_dir ../../data/mitdb --scheme intra
  python train.py --scheme inter --output_dir results/inter
"""

import os
import sys
import argparse
import csv
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))

from utils.dataset import (get_dataloaders, class_weights,
                           CLASS_NAMES, NUM_CLASSES)
from model.model import build_model


class FocalLoss(nn.Module):
    """
    Multi-class focal loss (Lin et al. 2017) with optional class weights (alpha).

    Down-weights easy, already-correct beats so the gradient concentrates on
    the hard minorities (S vs N, F vs N/V) that single-beat morphology + RR
    still confuses. gamma=0 reduces to weighted cross-entropy.
    """

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target, weight=self.weight, reduction='none')
        pt = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        return ((1 - pt) ** self.gamma * ce).mean()


# ============================================================
#  Metrics (beat-level, self-contained)
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for ecg, rr, y in loader:
            out = model(ecg.to(device), rr.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    return np.array(preds), np.array(labels)


def compute_metrics(preds, labels):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1
    acc = np.trace(cm) / max(np.sum(cm), 1)
    f1s, per_class = [], {}
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1   = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0
        per_class[CLASS_NAMES[i]] = dict(precision=prec, recall=rec, f1=f1,
                                         support=support)
        # F1-macro averages only over classes actually present in the labels;
        # a class with 0 true beats (e.g. Q dropped under the inter protocol)
        # would otherwise contribute a spurious F1=0.
        if support > 0:
            f1s.append(f1)
    return dict(accuracy=float(acc), f1_macro=float(np.mean(f1s)),
                confusion_matrix=cm, per_class=per_class)


def fold4_metrics(preds, labels):
    """
    4-class AAMI view (N/S/V/Q) by folding F (fusion) into V.

    The F class — fusion of a normal and a ventricular beat — sits on the
    physiological N/V boundary and is not reliably separable from single-beat
    morphology (only 802 beats in the whole DB). The standard MIT-BIH practice
    (Kachuee 2018; Hannun 2019) is to merge it into the ventricular class.
    We report this in parallel with the full 5-class numbers, not instead.
    """
    F_IDX, V_IDX = 3, 2
    p = np.where(preds  == F_IDX, V_IDX, preds)
    l = np.where(labels == F_IDX, V_IDX, labels)
    names4 = ['N', 'S', 'V', 'Q']
    remap = {0: 0, 1: 1, 2: 2, 4: 3}        # original idx → 4-class idx
    p = np.vectorize(remap.get)(p)
    l = np.vectorize(remap.get)(l)
    cm = np.zeros((4, 4), dtype=int)
    for t, pr in zip(l, p):
        cm[t][pr] += 1
    acc = np.trace(cm) / max(cm.sum(), 1)
    f1s, per_class = [], {}
    for i in range(4):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1   = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0
        per_class[names4[i]] = dict(precision=prec, recall=rec, f1=f1,
                                    support=int(cm[i, :].sum()))
        f1s.append(f1)
    return dict(accuracy=float(acc), f1_macro=float(np.mean(f1s)),
                confusion_matrix=cm, per_class=per_class), names4


def fold3_metrics(preds, labels):
    """
    3-class AAMI view N/S/V (F folded into V, Q already absent under inter).

    Matches the matched-filter CNN comparison (Sensors 2023, 23(3), 1365),
    which reports N/SVEB/VEB on the de Chazal DS1/DS2 split. Beats labelled Q
    do not occur in the inter protocol (paced records dropped); any stray Q is
    excluded here so the 3-class view is exactly N/S/V.
    """
    F_IDX, V_IDX, Q_IDX = 3, 2, 4
    keep = labels != Q_IDX
    preds, labels = preds[keep], labels[keep]
    p = np.where(preds  == F_IDX, V_IDX, preds)
    l = np.where(labels == F_IDX, V_IDX, labels)
    p = np.where(p == Q_IDX, V_IDX, p)          # any predicted-Q → V (rare)
    names3 = ['N', 'S', 'V']
    cm = np.zeros((3, 3), dtype=int)
    for t, pr in zip(l, p):
        cm[t][pr] += 1
    acc = np.trace(cm) / max(cm.sum(), 1)
    f1s, per_class = [], {}
    for i in range(3):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1   = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0
        per_class[names3[i]] = dict(precision=prec, recall=rec, f1=f1,
                                    support=int(cm[i, :].sum()))
        f1s.append(f1)
    return dict(accuracy=float(acc), f1_macro=float(np.mean(f1s)),
                confusion_matrix=cm, per_class=per_class), names3


def print_report(m, title="", names=None):
    if title:
        print(f"\n  {'-'*52}\n  {title}\n  {'-'*52}")
    print(f"  Accuracy : {m['accuracy']:.4f}")
    print(f"  F1-macro : {m['f1_macro']:.4f}\n")
    print(f"  {'Class':<8}{'Prec':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    print("  " + "-" * 48)
    for c, v in m['per_class'].items():
        print(f"  {c:<8}{v['precision']:>10.4f}{v['recall']:>10.4f}"
              f"{v['f1']:>10.4f}{v['support']:>10d}")
    names = names or CLASS_NAMES
    print("\n  Confusion (rows=true, cols=pred):")
    cm = m['confusion_matrix']
    print("  " + " " * 6 + "".join(f"{c:>8}" for c in names))
    for i, c in enumerate(names):
        print("  " + f"{c:<6}" + "".join(f"{cm[i,j]:>8d}" for j in range(len(names))))


# ============================================================
#  Epoch helpers
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for ecg, rr, y in loader:
        ecg, rr, y = ecg.to(device), rr.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(ecg, rr)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * ecg.size(0)
        correct    += out.detach().argmax(1).eq(y).sum().item()
        total      += y.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for ecg, rr, y in loader:
            ecg, rr, y = ecg.to(device), rr.to(device), y.to(device)
            out  = model(ecg, rr)
            loss = criterion(out, y)
            total_loss += loss.item() * ecg.size(0)
            correct    += out.argmax(1).eq(y).sum().item()
            total      += y.size(0)
    return total_loss / total, correct / total


# ============================================================
#  Main
# ============================================================

def train(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}  seed: {seed}  scheme: {args.scheme}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────
    print("\n[1/4] Loading dataset ...")
    aug_minority = [CLASS_NAMES.index(c) for c in args.aug_minority.split(',')
                    if c] if args.aug_minority else None
    train_loader, val_loader, test_loader, train_ds = get_dataloaders(
        args.data_dir, scheme=args.scheme, batch_size=args.batch_size,
        num_workers=args.num_workers, augment=args.augment,
        target_per_class=args.target_per_class, seed=seed,
        aug_minority=aug_minority, aug_p=args.aug_p)
    if aug_minority:
        print(f"[INFO] on-the-fly minority aug: classes={args.aug_minority} "
              f"p={args.aug_p}")

    # ── 2. Model ─────────────────────────────────────────────
    print(f"\n[2/4] Building model ({args.model}) ...")
    model = build_model(args.model, num_classes=NUM_CLASSES).to(device)
    print(model.layer_summary())
    n_params = model.count_parameters()
    print(f"\nTotal parameters: {n_params}")
    assert n_params <= args.param_budget, \
        f"param budget exceeded: {n_params} > {args.param_budget}"

    cw = class_weights(train_ds, temperature=args.weight_temp).to(device)
    print(f"[INFO] class weights (T={args.weight_temp}): "
          f"{cw.cpu().numpy().round(3)}")
    if args.loss == 'focal':
        criterion = FocalLoss(gamma=args.focal_gamma, weight=cw)
        print(f"[INFO] loss: focal (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=cw)
        print("[INFO] loss: weighted cross-entropy")
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 3. Train ─────────────────────────────────────────────
    print(f"\n[3/4] Training for {args.epochs} epochs ...")
    log_csv = os.path.join(args.output_dir, 'train_log.csv')
    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc',
                                'val_loss', 'val_acc', 'val_f1', 'lr',
                                'time_s', 'is_best'])

    best_val_f1 = -1.0   # select checkpoint on the imbalanced target metric
    best_epoch = 0
    epochs_no_improve = 0
    ckpt_path = os.path.join(args.output_dir, 'best_model.pth')
    history = {k: [] for k in ('train_loss', 'train_acc',
                               'val_loss', 'val_acc', 'val_f1')}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_ds.set_epoch(epoch)   # re-roll per-epoch minority jitter
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, device)
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        vp, vl = evaluate(model, val_loader, device)
        val_f1 = compute_metrics(vp, vl)['f1_macro']
        cur_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc, val_f1)):
            history[k].append(v)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1, best_epoch, epochs_no_improve = val_f1, epoch, 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': va_loss, 'val_acc': va_acc, 'val_f1': val_f1,
                        'scheme': args.scheme}, ckpt_path)
        else:
            epochs_no_improve += 1

        star = " *" if is_best else ""
        print(f"  Epoch {epoch:3d}/{args.epochs}  Train {tr_loss:.4f}/{tr_acc:.4f}"
              f"  Val {va_loss:.4f}/{va_acc:.4f}  F1 {val_f1:.4f}"
              f"  LR {cur_lr:.1e}  {elapsed:.1f}s{star}")

        with open(log_csv, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}",
                                    f"{va_loss:.6f}", f"{va_acc:.6f}",
                                    f"{val_f1:.6f}", f"{cur_lr:.1e}",
                                    f"{elapsed:.1f}", int(is_best)])
        scheduler.step()

        if args.patience and epochs_no_improve >= args.patience:
            print(f"  [early-stop] no val-loss improvement for {args.patience} epochs")
            break

    with open(os.path.join(args.output_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n  Best checkpoint: epoch {best_epoch}  val_f1={best_val_f1:.4f}")

    # ── 4. Test ──────────────────────────────────────────────
    print("\n[4/4] Evaluating on test set ...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    tp, tl = evaluate(model, test_loader, device)
    m = compute_metrics(tp, tl)
    print_report(m, title=f"Test set 5-class ({args.scheme})")

    # Secondary 4-class AAMI view (F folded into V) — see fold4_metrics docstring.
    m4, names4 = fold4_metrics(tp, tl)
    print_report(m4, title=f"Test set 4-class N/S/V/Q ({args.scheme}, F->V)",
                 names=names4)

    # 3-class N/S/V view for direct comparison with the matched-filter CNN.
    m3, names3 = fold3_metrics(tp, tl)
    print_report(m3, title=f"Test set 3-class N/S/V ({args.scheme}, F->V)",
                 names=names3)

    def pack(mm):
        return dict(accuracy=mm['accuracy'], f1_macro=mm['f1_macro'],
                    per_class={c: {k: float(v) for k, v in d.items()}
                               for c, d in mm['per_class'].items()},
                    confusion_matrix=mm['confusion_matrix'].tolist())

    results = dict(model=args.model, scheme=args.scheme, params=n_params,
                   best_epoch=best_epoch,
                   five_class=pack(m), four_class=pack(m4),
                   three_class=pack(m3),
                   # keep top-level 5-class for backward compat
                   accuracy=m['accuracy'], f1_macro=m['f1_macro'])
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] {args.output_dir}/  "
          f"5-class acc={m['accuracy']:.4f} F1={m['f1_macro']:.4f}  |  "
          f"4-class acc={m4['accuracy']:.4f} F1={m4['f1_macro']:.4f}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description='Train MIT-BIH beat classifier')
    p.add_argument('--data_dir', type=str, default='../../data/mitdb')
    p.add_argument('--output_dir', type=str, default='./results')
    p.add_argument('--scheme', type=str, default='intra',
                   choices=['intra', 'inter'])
    p.add_argument('--model', type=str, default='baseline',
                   choices=['baseline', 'rr8', 'inception', 'thesis',
                            'deep', 'incep15', 'tcn', 'dualbranch', 'tiny'])
    p.add_argument('--param_budget', type=int, default=15000,
                   help='hard cap on trainable params (lightweight constraint)')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--loss', type=str, default='focal',
                   choices=['ce', 'focal'])
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--weight_temp', type=float, default=0.5,
                   help='class-weight temperature: 1.0=inverse-freq, '
                        '0.5=sqrt (compresses the 114x spread to ~11x)')
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=12,
                   help='early-stop patience (0 = disabled)')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--augment', action='store_true', default=False,
                   help='enable minority oversampling+jitter (default off; '
                        'imbalance handled by class-weighted focal loss)')
    p.add_argument('--no_augment', dest='augment', action='store_false')
    p.add_argument('--target_per_class', type=int, default=0,
                   help='oversample minorities up to this count (0 = off)')
    p.add_argument('--aug_minority', type=str, default='',
                   help='comma classes for on-the-fly jitter, e.g. "S,F"')
    p.add_argument('--aug_p', type=float, default=0.5,
                   help='per-epoch jitter probability for aug_minority beats')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.target_per_class == 0:
        args.target_per_class = None
    train(args)
