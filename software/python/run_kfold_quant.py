"""5 test-fold robustness for the quantization ablation (Table 4 mean +/- std).

SCOPE / HONEST LABEL:
  This is NOT a full 5-fold cross-validation. Per the chosen design, a single
  float pruned model (best_model_pruned.pth, trained on the seed=42 split) is
  reused for every fold; only the quantization step (PTQ calibrate / QAT
  fine-tune) and the test split change per fold.

  => It measures the variance of the QUANTIZATION procedure and the TEST split,
     NOT generalization variance of the float model. The float net has seen all
     records during its original training, so there IS train/test leakage at the
     float level. Report this as "5 test-fold robustness", not "5-fold CV".

  A full leak-free 5-fold would retrain the float model from scratch per fold.

Variants (all bit-exact INT8 eval):
  A0  P2-PTQ           — calibrate power-of-2, no training
  A0' general-PTQ      — calibrate general scale, no training
  A2  P2-QAT (ours)    — fake-quant fine-tune, power-of-2
  A3  general-QAT      — fake-quant fine-tune, general scale
  A4  P2-QAT floor     — power-of-2, floor rescale

Usage:
    cd software/python
    python run_kfold_quant.py \\
        --pruned_checkpoint ./results/best_model_pruned.pth \\
        --data_dir ../../data/Chapman \\
        --output_dir ./results/ablation_quant/kfold \\
        --folds 5 --epochs 50
"""

import os
import sys
import argparse
import json
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prune_finetune import ECG_1DCNN_Pruned
from utils.dataset import ChapmanECGDataset, CLASS_NAMES
from utils.evaluate import compute_metrics

from quantization.qat_int8 import (
    ECG_1DCNN_QAT, build_qat_model as build_qat_p2,
    convert_to_int8, int8_forward,
)
from quantization.qat_int8_general import (
    ECG_1DCNN_QAT_General, build_qat_model as build_qat_gen,
    convert_to_int8_general, int8_forward_general, estimate_dsp,
)
from quantization.ptq_int8_general import calibrate_scales


# ============================================================
#  Data: load once, slice per fold
# ============================================================

def load_all(data_dir):
    ds = ChapmanECGDataset(data_dir, split='all')
    X = np.stack(ds.records).astype(np.float32)      # (N, 2500)
    y = np.array(ds.labels, dtype=np.int64)
    return X, y


def make_loaders(X, y, train_idx, test_idx, batch_size):
    def loader(idx, shuffle, drop_last):
        xb = torch.from_numpy(X[idx])
        yb = torch.from_numpy(y[idx])
        # dataset returns (ecg, label) — int8_forward/evaluate unpack batch[0], batch[1]
        ds = TensorDataset(xb, yb)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    train_loader = loader(train_idx, True,  True)
    test_loader  = loader(test_idx,  False, False)
    return train_loader, test_loader


# ============================================================
#  QAT fine-tune (shared by A2/A3/A4)
# ============================================================

def qat_finetune(qat_model, train_loader, device, epochs, lr):
    optimizer = optim.Adam(qat_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_state, best_acc = None, -1.0
    for ep in range(epochs):
        qat_model.train()
        for batch in train_loader:
            x = batch[0].to(device); yb = batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(qat_model(x, quantize=True), yb)
            loss.backward()
            optimizer.step()
        # quick train-acc proxy for best-state selection (no separate val in fold)
        qat_model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in train_loader:
                x = batch[0].to(device); yb = batch[1].to(device)
                correct += (qat_model(x, quantize=True).argmax(1) == yb).sum().item()
                total += yb.numel()
        acc = correct / max(total, 1)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(qat_model.state_dict())
    if best_state is not None:
        qat_model.load_state_dict(best_state)
    return qat_model


# ============================================================
#  Per-variant fold evaluation
# ============================================================

def eval_a0_p2(pruned_ckpt, train_loader, test_loader, device):
    base = build_pruned(pruned_ckpt, device)
    qm = build_qat_p2(base).to(device); qm.eval()
    w8, b8, wsh, nb, ish = convert_to_int8(qm, train_loader, device, n_cal_batches=20)
    return run_eval_p2(qm, test_loader, w8, b8, wsh, nb, ish, device)


def eval_a0_general(pruned_ckpt, train_loader, test_loader, device):
    base = build_pruned(pruned_ckpt, device)
    qm = build_qat_gen(base).to(device); qm.eval()
    w8, bf, ws, xin, xout, ish = calibrate_scales(qm, train_loader, device, n_cal_batches=20)
    return run_eval_general(qm, test_loader, w8, bf, ws, xin, xout, ish, 'round', device)


def eval_a2_p2(pruned_ckpt, train_loader, test_loader, device, epochs, lr):
    base = build_pruned(pruned_ckpt, device)
    qm = build_qat_p2(base).to(device)
    qm = qat_finetune(qm, train_loader, device, epochs, lr)
    qm.eval()
    w8, b8, wsh, nb, ish = convert_to_int8(qm, train_loader, device, n_cal_batches=20)
    return run_eval_p2(qm, test_loader, w8, b8, wsh, nb, ish, device)


def eval_a3_general(pruned_ckpt, train_loader, test_loader, device, epochs, lr):
    base = build_pruned(pruned_ckpt, device)
    qm = build_qat_gen(base).to(device)
    qm = qat_finetune(qm, train_loader, device, epochs, lr)
    qm.eval()
    w8, bf, ws, xin, xout, ish = convert_to_int8_general(qm, train_loader, device, n_cal_batches=20)
    return run_eval_general(qm, test_loader, w8, bf, ws, xin, xout, ish, 'round', device)


def eval_a4_p2floor(pruned_ckpt, train_loader, test_loader, device, epochs, lr):
    # A4 = power-of-2 QAT but floor rescale at eval. Train with the same P2 QAT
    # model; only the eval rescale differs (floor instead of round-half-up).
    base = build_pruned(pruned_ckpt, device)
    qm = build_qat_p2(base).to(device)
    qm = qat_finetune(qm, train_loader, device, epochs, lr)
    qm.eval()
    w8, b8, wsh, nb, ish = convert_to_int8(qm, train_loader, device, n_cal_batches=20)
    return run_eval_p2(qm, test_loader, w8, b8, wsh, nb, ish, device, floor=True)


# ---- shared eval helpers (return acc, f1) ----

def build_pruned(ckpt, device):
    m = ECG_1DCNN_Pruned(
        c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
        c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
    )
    m.load_state_dict(ckpt['model_state_dict'])
    return m.to(device).eval()


def _floor_shift_patch(enable):
    """Toggle round_shift in qat_int8 between round-half-up and floor."""
    import quantization.qat_int8 as q
    if enable:
        if not hasattr(q, '_orig_round_shift'):
            q._orig_round_shift = q.round_shift
        def floor_shift(x, n):
            return torch.floor(x / (2.0 ** n)) if n > 0 else x
        q.round_shift = floor_shift
    else:
        if hasattr(q, '_orig_round_shift'):
            q.round_shift = q._orig_round_shift


def run_eval_p2(qm, test_loader, w8, b8, wsh, nb, ish, device, floor=False):
    _floor_shift_patch(floor)
    try:
        preds, labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(device)
                logits = int8_forward(qm, x, w8, b8, wsh, nb, ish)
                preds.extend(logits.argmax(1).cpu().numpy())
                labels.extend(batch[1].numpy())
    finally:
        _floor_shift_patch(False)
    preds, labels = np.array(preds), np.array(labels)
    acc = (preds == labels).mean()
    f1 = compute_metrics(preds, labels, CLASS_NAMES)['f1_macro']
    return float(acc), float(f1)


def run_eval_general(qm, test_loader, w8, bf, ws, xin, xout, ish, mode, device):
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            logits = int8_forward_general(qm, x, w8, bf, ws, xin, xout, ish, mode)
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(batch[1].numpy())
    preds, labels = np.array(preds), np.array(labels)
    acc = (preds == labels).mean()
    f1 = compute_metrics(preds, labels, CLASS_NAMES)['f1_macro']
    return float(acc), float(f1)


# ============================================================
#  Main
# ============================================================

VARIANTS = [
    ('A0_ptq_p2',        '+0', False),
    ("A0g_ptq_general",  '+4', False),
    ('A2_qat_p2',        '+0', True),
    ('A3_qat_general',   '+4', True),
    ('A4_qat_p2_floor',  '+0', True),
]


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device} | folds={args.folds} | epochs={args.epochs}")
    print(f"[INFO] SCOPE: 5 test-fold robustness (shared float pruned model, "
          f"re-quant per fold). NOT leak-free 5-fold CV.")

    pruned_ckpt = torch.load(args.pruned_checkpoint, map_location=device, weights_only=False)

    print("[INFO] Loading all Chapman records once ...")
    X, y = load_all(args.data_dir)
    print(f"[INFO] {len(X)} records loaded.")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    # results[variant] = list of (acc, f1) per fold
    results = {name: [] for name, _, _ in VARIANTS}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}\n  FOLD {fold+1}/{args.folds}  "
              f"(train={len(train_idx)}, test={len(test_idx)})\n{'='*60}")
        train_loader, test_loader = make_loaders(X, y, train_idx, test_idx, args.batch_size)

        a, f = eval_a0_p2(pruned_ckpt, train_loader, test_loader, device)
        results['A0_ptq_p2'].append((a, f)); print(f"  A0  P2-PTQ        acc={a:.4f} f1={f:.4f}")

        a, f = eval_a0_general(pruned_ckpt, train_loader, test_loader, device)
        results['A0g_ptq_general'].append((a, f)); print(f"  A0' gen-PTQ       acc={a:.4f} f1={f:.4f}")

        a, f = eval_a2_p2(pruned_ckpt, train_loader, test_loader, device, args.epochs, args.lr)
        results['A2_qat_p2'].append((a, f)); print(f"  A2  P2-QAT        acc={a:.4f} f1={f:.4f}")

        a, f = eval_a3_general(pruned_ckpt, train_loader, test_loader, device, args.epochs, args.lr)
        results['A3_qat_general'].append((a, f)); print(f"  A3  gen-QAT       acc={a:.4f} f1={f:.4f}")

        a, f = eval_a4_p2floor(pruned_ckpt, train_loader, test_loader, device, args.epochs, args.lr)
        results['A4_qat_p2_floor'].append((a, f)); print(f"  A4  P2-QAT floor  acc={a:.4f} f1={f:.4f}")

        # checkpoint partial results each fold (resumable insight)
        _dump(results, args)

    _dump(results, args, final=True)


def _dump(results, args, final=False):
    summary = {}
    for name, dsp, _ in VARIANTS:
        vals = results[name]
        if not vals:
            continue
        accs = np.array([v[0] for v in vals]); f1s = np.array([v[1] for v in vals])
        summary[name] = {
            'dsp_extra': dsp,
            'n_folds': len(vals),
            'acc_mean': float(accs.mean()), 'acc_std': float(accs.std()),
            'f1_mean': float(f1s.mean()),   'f1_std': float(f1s.std()),
            'acc_per_fold': accs.round(4).tolist(),
            'f1_per_fold': f1s.round(4).tolist(),
        }
    with open(os.path.join(args.output_dir, 'kfold_summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'scope': '5 test-fold robustness (shared float pruned, re-quant per fold; NOT leak-free CV)',
                   'variants': summary}, f, indent=2)
    if final:
        txt = os.path.join(args.output_dir, 'kfold_table.txt')
        with open(txt, 'w', encoding='utf-8') as f:
            f.write("TABLE 4 (5 test-fold) - Quantization Ablation, mean +/- std\n")
            f.write("Scope: shared float pruned model, re-quant per fold (NOT leak-free CV)\n")
            f.write("="*72 + "\n")
            f.write(f"{'Variant':<20} {'Acc% (mean+/-std)':>22} {'F1 (mean+/-std)':>20} {'DSP':>5}\n")
            f.write("-"*72 + "\n")
            for name, dsp, _ in VARIANTS:
                s = summary.get(name)
                if not s: continue
                f.write(f"{name:<20} {s['acc_mean']*100:>7.2f} +/- {s['acc_std']*100:>4.2f}      "
                        f"{s['f1_mean']:>6.4f} +/- {s['f1_std']:>5.4f}   {dsp:>4}\n")
            f.write("="*72 + "\n")
        print(f"\n[INFO] Saved {txt}")
        # also print to console
        print(open(txt, encoding='utf-8').read())


def main():
    p = argparse.ArgumentParser(description="5 test-fold robustness for quant ablation")
    p.add_argument('--pruned_checkpoint', type=str, required=True)
    p.add_argument('--data_dir',   type=str, default='../../data/Chapman')
    p.add_argument('--output_dir', type=str, default='./results/ablation_quant/kfold')
    p.add_argument('--folds',      type=int, default=5)
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=128)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
