"""diag_capacity.py — why does from-scratch (8,8,8,8) underperform the (4,4,8,8)
production anchor? Trains float from scratch (same 2-phase recipe, seed 42) for a
set of topologies and logs train_acc/val_acc per epoch + final train-vs-test gap.

Reads:
  - train-test gap large (train>>test)  => overfit (capacity > data needs)
  - train ~ test, both low              => underfit / optimization (model can't
                                           even fit train -> LR/epochs/init, not
                                           a capacity defect)
  - (4,4,8,8) from-scratch test vs documented anchor 94.08% float
                                        => how much of the anchor edge is the
                                           production pipeline (prune-transfer)

float-only (INT8 ~= float here, so the gap lives in the float model). No
production file touched.
"""

import os
import sys
import copy
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prune_finetune import ECG_1DCNN_Pruned
from utils.dataset import get_dataloaders, CLASS_NAMES
from utils.evaluate import evaluate_model, compute_metrics


def acc_on(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for ecg, y, _ in loader:
            ecg = ecg.unsqueeze(1).float().to(device)
            y = y.to(device)
            correct += model(ecg).argmax(1).eq(y).sum().item()
            total += y.size(0)
    return correct / total


def train_logged(model, train_loader, val_loader, device, phases, tag):
    crit = nn.CrossEntropyLoss()
    best_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    curve = []
    ep = 0
    print(f"\n  [{tag}] epoch  train_acc  val_acc  (val_loss)")
    for n_ep, lr in phases:
        opt = optim.Adam(model.parameters(), lr=lr)
        for _ in range(n_ep):
            ep += 1
            model.train()
            tr_correct = tr_total = 0
            for ecg, y, _ in train_loader:
                ecg = ecg.unsqueeze(1).float().to(device)
                y = y.to(device)
                opt.zero_grad()
                out = model(ecg)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                tr_correct += out.argmax(1).eq(y).sum().item()
                tr_total += y.size(0)
            tr_acc = tr_correct / tr_total
            # val loss + acc
            model.eval()
            vl, n, v_correct = 0.0, 0, 0
            with torch.no_grad():
                for ecg, y, _ in val_loader:
                    ecg = ecg.unsqueeze(1).float().to(device)
                    y = y.to(device)
                    out = model(ecg)
                    vl += crit(out, y).item() * ecg.size(0)
                    v_correct += out.argmax(1).eq(y).sum().item()
                    n += y.size(0)
            vl /= n
            v_acc = v_correct / n
            if vl < best_loss:
                best_loss = vl
                best_state = copy.deepcopy(model.state_dict())
            curve.append({'epoch': ep, 'train_acc': tr_acc, 'val_acc': v_acc, 'val_loss': vl})
            if ep % 5 == 0 or ep == 1:
                print(f"  [{tag}] {ep:>3}    {tr_acc:.4f}    {v_acc:.4f}   ({vl:.4f})")
    model.load_state_dict(best_state)
    return model, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='../../data/Chapman')
    ap.add_argument('--output_dir', default='./results/elastic_pareto/diag')
    ap.add_argument('--topos', nargs='+', default=['4,4,8,8', '8,8,8,8'])
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loaders = get_dataloaders(args.data_dir, batch_size=128, num_workers=2)
    train_loader, val_loader, test_loader = loaders
    phases = [(30, 1e-3), (20, 1e-4)]

    out = []
    for s in args.topos:
        ch = tuple(int(x) for x in s.split(','))
        torch.manual_seed(42)
        np.random.seed(42)
        model = ECG_1DCNN_Pruned(*ch).to(device)
        params = model.count_parameters()
        model, curve = train_logged(model, train_loader, val_loader, device, phases, s)
        # final train vs test (best-val-loss checkpoint)
        tr_acc = acc_on(model, train_loader, device)
        tp, tl = evaluate_model(model, test_loader, device)
        m = compute_metrics(tp, tl, CLASS_NAMES)
        te_acc = m['accuracy']
        gap = tr_acc - te_acc
        rec = {'topology': list(ch), 'params': params,
               'final_train_acc': round(tr_acc, 4), 'final_test_acc': round(te_acc, 4),
               'test_f1': round(m['f1_macro'], 4), 'train_test_gap': round(gap, 4),
               'max_train_acc': round(max(c['train_acc'] for c in curve), 4),
               'curve': curve}
        out.append(rec)
        verdict = ('OVERFIT (train>>test)' if gap > 0.03 else
                   'underfit/plateau (train~test)' if tr_acc < 0.96 else
                   'good fit')
        print(f"\n  [{s}] params={params}  train={tr_acc:.4f}  test={te_acc:.4f}  "
              f"gap={gap:+.4f}  max_train={rec['max_train_acc']:.4f}  -> {verdict}")

    with open(os.path.join(args.output_dir, 'diag_capacity.json'), 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*64}\n  CAPACITY DIAGNOSIS (from-scratch, seed 42, 50 float ep)\n{'='*64}")
    print(f"  {'topology':<14}{'params':>8}{'train':>9}{'test':>9}{'gap':>9}")
    for r in out:
        print(f"  {str(tuple(r['topology'])):<14}{r['params']:>8}"
              f"{r['final_train_acc']:>9.4f}{r['final_test_acc']:>9.4f}"
              f"{r['train_test_gap']:>+9.4f}")
    print(f"\n  anchor (4,4,8,8) production float = 0.9408 / INT8 0.9465")


if __name__ == '__main__':
    main()
