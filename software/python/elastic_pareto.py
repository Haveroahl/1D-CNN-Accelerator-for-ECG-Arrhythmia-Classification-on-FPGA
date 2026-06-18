"""elastic_pareto.py — Trục A: latency–accuracy(–energy) Pareto across channel
topologies on the SAME runtime-reconfigurable bitstream.

Idea: the runtime-reconfig mechanism (per-layer in_ch/cp_en/nb via Avalon, proven
bit-exact for all 1..8/layer by tb_topo_sweep) is not just an enabling trick — it
exposes a family of operating points on ONE bitstream. This script produces the
ACCURACY axis of that family: for each topology (c1,c2,c3,c4), all <= 8 (HW cap),
it trains the production conv model + power-of-2 INT8 and reports float/INT8 acc.

  - float: ECG_1DCNN_Pruned trained from scratch (2-phase Adam, same recipe as
           prune_finetune.finetune, but epochs are parameterised for smoke runs).
  - INT8 : QAT power-of-2 (build_qat_model + qat train + convert_to_int8), the
           canonical project pipeline — nb/w_shift calibrated per topology.
  - (4,4,8,8) reuses the published canonical checkpoint so the anchor matches the
           documented 94.65% / F1 0.9396 deployed model.

Latency per topology is weight-invariant and measured separately by tb_topo_sweep;
merged in via --latency_json at the table step. This file ONLY writes the accuracy
axis + a results JSON. It imports model/helpers and touches NO production file.

Usage:
  python elastic_pareto.py                         # full 5-topology sweep
  python elastic_pareto.py --smoke                 # (2,2,2,2) only, tiny epochs
  python elastic_pareto.py --topos 2,2,2,2 4,4,4,4 # explicit subset
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
from quantization.qat_int8 import (
    ECG_1DCNN_QAT, build_qat_model, convert_to_int8, evaluate_int8,
)

CONV_LAYERS = ['conv1', 'conv2', 'conv3', 'conv4']
CANON_CKPT = './results/qat_int8/model_qat_int8.pth'   # anchor for (4,4,8,8)


# ── trainers (self-contained; mirror baseline recipe, epoch-parameterised) ──

def float_train(model, train_loader, val_loader, device, phases):
    """2-phase Adam float training (same schedule shape as prune_finetune.finetune).
    phases = [(n_epochs, lr), ...]; best-by-val-loss checkpoint restored."""
    crit = nn.CrossEntropyLoss()
    best_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    ep = 0
    for n_ep, lr in phases:
        opt = optim.Adam(model.parameters(), lr=lr)
        for _ in range(n_ep):
            ep += 1
            model.train()
            for ecg, y, _ in train_loader:
                ecg = ecg.unsqueeze(1).float().to(device)
                y = y.to(device)
                opt.zero_grad()
                loss = crit(model(ecg), y)
                loss.backward()
                opt.step()
            # val loss
            model.eval()
            vl, n = 0.0, 0
            with torch.no_grad():
                for ecg, y, _ in val_loader:
                    ecg = ecg.unsqueeze(1).float().to(device)
                    y = y.to(device)
                    vl += crit(model(ecg), y).item() * ecg.size(0)
                    n += y.size(0)
            vl /= n
            if vl < best_loss:
                best_loss = vl
                best_state = copy.deepcopy(model.state_dict())
        print(f"    float phase lr={lr:.0e} done (ep {ep}), best_val_loss={best_loss:.4f}")
    model.load_state_dict(best_state)
    return model


def qat_train(qat_model, train_loader, val_loader, device, epochs, lr):
    """QAT fake-quant fine-tune; best-by-val-acc restored (matches qat_int8.run)."""
    opt = optim.Adam(qat_model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state = copy.deepcopy(qat_model.state_dict())
    for _ in range(epochs):
        qat_model.train()
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            opt.zero_grad()
            loss = crit(qat_model(x, quantize=True), y)
            loss.backward()
            opt.step()
        qat_model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for batch in val_loader:
                preds.extend(qat_model(batch[0].to(device), quantize=True).argmax(1).cpu().numpy())
                labs.extend(batch[1].numpy())
        acc = (np.array(preds) == np.array(labs)).mean()
        if acc >= best_acc:
            best_acc = acc
            best_state = copy.deepcopy(qat_model.state_dict())
    qat_model.load_state_dict(best_state)
    print(f"    QAT done, best_val_acc={best_acc:.4f}")
    return qat_model


# ── per-topology driver ─────────────────────────────────────────────────────

def conv_weight_count(w_int8):
    return int(sum(w_int8[n].size for n in CONV_LAYERS))


def run_topology(ch, loaders, device, f_phases, qat_epochs, qat_lr, use_anchor):
    train_loader, val_loader, test_loader = loaders
    ch_t = tuple(ch)
    print(f"\n{'='*60}\n  Topology {ch_t}\n{'='*60}")

    if ch_t == (4, 4, 8, 8) and use_anchor and os.path.exists(CANON_CKPT):
        print(f"  [anchor] reusing canonical checkpoint {CANON_CKPT}")
        ck = torch.load(CANON_CKPT, map_location=device, weights_only=False)
        qat = ECG_1DCNN_QAT(c1_out=ck['c1_out'], c2_out=ck['c2_out'],
                            c3_out=ck['c3_out'], c4_out=ck['c4_out']).to(device)
        qat.load_state_dict(ck['model_state_dict'])
        qat.eval()
        w_int8 = {k: np.array(v, dtype=np.int8) for k, v in ck['w_int8'].items()}
        b_int8 = {k: np.array(v, dtype=np.float64) for k, v in ck['b_int8'].items()}
        w_shift, nb, input_shift = ck['w_shift'], ck['nb'], ck['input_shift_bits']
        # float acc = QAT model evaluated without fake-quant (the trained float weights)
        fp, fl = _eval_qat_float(qat, test_loader, device)
        m_float = compute_metrics(fp, fl, CLASS_NAMES)
        anchor = True
    else:
        torch.manual_seed(42)
        np.random.seed(42)
        model = ECG_1DCNN_Pruned(*ch_t).to(device)
        print(f"  float params: {model.count_parameters()}")
        model = float_train(model, train_loader, val_loader, device, f_phases)
        model.eval()
        fp, fl = evaluate_model(model, test_loader, device)
        m_float = compute_metrics(fp, fl, CLASS_NAMES)

        qat = build_qat_model(model).to(device)
        qat = qat_train(qat, train_loader, val_loader, device, qat_epochs, qat_lr)
        w_int8, b_int8, w_shift, nb, input_shift = convert_to_int8(
            qat, train_loader, device, n_cal_batches=20)
        anchor = False

    int8_acc, p, l = evaluate_int8(qat, test_loader, w_int8, b_int8,
                                   w_shift, nb, input_shift, device)
    m_int8 = compute_metrics(p, l, CLASS_NAMES)
    cw = conv_weight_count(w_int8)

    print(f"  float : acc={m_float['accuracy']:.4f}  f1={m_float['f1_macro']:.4f}")
    print(f"  INT8  : acc={int8_acc:.4f}  f1={m_int8['f1_macro']:.4f}  "
          f"conv_weights={cw}  nb={nb}")

    return {
        'topology': list(ch_t),
        'anchor': anchor,
        'float': {'acc': float(m_float['accuracy']), 'f1': float(m_float['f1_macro'])},
        'int8_p2': {'acc': float(int8_acc), 'f1': float(m_int8['f1_macro'])},
        'conv_weights': cw,
        'w_shift': {k: int(v) for k, v in w_shift.items()},
        'nb': {k: int(v) for k, v in nb.items()},
        'input_shift': int(input_shift),
    }


def _eval_qat_float(qat_model, test_loader, device):
    qat_model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = qat_model(batch[0].to(device), quantize=False)
            preds.extend(logits.argmax(1).cpu().numpy())
            labs.extend(batch[1].numpy())
    return np.array(preds), np.array(labs)


# ── main ────────────────────────────────────────────────────────────────────

def parse_topos(items, allow_oversize=False):
    out = []
    for s in items:
        parts = tuple(int(x) for x in s.split(','))
        assert len(parts) == 4, f"topology {s} must be 4 ints"
        if allow_oversize:
            # software-only ceiling probe: >8/layer does NOT run on the 8-CP-block
            # bitstream (off-bitstream reference for "does more capacity help?").
            assert all(p >= 1 for p in parts), f"topology {s} channels must be >=1"
            if any(p > 8 for p in parts):
                print(f"  [off-bitstream] {parts} exceeds 8/layer HW cap — "
                      f"software accuracy probe only, no bitstream latency")
        else:
            assert all(1 <= p <= 8 for p in parts), (
                f"topology {s} must be 4 ints in 1..8 (hardware cap); "
                f"use --allow_oversize for a software-only ceiling probe")
        out.append(parts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='../../data/Chapman')
    ap.add_argument('--output_dir', default='./results/elastic_pareto')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--topos', nargs='+', default=None,
                    help='e.g. --topos 2,2,2,2 4,4,8,8 ; default = full set')
    ap.add_argument('--qat_epochs', type=int, default=30)
    ap.add_argument('--qat_lr', type=float, default=1e-4)
    ap.add_argument('--no_anchor', action='store_true',
                    help='retrain (4,4,8,8) instead of reusing canonical ckpt')
    ap.add_argument('--smoke', action='store_true',
                    help='(2,2,2,2) only with tiny epochs to validate plumbing')
    ap.add_argument('--allow_oversize', action='store_true',
                    help='permit >8/layer (software-only ceiling probe, off-bitstream)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.smoke:
        topos = [(2, 2, 2, 2)]
        f_phases = [(2, 1e-3), (1, 1e-4)]
        qat_epochs = 2
    else:
        topos = parse_topos(args.topos, args.allow_oversize) if args.topos else \
            [(2, 2, 2, 2), (2, 2, 4, 4), (4, 4, 4, 4), (4, 4, 8, 8), (8, 8, 8, 8)]
        f_phases = [(30, 1e-3), (20, 1e-4)]
        qat_epochs = args.qat_epochs

    print(f"[INFO] device={device}  topos={topos}  smoke={args.smoke}")
    loaders = get_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=2)

    results = []
    t0 = time.time()
    for ch in topos:
        results.append(run_topology(ch, loaders, device, f_phases, qat_epochs,
                                    args.qat_lr, use_anchor=not args.no_anchor))

    out = {
        'baseline_int8': {'acc': 0.9465, 'f1': 0.9396, 'conv_weights': 580,
                          'topology': [4, 4, 8, 8], 'note': 'documented deployed model'},
        'recipe': {'float_phases': f_phases, 'qat_epochs': qat_epochs,
                   'qat_lr': args.qat_lr, 'quant': 'power-of-2 round-half-up'},
        'points': results,
        'wall_time_s': round(time.time() - t0, 1),
    }
    out_path = os.path.join(args.output_dir,
                            'smoke.json' if args.smoke else 'pareto_accuracy.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*60}\n  PARETO ACCURACY AXIS  (baseline INT8 94.65% / 580 w)\n{'='*60}")
    print(f"  {'topology':<16}{'float':>8}{'INT8p2':>8}{'conv_w':>8}")
    for r in results:
        print(f"  {str(tuple(r['topology'])):<16}{r['float']['acc']:>8.4f}"
              f"{r['int8_p2']['acc']:>8.4f}{r['conv_weights']:>8}")
    print(f"\n  -> {out_path}  ({out['wall_time_s']}s)")


if __name__ == '__main__':
    main()
