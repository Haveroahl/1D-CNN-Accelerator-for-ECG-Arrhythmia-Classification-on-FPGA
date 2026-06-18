"""prune_pareto.py — DEPLOYMENT Pareto: best-deployable model at each operating
point, using the SAME production recipe as the shipped (4,4,8,8) model.

The shipped model = train full (4,8,8,16) -> prune-transfer -> finetune -> QAT =
94.65% INT8. This script extends that recipe to the other on-bitstream operating
points so the frontier reflects what you would actually DEPLOY (not lazily
from-scratch-trained models):

  (2,2,2,2) (2,2,4,4) (4,4,4,4) : prune-transfer from the SAME full (4,8,8,16)
                                  parent (all are channel subsets) -> finetune
                                  -> QAT power-of-2.  method = prune-transfer
  (4,4,8,8)                     : the shipped canonical checkpoint (94.65%).
                                  method = production-anchor
  (8,8,8,8)                     : from-scratch — conv1=8 > 4 of the parent, so it
                                  is NOT a subset and cannot be prune-grown; its
                                  honest best-effort is from-scratch. method =
                                  from-scratch

Reuses prune_finetune (ranking + finetune) and qat_int8 (QAT/INT8) WITHOUT
modifying them. Reads the (4,4,8,8) anchor and (8,8,8,8) from-scratch numbers from
the existing elastic_pareto run.
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.model import ECG_1DCNN
from prune_finetune import (ECG_1DCNN_Pruned, _l1_rank, _taylor_rank, finetune)
from quantization.qat_int8 import build_qat_model, convert_to_int8, evaluate_int8
from elastic_pareto import qat_train, conv_weight_count
from utils.dataset import get_dataloaders, CLASS_NAMES
from utils.evaluate import evaluate_model, compute_metrics

LAYER_NAMES = ['conv1', 'conv2', 'conv3', 'conv4']
PRUNE_TARGETS = [(2, 2, 2, 2), (2, 2, 4, 4), (4, 4, 4, 4)]   # subsets of (4,8,8,16)
CANON_CKPT = './results/qat_int8/model_qat_int8.pth'


def prune_to(full_model, ch, train_loader, device):
    """Prune-transfer full (4,8,8,16) -> ECG_1DCNN_Pruned(ch). Taylor ranking
    (same as prune_finetune.prune_model) but with arbitrary per-layer keep counts."""
    targets = dict(zip(LAYER_NAMES, ch))
    taylor = _taylor_rank(full_model, train_loader, LAYER_NAMES, device, n_batches=20)
    kept = {}
    for name in LAYER_NAMES:
        keep_n = targets[name]
        if taylor[name] is not None:
            idx = torch.argsort(taylor[name], descending=True)[:keep_n]
        else:
            idx = _l1_rank(getattr(full_model, name).weight.data)[:keep_n]
        kept[name] = torch.sort(idx)[0].cpu()

    pruned = ECG_1DCNN_Pruned(*ch).to(device)
    with torch.no_grad():
        pruned.conv1.weight.copy_(full_model.conv1.weight.data[kept['conv1']])
        pruned.conv1.bias.copy_(full_model.conv1.bias.data[kept['conv1']])
        pruned.conv2.weight.copy_(full_model.conv2.weight.data[kept['conv2']][:, kept['conv1'], :])
        pruned.conv2.bias.copy_(full_model.conv2.bias.data[kept['conv2']])
        pruned.conv3.weight.copy_(full_model.conv3.weight.data[kept['conv3']][:, kept['conv2'], :])
        pruned.conv3.bias.copy_(full_model.conv3.bias.data[kept['conv3']])
        pruned.conv4.weight.copy_(full_model.conv4.weight.data[kept['conv4']][:, kept['conv3'], :])
        pruned.conv4.bias.copy_(full_model.conv4.bias.data[kept['conv4']])
        pruned.fc.weight.copy_(full_model.fc.weight.data[:, kept['conv4']])
        pruned.fc.bias.copy_(full_model.fc.bias.data)
    return pruned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full_ckpt', default='./results/best_model.pth')
    ap.add_argument('--fromscratch_json',
                    default='./results/elastic_pareto/pareto_accuracy.json',
                    help='source for the (4,4,8,8) anchor + (8,8,8,8) from-scratch points')
    ap.add_argument('--output_dir', default='./results/elastic_pareto')
    ap.add_argument('--qat_epochs', type=int, default=30)
    ap.add_argument('--qat_lr', type=float, default=1e-4)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = get_dataloaders(
        '../../data/Chapman', batch_size=128, num_workers=2)

    full = ECG_1DCNN(num_classes=4).to(device)
    full.load_state_dict(torch.load(args.full_ckpt, map_location=device,
                                    weights_only=False)['model_state_dict'])
    full.eval()

    points = []
    for ch in PRUNE_TARGETS:
        print(f"\n{'='*60}\n  prune-transfer (4,8,8,16) -> {ch}\n{'='*60}")
        pruned = prune_to(full, ch, train_loader, device)
        pruned = finetune(pruned, train_loader, val_loader, device)
        pruned.eval()
        fp, fl = evaluate_model(pruned, test_loader, device)
        m_float = compute_metrics(fp, fl, CLASS_NAMES)
        qat = build_qat_model(pruned).to(device)
        qat = qat_train(qat, train_loader, val_loader, device, args.qat_epochs, args.qat_lr)
        w_int8, b_int8, w_shift, nb, input_shift = convert_to_int8(
            qat, train_loader, device, n_cal_batches=20)
        int8_acc, p, l = evaluate_int8(qat, test_loader, w_int8, b_int8,
                                       w_shift, nb, input_shift, device)
        m_int8 = compute_metrics(p, l, CLASS_NAMES)
        cw = conv_weight_count(w_int8)
        print(f"  {ch}: float={m_float['accuracy']:.4f}  INT8={int8_acc:.4f}  "
              f"f1={m_int8['f1_macro']:.4f}  conv_w={cw}")
        points.append({'topology': list(ch), 'method': 'prune-transfer',
                       'float': {'acc': float(m_float['accuracy']),
                                 'f1': float(m_float['f1_macro'])},
                       'int8_p2': {'acc': float(int8_acc),
                                   'f1': float(m_int8['f1_macro'])},
                       'conv_weights': cw})

    # pull (4,4,8,8) anchor + (8,8,8,8) from-scratch from the existing run
    with open(args.fromscratch_json) as f:
        fs = json.load(f)
    for pt in fs['points']:
        ch = tuple(pt['topology'])
        if ch == (4, 4, 8, 8):
            pt = dict(pt); pt['method'] = 'production-anchor'
            points.append(pt)
        elif ch == (8, 8, 8, 8):
            pt = dict(pt); pt['method'] = 'from-scratch (wider than parent)'
            points.append(pt)

    points.sort(key=lambda r: r['conv_weights'])
    out = {'mode': 'deployment',
           'note': 'best-deployable per operating point; same prune-transfer recipe '
                   'as the shipped (4,4,8,8); (8,8,8,8) from-scratch (not a subset '
                   'of the (4,8,8,16) parent)',
           'recipe': {'prune': 'Taylor-rank transfer from full (4,8,8,16)',
                      'finetune': '50ep (30@1e-3 + 20@1e-4)',
                      'qat_epochs': args.qat_epochs, 'quant': 'power-of-2 round-half-up'},
           'points': points}
    out_path = os.path.join(args.output_dir, 'deployment_pareto.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*60}\n  DEPLOYMENT PARETO (best per point)\n{'='*60}")
    print(f"  {'topology':<14}{'INT8':>8}{'conv_w':>8}  method")
    for r in points:
        print(f"  {str(tuple(r['topology'])):<14}{r['int8_p2']['acc']:>8.4f}"
              f"{r['conv_weights']:>8}  {r['method']}")
    print(f"\n  -> {out_path}")


if __name__ == '__main__':
    main()
