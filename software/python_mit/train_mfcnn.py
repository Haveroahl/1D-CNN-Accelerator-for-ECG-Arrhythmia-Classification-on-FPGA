"""
Train the Matched-Filter CNN — Model 6 (Sensors 2023, 23/3/1365)
================================================================
Reproduces the paper's best 3-class N/S/V inter-patient result
(target: acc ~98.18%, F1-macro ~92.17%).

  - input: 128 Hz derivative beats, 4 RR features (local-80 / global-400)
  - conv: 13 matched-filter templates (DS1 mean derivative beats), FROZEN
  - inter-patient de Chazal DS1/DS2; best ckpt by val F1-macro

Usage:
  python train_mfcnn.py --epochs 80
  python train_mfcnn.py --trainable_conv      # Model-1-style (conv fine-tuned)
"""

import os, sys, argparse, csv, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from dataset_mfcnn import (get_dataloaders, build_mf_templates, class_weights,
                          CLASS_NAMES, NUM_CLASSES)
from model.mfcnn import build_mfcnn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma, self.weight = gamma, weight

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target, weight=self.weight, reduction='none')
        pt = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        return ((1 - pt) ** self.gamma * ce).mean()


def _metrics(cm):
    acc = np.trace(cm) / max(cm.sum(), 1)
    f1s, per = [], []
    for i in range(cm.shape[0]):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        r = tp / (tp + fn) if (tp + fn) else 0.0
        p = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per.append((p, r, f1, int(cm[i, :].sum())))
        if cm[i, :].sum() > 0:
            f1s.append(f1)
    return float(acc), float(np.mean(f1s)), per


def evaluate(model, loader, device):
    model.eval()
    P, L = [], []
    with torch.no_grad():
        for x, rr, y in loader:
            o = model(x.to(device), rr.to(device))
            P.extend(o.argmax(1).cpu().numpy()); L.extend(y.numpy())
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    for t, p in zip(L, P):
        cm[t][p] += 1
    return cm


def print_cm(cm, title):
    acc, f1, per = _metrics(cm)
    print(f"\n  {title}:  acc={acc:.4f}  F1-macro={f1:.4f}")
    print(f"  {'cls':<6}{'P':>8}{'R':>8}{'F1':>8}{'sup':>8}")
    for i, n in enumerate(CLASS_NAMES):
        p, r, f, s = per[i]
        print(f"  {n:<6}{p:>8.3f}{r:>8.3f}{f:>8.3f}{s:>8d}")
    print("  confusion (rows=true, cols=pred):")
    print("        " + "".join(f"{n:>7}" for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:<5}" + "".join(f"{int(cm[i,j]):>7d}" for j in range(NUM_CLASSES)))
    return acc, f1


def train(args):
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device={device} seed={args.seed}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] building matched-filter templates from DS1 ...")
    templates, present = build_mf_templates(args.data_dir, seed=args.seed)

    tr_loader, va_loader, te_loader, tr_ds = get_dataloaders(
        args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed)

    model = build_mfcnn(templates=templates,
                        freeze_conv=not args.trainable_conv,
                        normalize_templates=args.normalize_templates).to(device)
    print(model.layer_summary())

    w = class_weights(tr_ds, args.weight_temp).to(device)
    print(f"[INFO] class weights={w.cpu().numpy().round(3)}  loss={args.loss}")
    crit = (FocalLoss(args.focal_gamma, w) if args.loss == 'focal'
            else nn.CrossEntropyLoss(weight=w))      # paper: weighted CE

    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log = os.path.join(args.output_dir, 'train_log.csv')
    with open(log, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'loss', 'val_f1', 'lr', 'time_s', 'best'])
    best_f1, best_ep, noimp = -1.0, 0, 0
    ckpt = os.path.join(args.output_dir, 'best_model.pth')

    for ep in range(1, args.epochs + 1):
        t0 = time.time(); model.train(); tot = 0.0
        for x, rr, y in tr_loader:
            x, rr, y = x.to(device), rr.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x, rr), y)
            loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        tot /= len(tr_loader.dataset)
        _, v_f1, _ = _metrics(evaluate(model, va_loader, device))
        lr = opt.param_groups[0]['lr']; dt = time.time() - t0
        is_best = v_f1 > best_f1
        if is_best:
            best_f1, best_ep, noimp = v_f1, ep, 0
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                        'val_f1': v_f1, 'rr_stats': tr_ds.rr_stats}, ckpt)
        else:
            noimp += 1
        print(f"  Ep{ep:3d}/{args.epochs} loss{tot:.4f} valF1{v_f1:.4f} "
              f"lr{lr:.1e} {dt:.1f}s" + (" *" if is_best else ""))
        with open(log, 'a', newline='') as f:
            csv.writer(f).writerow([ep, f"{tot:.6f}", f"{v_f1:.6f}",
                                    f"{lr:.1e}", f"{dt:.1f}", int(is_best)])
        sched.step()
        if args.patience and noimp >= args.patience:
            print(f"  [early-stop] {args.patience} epochs no val-F1 gain"); break

    print(f"\n  Best epoch {best_ep}  val_f1={best_f1:.4f}")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    cm = evaluate(model, te_loader, device)
    acc, f1 = print_cm(cm, "TEST 3-class N/S/V (inter de Chazal)")

    res = dict(params_trainable=model.count_parameters(),
               params_total=model.count_all(), best_epoch=best_ep,
               accuracy=acc, f1_macro=f1, confusion_matrix=cm.tolist(),
               frozen_conv=not args.trainable_conv,
               paper_target=dict(accuracy=0.9818, f1_macro=0.9217))
    json.dump(res, open(os.path.join(args.output_dir, 'results.json'), 'w'),
              indent=2)
    print(f"\n[DONE] {args.output_dir}/  acc={acc:.4f} F1={f1:.4f}  "
          f"(paper Model 6: acc=0.9818 F1=0.9217)")
    return res


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='../../data/mitdb')
    p.add_argument('--output_dir', default='./results/mfcnn_model6')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--loss', type=str, default='ce', choices=['ce', 'focal'],
                   help='ce=weighted cross-entropy (paper); focal=focal loss')
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--weight_temp', type=float, default=1.0,
                   help='class-weight temperature; 1.0 = plain inverse-freq (paper)')
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--trainable_conv', action='store_true',
                   help='unfreeze conv (Model-1 style); default Model 6 frozen')
    p.add_argument('--normalize_templates', action='store_true',
                   help='L2-normalize MF templates (NOT paper); default raw')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
