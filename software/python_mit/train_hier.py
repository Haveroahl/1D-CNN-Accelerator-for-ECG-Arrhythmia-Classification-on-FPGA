"""
Training — tiny multi-task MIT-BIH 5-symbol classifier (~500 weights)
=====================================================================
Two independent heads on a shared backbone:
  Dense 1 (binary)  : normal {N,L,R} vs abnormal {A,V}
  Dense 2 (5-class) : N / L / R / A / V

Loss = CE(bin) + lambda * CE(5class), lambda>1 prioritizes the harder 5-class
head. Both heads use sqrt-tempered class weights. Best ckpt by 5-class val F1.

Usage:
  python train_hier.py --data_dir ../../data/mitdb --epochs 80
  python train_hier.py --aug_minority A --aug_p 0.5
"""

import os, sys, argparse, csv, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from dataset_hier import (get_dataloaders, class_weights_bin, class_weights_5,
                          CLASS5, BIN_NAMES, N5, SYM5)
from model.model_hier import build_model


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
    K = cm.shape[0]
    acc = np.trace(cm) / max(cm.sum(), 1)
    f1s = []
    per = []
    for i in range(K):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        r = tp / (tp + fn) if (tp + fn) else 0.0
        p = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * r * p / (r + p) if (r + p) else 0.0
        per.append((p, r, f1, int(cm[i, :].sum())))
        if cm[i, :].sum() > 0:
            f1s.append(f1)
    return float(acc), float(np.mean(f1s)), per


def evaluate(model, loader, device):
    model.eval()
    pb, lb, p5, l5 = [], [], [], []
    with torch.no_grad():
        for ecg, rr, ybin, y5 in loader:
            o_bin, o_5 = model(ecg.to(device), rr.to(device))
            pb.extend(o_bin.argmax(1).cpu().numpy()); lb.extend(ybin.numpy())
            p5.extend(o_5.argmax(1).cpu().numpy());   l5.extend(y5.numpy())
    pb, lb, p5, l5 = map(np.array, (pb, lb, p5, l5))
    cm_bin = np.zeros((2, 2), int)
    for t, p in zip(lb, pb): cm_bin[t][p] += 1
    cm_5 = np.zeros((N5, N5), int)
    for t, p in zip(l5, p5): cm_5[t][p] += 1
    return cm_bin, cm_5


def print_cm(cm, names, title):
    acc, f1, per = _metrics(cm)
    print(f"\n  {title}:  acc={acc:.4f}  F1-macro={f1:.4f}")
    print(f"  {'cls':<8}{'P':>8}{'R':>8}{'F1':>8}{'sup':>8}")
    for i, n in enumerate(names):
        p, r, f, s = per[i]
        print(f"  {n:<8}{p:>8.3f}{r:>8.3f}{f:>8.3f}{s:>8d}")
    return acc, f1


def train(args):
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device={device} seed={args.seed} lambda5={args.lambda5}")
    os.makedirs(args.output_dir, exist_ok=True)

    aug = [SYM5[c] for c in args.aug_minority.split(',') if c] \
        if args.aug_minority else None
    oversample = {SYM5['A']: args.oversample_A} if args.oversample_A > 1 else None
    tr_loader, va_loader, te_loader, tr_ds = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        seed=args.seed, aug_minority=aug, aug_p=args.aug_p,
        oversample=oversample, scheme=args.scheme)
    if aug:
        print(f"[INFO] minority aug classes={args.aug_minority} p={args.aug_p}")
    if oversample:
        print(f"[INFO] oversample A x{args.oversample_A} (jittered copies)")

    model = build_model(n_rr=args.n_rr, rr_hidden=args.rr_hidden).to(device)
    n_params = model.count_parameters()
    print(model.layer_summary())
    assert n_params <= args.param_budget, \
        f"{n_params} > budget {args.param_budget}"

    w_bin = class_weights_bin(tr_ds).to(device)
    w_5   = class_weights_5(tr_ds).to(device)
    if args.a_weight_mult != 1.0:                     # extra push on class A (idx 3)
        w_5[SYM5['A']] *= args.a_weight_mult
    print(f"[INFO] w_bin={w_bin.cpu().numpy().round(3)}  "
          f"w_5={w_5.cpu().numpy().round(3)}")
    crit_bin = FocalLoss(args.focal_gamma, w_bin)
    crit_5   = FocalLoss(args.focal_gamma, w_5)

    opt = optim.Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    if args.lr_schedule == 'step':
        # high LR for the first lr_drop epochs, then ×0.1 (e.g. 1e-3 → 1e-4)
        sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[args.lr_drop],
                                               gamma=0.1)
        print(f"[INFO] LR step: {args.lr:.0e} for {args.lr_drop} epochs "
              f"then ×0.1")
    else:
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log = os.path.join(args.output_dir, 'train_log.csv')
    with open(log, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'loss', 'val_bin_f1', 'val_5_f1',
                                 'lr', 'time_s', 'is_best'])
    best_f1, best_ep, noimp = -1.0, 0, 0
    ckpt = os.path.join(args.output_dir, 'best_model.pth')

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_ds.set_epoch(ep)
        model.train()
        tot = 0.0
        for ecg, rr, ybin, y5 in tr_loader:
            ecg, rr = ecg.to(device), rr.to(device)
            ybin, y5 = ybin.to(device), y5.to(device)
            opt.zero_grad()
            o_bin, o_5 = model(ecg, rr)
            loss = crit_bin(o_bin, ybin) + args.lambda5 * crit_5(o_5, y5)
            loss.backward(); opt.step()
            tot += loss.item() * ecg.size(0)
        tot /= len(tr_loader.dataset)

        cmb, cm5 = evaluate(model, va_loader, device)
        _, vb_f1, _ = _metrics(cmb)
        _, v5_f1, _ = _metrics(cm5)
        lr = opt.param_groups[0]['lr']; dt = time.time() - t0

        is_best = v5_f1 > best_f1
        if is_best:
            best_f1, best_ep, noimp = v5_f1, ep, 0
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                        'val_5_f1': v5_f1}, ckpt)
        else:
            noimp += 1
        print(f"  Ep{ep:3d}/{args.epochs} loss{tot:.4f} "
              f"valBinF1{vb_f1:.4f} val5F1{v5_f1:.4f} lr{lr:.1e} {dt:.1f}s"
              + (" *" if is_best else ""))
        with open(log, 'a', newline='') as f:
            csv.writer(f).writerow([ep, f"{tot:.6f}", f"{vb_f1:.6f}",
                                    f"{v5_f1:.6f}", f"{lr:.1e}", f"{dt:.1f}",
                                    int(is_best)])
        sched.step()
        if args.patience and noimp >= args.patience:
            print(f"  [early-stop] {args.patience} epochs no val-5-F1 gain")
            break

    print(f"\n  Best epoch {best_ep}  val_5_f1={best_f1:.4f}")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    cmb, cm5 = evaluate(model, te_loader, device)
    bin_acc, bin_f1 = print_cm(cmb, BIN_NAMES, "TEST Dense1 binary")
    f5_acc, f5_f1   = print_cm(cm5, CLASS5,   "TEST Dense2 5-class")

    res = dict(params=n_params, best_epoch=best_ep,
               binary=dict(accuracy=bin_acc, f1_macro=bin_f1,
                           confusion_matrix=cmb.tolist()),
               five_class=dict(accuracy=f5_acc, f1_macro=f5_f1,
                               confusion_matrix=cm5.tolist()))
    json.dump(res, open(os.path.join(args.output_dir, 'results.json'), 'w'),
              indent=2)
    print(f"\n[DONE] {args.output_dir}/  params={n_params}  "
          f"bin acc={bin_acc:.4f}/F1={bin_f1:.4f}  "
          f"5cls acc={f5_acc:.4f}/F1={f5_f1:.4f}")
    return res


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='../../data/mitdb')
    p.add_argument('--output_dir', default='./results/hier')
    p.add_argument('--scheme', type=str, default='intra',
                   choices=['intra', 'inter'],
                   help='intra=beat 80/10/10 (leak); inter=de Chazal DS1/DS2')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr_schedule', type=str, default='cosine',
                   choices=['cosine', 'step'])
    p.add_argument('--lr_drop', type=int, default=50,
                   help='[step] epoch to drop LR ×0.1 (e.g. 1e-3→1e-4 at 50)')
    p.add_argument('--lambda5', type=float, default=2.0,
                   help='weight on the 5-class head loss (>1 prioritizes it)')
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--n_rr', type=int, default=8)
    p.add_argument('--rr_hidden', type=int, default=8,
                   help='RR-MLP hidden size (0 = raw concat, no MLP)')
    p.add_argument('--a_weight_mult', type=float, default=1.0,
                   help='extra multiplier on class-A loss weight')
    p.add_argument('--param_budget', type=int, default=800)
    p.add_argument('--aug_minority', type=str, default='')
    p.add_argument('--aug_p', type=float, default=0.5)
    p.add_argument('--oversample_A', type=int, default=4,
                   help='replicate class-A beats x this (jittered copies)')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
