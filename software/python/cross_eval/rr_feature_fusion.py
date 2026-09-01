"""
RR feature-fusion experiment — does concatenating 8 HRV features into the FC
input improve classification?  (float32 prototype)
=============================================================================
Different from rr_fusion_probe.py (which OVERRIDES argmax post-hoc). Here the
8 RR features are an EARLY/feature-level input: concat with the 8 GAP features
-> FC(16->4), trained jointly.

Compared head-to-head against a CNN-only baseline that retrains the SAME FC
(8->4) on the SAME data, so the only difference is the +8 RR inputs (fair).

Two protocols, run on Chapman (in-dist) and PTB-XL (cross-dataset):
  - linear-probe : freeze conv (the trained extractor), train only FC
  - full-ft      : train everything

8 HRV features (from Pan-Tompkins R-peaks on the z-scored 250Hz signal):
  mean_RR, SDNN, RMSSD, pNN50, mean_HR_bpm, min_RR, max_RR, RR_range
RR features are z-normalized per-feature (stats from the train split).

Read-only on data: loads the QAT float checkpoint + Chapman CSVs + PTB-XL npz.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pan_tompkins import pan_tompkins_hr
from ptbxl_eval import load_qat_checkpoint, CLASS_NAMES

FS = 250
N_RR_FEATS = 8


# ── 8 HRV features from one z-scored ECG window ────────────────────────────

def rr_features(sig):
    """Return 8 HRV features. NaN-safe (returns zeros if <2 peaks)."""
    _, peaks = pan_tompkins_hr(sig, fs=FS)
    if len(peaks) < 2:
        return np.zeros(N_RR_FEATS, dtype=np.float32)
    rr = np.diff(peaks) / FS                      # RR intervals in seconds
    mean_rr = rr.mean()
    sdnn    = rr.std()
    rmssd   = np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else 0.0
    pnn50   = np.mean(np.abs(np.diff(rr)) > 0.05) if len(rr) > 1 else 0.0
    mean_hr = 60.0 / mean_rr if mean_rr > 0 else 0.0
    return np.array([mean_rr, sdnn, rmssd, pnn50, mean_hr,
                     rr.min(), rr.max(), rr.max() - rr.min()],
                    dtype=np.float32)


def extract_rr(X):
    """X: (N, 2500) -> (N, 8)."""
    return np.stack([rr_features(X[i]) for i in range(len(X))]).astype(np.float32)


# ── Fusion head: conv extractor (from ckpt) + FC over [GAP || RR] ──────────

class FusionModel(nn.Module):
    """Wraps the trained conv extractor; FC takes GAP(8) [+ RR(8) if use_rr]."""
    def __init__(self, base, use_rr, n_rr=N_RR_FEATS, num_classes=4):
        super().__init__()
        self.conv1, self.conv2 = base.conv1, base.conv2
        self.conv3, self.conv4 = base.conv3, base.conv4
        self.pool = nn.MaxPool1d(5)
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.use_rr = use_rr
        c4 = base.conv4.out_channels
        self.fc = nn.Linear(c4 + (n_rr if use_rr else 0), num_classes, bias=True)

    def features(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        return self.gap(x).squeeze(-1)            # (B, c4)

    def forward(self, x, rr=None):
        g = self.features(x)
        if self.use_rr:
            g = torch.cat([g, rr], dim=1)
        return self.fc(g)

    def freeze_conv(self):
        for n, p in self.named_parameters():
            if not n.startswith('fc'):
                p.requires_grad = False


# ── train / eval over tensors (X, rr, y) ───────────────────────────────────

def run_epoch(model, X, RR, y, opt, crit, device, train, bs=128):
    idx = np.arange(len(y))
    if train:
        model.train(); np.random.shuffle(idx)
    else:
        model.eval()
    preds = []
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        xb = torch.from_numpy(X[b]).to(device)
        rb = torch.from_numpy(RR[b]).to(device) if model.use_rr else None
        yb = torch.from_numpy(y[b]).long().to(device)
        if train:
            opt.zero_grad()
            out = model(xb, rb)
            loss = crit(out, yb); loss.backward(); opt.step()
        else:
            with torch.no_grad():
                out = model(xb, rb)
        preds.append(out.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def fit_eval(base, use_rr, freeze, data, device, epochs, lr):
    """data = dict of (X, RR, y) for train/val/test. Returns test metrics."""
    model = FusionModel(base, use_rr).to(device)
    if freeze:
        model.freeze_conv()
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    Xtr, RRtr, ytr = data['train']
    Xv,  RRv,  yv  = data['val']
    best_acc, best_sd = -1.0, None
    for ep in range(epochs):
        run_epoch(model, Xtr, RRtr, ytr, opt, crit, device, train=True)
        pv = run_epoch(model, Xv, RRv, yv, opt, crit, device, train=False)
        acc = accuracy_score(yv, pv)
        if acc > best_acc:
            best_acc = acc
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
    if best_sd:
        model.load_state_dict(best_sd)

    Xte, RRte, yte = data['test']
    pt = run_epoch(model, Xte, RRte, yte, None, crit, device, train=False)
    return {
        'acc': float(accuracy_score(yte, pt)),
        'f1_macro': float(f1_score(yte, pt, average='macro', zero_division=0)),
        'f1_per_class': f1_score(yte, pt, average=None, labels=[0,1,2,3],
                                 zero_division=0).tolist(),
        'confusion_matrix': confusion_matrix(yte, pt, labels=[0,1,2,3]).tolist(),
        'n': int(len(yte)),
    }


# ── data loading ───────────────────────────────────────────────────────────

def load_chapman(data_dir):
    from utils.dataset import ChapmanECGDataset
    out = {}
    for split in ('train', 'val', 'test'):
        ds = ChapmanECGDataset(data_dir, split=split)
        X = np.stack(ds.records).astype(np.float32)
        y = np.array(ds.labels, dtype=np.int64)
        out[split] = (X, y)
    return out


def load_ptbxl(npz_path):
    d = np.load(npz_path)
    return {
        'train': (d['X_train'].astype(np.float32), d['y_train'].astype(np.int64)),
        'val':   (d['X_val'].astype(np.float32),   d['y_val'].astype(np.int64)),
        'test':  (d['X_test'].astype(np.float32),  d['y_test'].astype(np.int64)),
    }


def build_data(raw, cache_tag, cache_dir):
    """raw = {split:(X,y)}. Adds z-normalized RR features (cached)."""
    cache = os.path.join(cache_dir, f'rr_{cache_tag}.npz')
    if os.path.exists(cache):
        print(f"  [cache] {cache}")
        c = np.load(cache)
        rr = {s: c[f'rr_{s}'] for s in raw}
    else:
        rr = {}
        for s in raw:
            print(f"  extracting RR for {cache_tag}/{s} (n={len(raw[s][1])}) ...")
            rr[s] = extract_rr(raw[s][0])
        np.savez(cache, **{f'rr_{s}': rr[s] for s in raw})
        print(f"  [saved] {cache}")
    # z-normalize RR per-feature using TRAIN stats
    mu = rr['train'].mean(0, keepdims=True)
    sd = rr['train'].std(0, keepdims=True) + 1e-6
    data = {}
    for s in raw:
        X, y = raw[s]
        data[s] = (X, ((rr[s] - mu) / sd).astype(np.float32), y)
    return data


# ── main ─────────────────────────────────────────────────────────────────

def fmt(m):
    fp = '/'.join(f'{v:.2f}' for v in m['f1_per_class'])
    return f"acc={m['acc']:.4f}  f1={m['f1_macro']:.4f}  per-class[{fp}]"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',    default=os.path.join(here, '..', 'results', 'qat_int8', 'model_qat_int8.pth'))
    p.add_argument('--ptbxl',   default=os.path.join(here, '..', '..', '..', 'data', 'ptbxl_processed', 'ptbxl_dataset.npz'))
    p.add_argument('--chapman', default=os.path.join(here, '..', '..', '..', 'data', 'Chapman'))
    p.add_argument('--output',  default=os.path.join(here, '..', 'results', 'cross_eval'))
    p.add_argument('--epochs',  type=int, default=30)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] device={device}  epochs={args.epochs}")

    base = load_qat_checkpoint(args.ckpt, device)

    print("\n[DATA] Chapman ...")
    chap = build_data(load_chapman(args.chapman), 'chapman', args.output)
    print("\n[DATA] PTB-XL ...")
    ptb = build_data(load_ptbxl(args.ptbxl), 'ptbxl', args.output)

    datasets = {'Chapman(in-dist)': chap, 'PTB-XL(cross)': ptb}
    protocols = [('linear-probe', True, 1e-3), ('full-ft', False, 5e-4)]

    results = {}
    print("\n" + "=" * 78)
    for dname, data in datasets.items():
        results[dname] = {}
        for pname, freeze, lr in protocols:
            base_eval = load_qat_checkpoint(args.ckpt, device)  # fresh conv each run
            m_cnn = fit_eval(base_eval, False, freeze, data, device, args.epochs, lr)
            base_eval = load_qat_checkpoint(args.ckpt, device)
            m_rr  = fit_eval(base_eval, True,  freeze, data, device, args.epochs, lr)
            d_acc = (m_rr['acc'] - m_cnn['acc']) * 100
            d_f1  = (m_rr['f1_macro'] - m_cnn['f1_macro']) * 100
            results[dname][pname] = {'cnn_only': m_cnn, 'cnn_plus_rr': m_rr,
                                     'delta_acc_pp': d_acc, 'delta_f1_pp': d_f1}
            print(f"\n{dname}  |  {pname}  (lr={lr})")
            print(f"  CNN-only   : {fmt(m_cnn)}")
            print(f"  CNN+8RR    : {fmt(m_rr)}")
            print(f"  -> delta   : acc {d_acc:+.2f} pp   f1 {d_f1:+.2f} pp")
    print("\n" + "=" * 78)

    print("\nSUMMARY (delta = CNN+RR minus CNN-only)")
    print(f"{'dataset':<18}{'protocol':<14}{'d_acc(pp)':>10}{'d_f1(pp)':>10}")
    print("-" * 52)
    for dname in results:
        for pname in results[dname]:
            r = results[dname][pname]
            print(f"{dname:<18}{pname:<14}{r['delta_acc_pp']:>+10.2f}{r['delta_f1_pp']:>+10.2f}")

    out = os.path.join(args.output, 'rr_feature_fusion.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] saved {out}")


if __name__ == '__main__':
    main()
