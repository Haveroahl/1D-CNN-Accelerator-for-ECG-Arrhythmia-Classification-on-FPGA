"""
PROBE E — MIT-BIH 5-class, window 250, pool 5/5/5, 3 CONV layers (L=10 SIMD-friendly).
================================================================================
Why this config: SIMD lane count L must divide every pre-pool out_len AND be a
multiple of the (uniform) pool stride. Window 250 + pool 5/5/5 gives pre-pool
lengths 250/50/10 -> gcd=10, uniform stride 5 -> L=10 is clean (100% utilization,
combinational pool). The price: only 3 conv layers (vs 4 in the deployed model).

Runs TWO channel configs to separate "3-conv vs 4-conv" from "narrow vs wide":
  - narrow (4,8,16)
  - wide   (8,16,16)
Clean split (split first, oversample TRAIN only). FC single layer. Compare A-F1
to the 4-conv probes: 4-4-8-8=0.802, 8-8-16-16=0.874.
"""
import os, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from sklearn.metrics import f1_score, confusion_matrix, classification_report

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'mitdb')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'results', 'probe_mitbih')
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = ['N', 'L', 'R', 'A', 'V']
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
HALF = 125            # window = 250 samples (+/-125 around R-peak)
WIN  = 2 * HALF
SEED = 42
PACED = {'102', '104', '107', '217'}


def load_beats():
    records = [r.strip() for r in open(os.path.join(DATA_DIR, 'RECORDS'))]
    X, y = [], []
    for rec in records:
        if rec in PACED:
            continue
        try:
            sig, fields = wfdb.rdsamp(os.path.join(DATA_DIR, rec))
            ann = wfdb.rdann(os.path.join(DATA_DIR, rec), 'atr')
        except Exception as e:
            print(f"  skip {rec}: {e}")
            continue
        sig_names = fields['sig_name']
        ch = sig_names.index('MLII') if 'MLII' in sig_names else 0
        x = sig[:, ch].astype(np.float32)
        for samp, sym in zip(ann.sample, ann.symbol):
            if sym not in CLS2IDX:
                continue
            a, b = samp - HALF, samp + HALF
            if a < 0 or b > len(x):
                continue
            beat = x[a:b]
            mu, sd = beat.mean(), beat.std()
            beat = (beat - mu) / (sd + 1e-6)
            X.append(beat); y.append(CLS2IDX[sym])
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


class ECG_3Conv(nn.Module):
    """3 conv layers, pool 5/5/5 (250 -> 50 -> 10 -> 2), GAP, FC single layer."""
    def __init__(self, ch, num_classes=5):
        super().__init__()
        c1, c2, c3 = ch
        self.conv1 = nn.Conv1d(1,  c1, 5, padding=2, bias=True); self.pool1 = nn.MaxPool1d(5)
        self.conv2 = nn.Conv1d(c1, c2, 5, padding=2, bias=True); self.pool2 = nn.MaxPool1d(5)
        self.conv3 = nn.Conv1d(c2, c3, 5, padding=2, bias=True); self.pool3 = nn.MaxPool1d(5)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(c3, num_classes, bias=True)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))            # 250 -> 50
        x = self.pool2(self.conv2(x))            # 50 -> 10
        x = self.pool3(F.relu(self.conv3(x)))    # 10 -> 2  (ReLU on last conv)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


def run(ch, Xtr_t, ytr_t, Xte_t, yte_t, yte, dev, epochs=30):
    model = ECG_3Conv(ch).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    bs = 256; n = len(ytr_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            b = perm[i:i+bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[b]), ytr_t[b])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).cpu().numpy()
    acc = (pred == yte).mean()
    f1m = f1_score(yte, pred, average='macro', labels=list(range(5)), zero_division=0)
    f1pc = f1_score(yte, pred, average=None, labels=list(range(5)), zero_division=0)
    cm = confusion_matrix(yte, pred, labels=list(range(5)))
    nparams = sum(p.numel() for p in model.parameters())
    return acc, f1m, f1pc, cm, nparams, pred


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading MIT-BIH beats (window 250)...")
    X, y = load_beats()
    print(f"Total beats: {len(y)}  dist: {dict(collections.Counter([CLASSES[i] for i in y]))}")

    rng = np.random.default_rng(SEED)
    # split first 80/20 stratified
    tr_idx, te_idx = [], []
    for c in range(5):
        ci = np.where(y == c)[0]; rng.shuffle(ci)
        cut = int(0.8 * len(ci)); tr_idx.append(ci[:cut]); te_idx.append(ci[cut:])
    tr = np.concatenate(tr_idx); te = np.concatenate(te_idx)
    Xtr, ytr = X[tr], y[tr]; Xte, yte = X[te], y[te]
    # oversample TRAIN only
    n_count = max(collections.Counter(ytr).values())
    pX, pY = [], []
    for c in range(5):
        ci = np.where(ytr == c)[0]
        pick = rng.choice(ci, size=n_count, replace=True)
        pX.append(Xtr[pick]); pY.append(ytr[pick])
    Xtr_b = np.concatenate(pX); ytr_b = np.concatenate(pY)

    Xtr_t = torch.tensor(Xtr_b).to(dev); ytr_t = torch.tensor(ytr_b).to(dev)
    Xte_t = torch.tensor(Xte).to(dev)

    configs = [("narrow (4,8,16)", (4, 8, 16)), ("wide (8,16,16)", (8, 16, 16))]
    L = []
    L.append("# PROBE E — MIT-BIH 5-class, window 250, pool 5/5/5, 3 CONV (L=10 SIMD-friendly)\n")
    L.append(f"- Window {WIN}, split-first 80/20, oversample TRAIN to {n_count}/class, FC single layer, 30 ep\n")
    L.append(f"- Test dist: {dict(collections.Counter([CLASSES[i] for i in yte]))}\n")
    for name, ch in configs:
        acc, f1m, f1pc, cm, nparams, _ = run(ch, Xtr_t, ytr_t, Xte_t, None, yte, dev)
        print(f"\n=== {name}  acc={acc:.4f}  macroF1={f1m:.4f}  A-F1={f1pc[3]:.3f}  params={nparams}")
        L.append(f"\n## Config {name} — params {nparams}\n")
        L.append(f"- **Acc {acc:.4f} / Macro-F1 {f1m:.4f}**\n")
        L.append(f"- Per-class F1: " + ", ".join(f"{c}={f:.3f}" for c, f in zip(CLASSES, f1pc)) + "\n")
        L.append(f"- Confusion (rows=true {CLASSES}):\n```\n{cm}\n```\n")

    L.append("\n## Comparison vs 4-conv (256-window) probes — A-F1 is the key metric\n")
    L.append("| config | conv | window | pool | acc | macroF1 | A-F1 | SIMD L |\n")
    L.append("|---|---|---|---|---|---|---|---|\n")
    L.append("| 4-4-8-8 | 4 | 256 | 5/5/2/2 | 0.9742 | 0.9354 | 0.802 | none (mixed) |\n")
    L.append("| 8-8-16-16 | 4 | 256 | 5/5/2/2 | 0.9866 | 0.9619 | 0.874 | none (mixed) |\n")
    for name, ch in configs:
        acc, f1m, f1pc, cm, nparams, _ = run(ch, Xtr_t, ytr_t, Xte_t, None, yte, dev)
        L.append(f"| {ch} | 3 | 250 | 5/5/5 | {acc:.4f} | {f1m:.4f} | {f1pc[3]:.3f} | **10** |\n")
    report = "".join(L)
    with open(os.path.join(OUT_DIR, 'REPORT_3conv250.md'), 'w') as f:
        f.write(report)
    print("\n" + report)
    print(f"Saved -> {os.path.join(OUT_DIR, 'REPORT_3conv250.md')}")


if __name__ == "__main__":
    main()
