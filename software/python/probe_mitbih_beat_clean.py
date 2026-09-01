"""
PROBE B (clean) — MIT-BIH 5-class beat morphology, NO leakage.
================================================================================
Difference from probe_mitbih_beat.py:
  - SPLIT FIRST (80/20), THEN random oversample TRAIN ONLY.
  - Test set keeps the REAL class distribution (imbalanced) -> honest number.
Compare test_acc / macro-F1 against the leaky probe (97.70% / 0.9771) to see
how much the leaky version was inflated.

Classes N,L,R,A,V; 256-sample beat window; same 4-4-8-8 hardware topology, FC 8->5.
Split here is intra-patient random (probe). Inter-patient DS1/DS2 left for production.
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
HALF = 128
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


class ECG_1DCNN_5(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, 5, padding=2, bias=True); self.pool1 = nn.MaxPool1d(5)
        self.conv2 = nn.Conv1d(4, 4, 5, padding=2, bias=True); self.pool2 = nn.MaxPool1d(5)
        self.conv3 = nn.Conv1d(4, 8, 5, padding=2, bias=True); self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(8, 8, 5, padding=2, bias=True); self.pool4 = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(8, num_classes, bias=True)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading MIT-BIH beats...")
    X, y = load_beats()
    print(f"Total beats: {len(y)}  dist: {dict(collections.Counter([CLASSES[i] for i in y]))}")

    rng = np.random.default_rng(SEED)

    # SPLIT FIRST 80/20 (stratified per class so rare classes appear in test)
    tr_idx, te_idx = [], []
    for c in range(len(CLASSES)):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        cut = int(0.8 * len(ci))
        tr_idx.append(ci[:cut]); te_idx.append(ci[cut:])
    tr = np.concatenate(tr_idx); te = np.concatenate(te_idx)
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]
    print(f"Train (real): {dict(collections.Counter([CLASSES[i] for i in ytr]))}")
    print(f"Test  (real, kept imbalanced): {dict(collections.Counter([CLASSES[i] for i in yte]))}")

    # oversample TRAIN ONLY up to majority count
    n_count = max(collections.Counter(ytr).values())
    parts_X, parts_y = [], []
    for c in range(len(CLASSES)):
        ci = np.where(ytr == c)[0]
        pick = rng.choice(ci, size=n_count, replace=True)
        parts_X.append(Xtr[pick]); parts_y.append(ytr[pick])
    Xtr_b = np.concatenate(parts_X); ytr_b = np.concatenate(parts_y)
    print(f"Train after oversample: total {len(ytr_b)} (per class = {n_count})")

    Xtr_t = torch.tensor(Xtr_b).to(dev); ytr_t = torch.tensor(ytr_b).to(dev)
    Xte_t = torch.tensor(Xte).to(dev);   yte_t = torch.tensor(yte).to(dev)

    model = ECG_1DCNN_5().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    bs, epochs = 256, 30
    n = len(ytr_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i+bs]
            opt.zero_grad()
            out = model(Xtr_t[b]); loss = lossf(out, ytr_t[b])
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if (ep + 1) % 5 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                pred = model(Xte_t).argmax(1)
                acc = (pred == yte_t).float().mean().item()
            print(f"  ep {ep+1:2d}  loss {tot/n:.4f}  test_acc {acc:.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).cpu().numpy()
    yt = yte
    acc = (pred == yt).mean()
    f1m = f1_score(yt, pred, average='macro', labels=list(range(len(CLASSES))), zero_division=0)
    f1pc = f1_score(yt, pred, average=None, labels=list(range(len(CLASSES))), zero_division=0)
    cm = confusion_matrix(yt, pred, labels=list(range(len(CLASSES))))

    L = []
    L.append("# PROBE B (CLEAN, no leakage) — MIT-BIH 5-class beat morphology\n")
    L.append(f"- Window {WIN} samples, split FIRST 80/20 then oversample TRAIN ONLY to {n_count}/class\n")
    L.append(f"- Test keeps REAL imbalanced distribution\n")
    L.append(f"- Model 4-4-8-8 (same as deployed), FC 8->5, float32, {epochs} ep\n")
    L.append(f"\n## Results\n- **Test accuracy: {acc:.4f}**\n- **Macro-F1: {f1m:.4f}**\n")
    L.append(f"- Test dist: {dict(collections.Counter([CLASSES[i] for i in yt]))}\n")
    L.append(f"\n### Per-class F1\n")
    for c, f in zip(CLASSES, f1pc):
        L.append(f"- {c}: {f:.4f}\n")
    L.append(f"\n### Confusion matrix (rows=true, cols=pred; order {CLASSES})\n```\n{cm}\n```\n")
    L.append(f"\n### sklearn report\n```\n{classification_report(yt, pred, target_names=CLASSES, zero_division=0)}\n```\n")
    L.append(f"\n## vs leaky probe\n- Leaky (oversample-before-split): 0.9770 acc / 0.9771 macro-F1\n- This (clean): {acc:.4f} acc / {f1m:.4f} macro-F1\n")
    report = "".join(L)
    with open(os.path.join(OUT_DIR, 'REPORT_clean.md'), 'w') as f:
        f.write(report)
    print("\n" + report)
    print(f"Saved -> {os.path.join(OUT_DIR, 'REPORT_clean.md')}")


if __name__ == "__main__":
    main()
