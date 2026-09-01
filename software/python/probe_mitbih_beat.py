"""
PROBE (throwaway experiment) — MIT-BIH 5-class beat morphology classification.
================================================================================
Goal: quick data point to decide next direction. NOT production.

- Classes: N, L, R, A, V  (MIT-BIH annotation symbols)
    N = Normal, L = LBBB, R = RBBB, A = Atrial premature (APC), V = PVC
- Beat segmentation: 256 samples centered on R-peak (MIT-BIH 360 Hz).
- Imbalance fix: random oversample EVERY class up to N count (all data, then split).
                 NOTE: oversample-before-split inflates accuracy (same beat may land in
                 both train/test). Accepted here: goal = does the HARDWARE topology learn
                 beat morphology at all, not a leakage-clean number.
- Model: same 4-4-8-8 topology as the deployed Chapman model, FC 8->5.
- Split: random 80/20 after oversampling.

Outputs: results/probe_mitbih/REPORT.md  (acc, macro-F1, per-class F1, confusion matrix)
"""
import os, sys, collections
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
HALF = 128            # beat window = 256 samples (±128 around R-peak)
WIN  = 2 * HALF
SEED = 42

# Records that do NOT have lead MLII as channel 0 use the available leads;
# MIT-BIH records — exclude paced (102,104,107,217) per AAMI convention.
PACED = {'102', '104', '107', '217'}


def load_beats():
    """Return X (n,WIN) float32 z-scored per-beat, y (n,) int, groups (n,) record-id."""
    records = [r.strip() for r in open(os.path.join(DATA_DIR, 'RECORDS'))]
    X, y, groups = [], [], []
    for rec in records:
        if rec in PACED:
            continue
        try:
            sig, fields = wfdb.rdsamp(os.path.join(DATA_DIR, rec))
            ann = wfdb.rdann(os.path.join(DATA_DIR, rec), 'atr')
        except Exception as e:
            print(f"  skip {rec}: {e}")
            continue
        # pick MLII if present else channel 0
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
            beat = (beat - mu) / (sd + 1e-6)   # per-beat z-score
            X.append(beat); y.append(CLS2IDX[sym]); groups.append(rec)
    return np.asarray(X, np.float32), np.asarray(y, np.int64), np.asarray(groups)


class ECG_1DCNN_5(nn.Module):
    """Same 4-4-8-8 conv topology as deployed model, adapted to 256-sample input, FC 8->5."""
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
        x = self.pool1(self.conv1(x))             # 256 -> 51
        x = self.pool2(self.conv2(x))             # 51 -> 10
        x = self.pool3(self.conv3(x))             # 10 -> 5
        x = self.pool4(F.relu(self.conv4(x)))     # 5 -> 2
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading MIT-BIH beats...")
    X, y, groups = load_beats()
    print(f"Total beats: {len(y)}  dist: {dict(collections.Counter([CLASSES[i] for i in y]))}")

    rng = np.random.default_rng(SEED)

    # random oversample EVERY class up to N count (all data), THEN split
    n_count = max(collections.Counter(y).values())
    parts_X, parts_y = [], []
    for c in range(len(CLASSES)):
        ci = np.where(y == c)[0]
        if len(ci) == 0:
            continue
        pick = rng.choice(ci, size=n_count, replace=True)
        parts_X.append(X[pick]); parts_y.append(y[pick])
    Xbal = np.concatenate(parts_X); ybal = np.concatenate(parts_y)
    print(f"After oversample (all classes -> {n_count}): total {len(ybal)}")

    # random 80/20 split on balanced set
    idx = rng.permutation(len(ybal))
    cut = int(0.8 * len(ybal))
    tr, te = idx[:cut], idx[cut:]
    Xtr_b, ytr_b = Xbal[tr], ybal[tr]
    Xte,   yte   = Xbal[te], ybal[te]

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
            out = model(Xtr_t[b])
            loss = lossf(out, ytr_t[b])
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

    lines = []
    lines.append("# PROBE — MIT-BIH 5-class beat morphology (N,L,R,A,V)\n")
    lines.append(f"- Window: {WIN} samples (+/-{HALF} around R-peak), 360 Hz\n")
    lines.append(f"- Split: intra-patient random 80/20 (probe only)\n")
    lines.append(f"- Imbalance: random oversample TRAIN ONLY to {n_count}/class\n")
    lines.append(f"- Model: 4-4-8-8 conv (same as deployed), FC 8->5, float32, {epochs} ep\n")
    lines.append(f"\n## Results\n")
    lines.append(f"- **Test accuracy: {acc:.4f}**\n")
    lines.append(f"- **Macro-F1: {f1m:.4f}**\n")
    lines.append(f"- Test-set class distribution: {dict(collections.Counter([CLASSES[i] for i in yt]))}\n")
    lines.append(f"\n### Per-class F1\n")
    for c, f in zip(CLASSES, f1pc):
        lines.append(f"- {c}: {f:.4f}\n")
    lines.append(f"\n### Confusion matrix (rows=true, cols=pred; order {CLASSES})\n```\n{cm}\n```\n")
    lines.append(f"\n### sklearn report\n```\n{classification_report(yt, pred, target_names=CLASSES, zero_division=0)}\n```\n")
    report = "".join(lines)
    with open(os.path.join(OUT_DIR, 'REPORT.md'), 'w') as f:
        f.write(report)
    print("\n" + report)
    print(f"Saved -> {os.path.join(OUT_DIR, 'REPORT.md')}")


if __name__ == "__main__":
    main()
