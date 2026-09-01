"""
PROBE F — MIT-BIH AFDB, 2-class AF vs non-AF, 10s segments (2500 @ 250Hz).
================================================================================
- AFDB: 23 records w/ .dat (00735, 03665 have no .dat -> skipped), 250 Hz, 2 ECG leads.
- Rhythm annotations (aux_note): (AFIB, (AFL, (N, (J spanning regions.
    AF      = AFIB + AFL  (atrial fibrillation + flutter)
    non-AF  = N + J
- Segment: non-overlapping 10s windows = 2500 samples (matches Chapman input len!).
    Label = rhythm region the segment's CENTER falls in. Segments straddling a
    rhythm boundary into a different class are dropped (clean labels).
- Lead: ECG1 (channel 0), per-segment z-score.
- Model: deployed 4-4-8-8 topology, FC 8->2. Clean split (split-first, oversample TRAIN only).

Goal: feasibility of the 8-PE architecture's model on a clinically-standard AF task,
with segment length 2500 = identical to the Chapman deployment (no RTL change needed).
"""
import os, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from sklearn.metrics import f1_score, confusion_matrix, classification_report

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'afdb')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'results', 'probe_afdb')
os.makedirs(OUT_DIR, exist_ok=True)

FS   = 250
WIN  = 10 * FS          # 2500 samples = 10 s
SEED = 42
AF_SET    = {'(AFIB', '(AFL'}
NONAF_SET = {'(N', '(J'}


def label_at(sample_idx, ann_samples, ann_labels):
    """Return rhythm class (1=AF, 0=non-AF, -1=unknown) active at sample_idx."""
    # find last annotation at or before sample_idx
    pos = np.searchsorted(ann_samples, sample_idx, side='right') - 1
    if pos < 0:
        return -1
    lab = ann_labels[pos]
    if lab in AF_SET:
        return 1
    if lab in NONAF_SET:
        return 0
    return -1


def load_segments():
    records = [r.strip() for r in open(os.path.join(DATA_DIR, 'RECORDS'))]
    X, y = [], []
    dropped_boundary = 0
    for rec in records:
        if not os.path.exists(os.path.join(DATA_DIR, rec + '.dat')):
            continue
        sig, fields = wfdb.rdsamp(os.path.join(DATA_DIR, rec))
        ann = wfdb.rdann(os.path.join(DATA_DIR, rec), 'atr')
        ann_s = np.asarray(ann.sample)
        ann_l = [a.strip('\x00').strip() for a in ann.aux_note]
        x = sig[:, 0].astype(np.float32)   # ECG1
        n = len(x)
        for start in range(0, n - WIN + 1, WIN):
            mid = start + WIN // 2
            lab_mid   = label_at(mid,         ann_s, ann_l)
            lab_start = label_at(start,       ann_s, ann_l)
            lab_end   = label_at(start+WIN-1, ann_s, ann_l)
            if lab_mid < 0:
                continue
            # drop segment straddling a class boundary (mixed rhythm) for clean labels
            if lab_start != lab_mid or lab_end != lab_mid:
                dropped_boundary += 1
                continue
            seg = x[start:start+WIN]
            mu, sd = seg.mean(), seg.std()
            seg = (seg - mu) / (sd + 1e-6)
            X.append(seg); y.append(lab_mid)
    print(f"dropped (boundary/mixed): {dropped_boundary}")
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


class ECG_4488_2(nn.Module):
    """Deployed 4-4-8-8 topology, 2500 input, FC 8->2."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, 5, padding=2, bias=True); self.pool1 = nn.MaxPool1d(5)
        self.conv2 = nn.Conv1d(4, 4, 5, padding=2, bias=True); self.pool2 = nn.MaxPool1d(5)
        self.conv3 = nn.Conv1d(4, 8, 5, padding=2, bias=True); self.pool3 = nn.MaxPool1d(5)
        self.conv4 = nn.Conv1d(8, 8, 5, padding=2, bias=True); self.pool4 = nn.MaxPool1d(5)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(8, num_classes, bias=True)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))            # 2500 -> 500
        x = self.pool2(self.conv2(x))            # 500 -> 100
        x = self.pool3(self.conv3(x))            # 100 -> 20
        x = self.pool4(F.relu(self.conv4(x)))    # 20 -> 4
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Loading AFDB 10s segments...")
    X, y = load_segments()
    print(f"Total segments: {len(y)}  dist: AF={int((y==1).sum())} non-AF={int((y==0).sum())}")

    rng = np.random.default_rng(SEED)
    # split-first 80/20 stratified
    tr_idx, te_idx = [], []
    for c in (0, 1):
        ci = np.where(y == c)[0]; rng.shuffle(ci)
        cut = int(0.8 * len(ci)); tr_idx.append(ci[:cut]); te_idx.append(ci[cut:])
    tr = np.concatenate(tr_idx); te = np.concatenate(te_idx)
    Xtr, ytr = X[tr], y[tr]; Xte, yte = X[te], y[te]
    print(f"Train (real): AF={int((ytr==1).sum())} non-AF={int((ytr==0).sum())}")
    print(f"Test  (real): AF={int((yte==1).sum())} non-AF={int((yte==0).sum())}")

    # oversample TRAIN only to balance
    n_count = max(int((ytr==0).sum()), int((ytr==1).sum()))
    pX, pY = [], []
    for c in (0, 1):
        ci = np.where(ytr == c)[0]
        pick = rng.choice(ci, size=n_count, replace=True)
        pX.append(Xtr[pick]); pY.append(ytr[pick])
    Xtr_b = np.concatenate(pX); ytr_b = np.concatenate(pY)

    Xtr_t = torch.tensor(Xtr_b).to(dev); ytr_t = torch.tensor(ytr_b).to(dev)
    Xte_t = torch.tensor(Xte).to(dev)

    model = ECG_4488_2().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    bs, epochs = 128, 40
    n = len(ytr_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i+bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[b]), ytr_t[b])
            loss.backward(); opt.step()
            tot += loss.item()*len(b)
        if (ep+1) % 5 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xte_t).argmax(1).cpu().numpy() == yte).mean()
            print(f"  ep {ep+1:2d}  loss {tot/n:.4f}  test_acc {acc:.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).cpu().numpy()
    acc = (pred == yte).mean()
    f1m = f1_score(yte, pred, average='macro', zero_division=0)
    f1pc = f1_score(yte, pred, average=None, labels=[0,1], zero_division=0)
    cm = confusion_matrix(yte, pred, labels=[0,1])
    nparams = sum(p.numel() for p in model.parameters())

    L = []
    L.append("# PROBE F — AFDB AF vs non-AF, 10s segments (2500@250Hz), 4-4-8-8 FC 8->2\n")
    L.append(f"- Segment 2500 samples (= Chapman input len), lead ECG1, clean split, oversample TRAIN to {n_count}/class\n")
    L.append(f"- AF = AFIB+AFL ; non-AF = N+J ; boundary-straddling segments dropped\n")
    L.append(f"- params {nparams}, {epochs} ep\n")
    L.append(f"\n## Results\n- **Accuracy {acc:.4f} / Macro-F1 {f1m:.4f}**\n")
    L.append(f"- Test: AF={int((yte==1).sum())} non-AF={int((yte==0).sum())}\n")
    L.append(f"- Per-class F1: non-AF={f1pc[0]:.4f}, AF={f1pc[1]:.4f}\n")
    L.append(f"\n### Confusion (rows=true [non-AF, AF], cols=pred)\n```\n{cm}\n```\n")
    L.append(f"\n### sklearn report\n```\n{classification_report(yte, pred, target_names=['non-AF','AF'], zero_division=0)}\n```\n")
    report = "".join(L)
    with open(os.path.join(OUT_DIR, 'REPORT.md'), 'w') as f:
        f.write(report)
    print("\n" + report)
    print(f"Saved -> {os.path.join(OUT_DIR, 'REPORT.md')}")


if __name__ == "__main__":
    main()
