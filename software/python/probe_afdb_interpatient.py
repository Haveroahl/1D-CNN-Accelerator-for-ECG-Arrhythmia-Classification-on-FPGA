"""
PROBE H — AFDB AF/non-AF, INTER-PATIENT (record-level split), 5-fold.
================================================================================
The honest number. Unlike PROBE F (segment-level split = same patient in train+test),
here ENTIRE RECORDS are held out. 23 records w/ .dat -> 5 folds (~4-5 records each).
Each fold: train on the other records (oversample balanced), test on held-out records.
Metrics pooled across all 5 test folds. Report Acc/MacroF1/AF-F1/Sens/Spec/AUC.

Model: deployed 4-4-8-8, FC 8->2, 622 params. 10s/2500@250Hz, lead ECG1.
Compare to intra-patient PROBE F (0.9848 acc / 0.9843 macroF1).
"""
import os, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'afdb')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'results', 'probe_afdb')
os.makedirs(OUT_DIR, exist_ok=True)
FS = 250; WIN = 10 * FS; SEED = 42; NFOLD = 5
AF_SET = {'(AFIB', '(AFL'}; NON_SET = {'(N', '(J'}


def lab_at(i, s, l):
    p = np.searchsorted(s, i, side='right') - 1
    if p < 0: return -1
    x = l[p]
    return 1 if x in AF_SET else (0 if x in NON_SET else -1)


def load_by_record():
    recs = [r.strip() for r in open(os.path.join(DATA_DIR, 'RECORDS'))]
    data = {}   # rec -> (X, y)
    for rec in recs:
        if not os.path.exists(os.path.join(DATA_DIR, rec + '.dat')):
            continue
        sig, _ = wfdb.rdsamp(os.path.join(DATA_DIR, rec))
        ann = wfdb.rdann(os.path.join(DATA_DIR, rec), 'atr')
        s = np.asarray(ann.sample); l = [a.strip('\x00').strip() for a in ann.aux_note]
        x = sig[:, 0].astype(np.float32)
        X, y = [], []
        for st in range(0, len(x) - WIN + 1, WIN):
            m = lab_at(st + WIN // 2, s, l)
            if m < 0: continue
            if lab_at(st, s, l) != m or lab_at(st+WIN-1, s, l) != m: continue
            seg = x[st:st+WIN]; mu, sd = seg.mean(), seg.std()
            X.append((seg-mu)/(sd+1e-6)); y.append(m)
        if X:
            data[rec] = (np.asarray(X, np.float32), np.asarray(y, np.int64))
    return data


class ECG_4488_2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, 5, padding=2, bias=True); self.pool1 = nn.MaxPool1d(5)
        self.conv2 = nn.Conv1d(4, 4, 5, padding=2, bias=True); self.pool2 = nn.MaxPool1d(5)
        self.conv3 = nn.Conv1d(4, 8, 5, padding=2, bias=True); self.pool3 = nn.MaxPool1d(5)
        self.conv4 = nn.Conv1d(8, 8, 5, padding=2, bias=True); self.pool4 = nn.MaxPool1d(5)
        self.gap = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(8, num_classes, bias=True)
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x)); x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x)); x = self.pool4(F.relu(self.conv4(x)))
        return self.fc(self.gap(x).squeeze(-1))


def train_eval(train_recs, test_recs, data, dev, rng):
    Xtr = np.concatenate([data[r][0] for r in train_recs])
    ytr = np.concatenate([data[r][1] for r in train_recs])
    Xte = np.concatenate([data[r][0] for r in test_recs])
    yte = np.concatenate([data[r][1] for r in test_recs])
    # oversample train balanced
    n_count = max(int((ytr==0).sum()), int((ytr==1).sum()))
    pX, pY = [], []
    for c in (0, 1):
        ci = np.where(ytr == c)[0]
        if len(ci) == 0: continue
        pick = rng.choice(ci, size=n_count, replace=True)
        pX.append(Xtr[pick]); pY.append(ytr[pick])
    Xtr_b = torch.tensor(np.concatenate(pX)).to(dev)
    ytr_b = torch.tensor(np.concatenate(pY)).to(dev)
    Xte_t = torch.tensor(Xte).to(dev)

    model = ECG_4488_2().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
    bs, epochs = 128, 30; n = len(ytr_b)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            b = perm[i:i+bs]; opt.zero_grad()
            lossf(model(Xtr_b[b]), ytr_b[b]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(Xte_t).cpu()
        prob = torch.softmax(logits, 1)[:, 1].numpy()
        pred = logits.argmax(1).numpy()
    return yte, pred, prob


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(SEED)
    print("Loading AFDB by record...")
    data = load_by_record()
    recs = list(data.keys())
    # sort by AF fraction so folds are balanced in AF content
    af_frac = {r: data[r][1].mean() for r in recs}
    recs_sorted = sorted(recs, key=lambda r: af_frac[r])
    # round-robin assign to folds -> each fold gets a spread of AF fractions
    folds = [[] for _ in range(NFOLD)]
    for i, r in enumerate(recs_sorted):
        folds[i % NFOLD].append(r)
    print("Folds (record -> AF%):")
    for fi, f in enumerate(folds):
        print(f"  fold{fi}: " + ", ".join(f"{r}({100*af_frac[r]:.0f}%)" for r in f))

    all_y, all_pred, all_prob = [], [], []
    per_fold = []
    for fi in range(NFOLD):
        test_recs = folds[fi]
        train_recs = [r for r in recs if r not in test_recs]
        yte, pred, prob = train_eval(train_recs, test_recs, data, dev, rng)
        f1m = f1_score(yte, pred, average='macro', zero_division=0)
        try: auc = roc_auc_score(yte, prob) if len(set(yte)) > 1 else float('nan')
        except Exception: auc = float('nan')
        per_fold.append((fi, len(yte), f1m, auc))
        print(f"  fold{fi}: n={len(yte)} macroF1={f1m:.4f} AUC={auc:.4f}")
        all_y.append(yte); all_pred.append(pred); all_prob.append(prob)

    y = np.concatenate(all_y); pred = np.concatenate(all_pred); prob = np.concatenate(all_prob)
    acc = (pred == y).mean()
    f1m = f1_score(y, pred, average='macro', zero_division=0)
    f1_af = f1_score(y, pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    sens = tp/(tp+fn) if (tp+fn) else 0.0; spec = tn/(tn+fp) if (tn+fp) else 0.0
    auc = roc_auc_score(y, prob)

    L = []
    L.append("# PROBE H — AFDB AF/non-AF, INTER-PATIENT (record-level 5-fold)\n")
    L.append("- Entire records held out per fold (no patient in both train/test)\n")
    L.append("- Model 4-4-8-8 FC 8->2 (622 params), 10s/2500@250Hz, oversample TRAIN balanced\n")
    L.append(f"\n## Pooled across 5 folds (n={len(y)})\n")
    L.append(f"- Accuracy: {acc:.4f}\n- **Macro-F1: {f1m:.4f}**\n- AF F1: {f1_af:.4f}\n")
    L.append(f"- **Sensitivity (AF recall): {sens:.4f}**\n- **Specificity: {spec:.4f}**\n- **ROC-AUC: {auc:.4f}**\n")
    L.append(f"\n### Confusion (rows=true [non-AF, AF])\n```\n{cm}\n```\n")
    L.append("\n### Per-fold\n| fold | n | macroF1 | AUC |\n|---|---|---|---|\n")
    for fi, n, fm, a in per_fold:
        L.append(f"| {fi} | {n} | {fm:.4f} | {a:.4f} |\n")
    L.append("\n## vs intra-patient (PROBE F)\n")
    L.append("- Intra-patient (segment split): 0.9848 acc / 0.9843 macroF1\n")
    L.append(f"- **Inter-patient (this):       {acc:.4f} acc / {f1m:.4f} macroF1 / AUC {auc:.4f}**\n")
    report = "".join(L)
    with open(os.path.join(OUT_DIR, 'REPORT_interpatient.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print("\n" + report)
    print(f"Saved -> {os.path.join(OUT_DIR, 'REPORT_interpatient.md')}")


if __name__ == "__main__":
    main()
