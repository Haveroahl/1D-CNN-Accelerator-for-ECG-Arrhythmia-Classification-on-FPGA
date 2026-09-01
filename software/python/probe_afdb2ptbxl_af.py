"""
PROBE G — Cross-dataset zero-shot AF: train AFDB -> test PTB-XL.
================================================================================
Train: AFDB, 10s/2500@250Hz, AF(AFIB+AFL) vs non-AF(N+J), lead ECG1, oversample balanced.
Test : PTB-XL, zero-shot. AF = AFIB or AFLT in scp_codes; non-AF = otherwise.
       PTB-XL is 500Hz 12-lead -> decimate to 250Hz, take lead II, take first 2500 samples
       (10s window). Class imbalance ~7% AF -> report F1 / sensitivity / specificity / AUC,
       NOT accuracy alone.

CAVEAT (stated honestly in report): AFDB uses ECG1 (limb-style 2-lead Holter); PTB-XL uses
lead II from a 12-lead resting recording. Lead + recording-context mismatch is a real source
of distribution shift, inherent to cross-dataset AF transfer (cannot be removed: AFDB has no
12-lead). This is what the study measures, not a bug.
"""
import os, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb, pandas as pd, ast
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, classification_report

AFDB_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'afdb')
PTBXL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ptbxl')
OUT_DIR   = os.path.join(os.path.dirname(__file__), 'results', 'probe_afdb')
os.makedirs(OUT_DIR, exist_ok=True)

FS_TGT = 250
WIN    = 10 * FS_TGT       # 2500
SEED   = 42
AF_SET    = {'(AFIB', '(AFL'}
NONAF_SET = {'(N', '(J'}
PTBXL_AF  = {'AFIB', 'AFLT'}


# ---------- AFDB train data ----------
def label_at(idx, s, l):
    pos = np.searchsorted(s, idx, side='right') - 1
    if pos < 0: return -1
    lab = l[pos]
    if lab in AF_SET: return 1
    if lab in NONAF_SET: return 0
    return -1

def load_afdb():
    recs = [r.strip() for r in open(os.path.join(AFDB_DIR, 'RECORDS'))]
    X, y = [], []
    for rec in recs:
        if not os.path.exists(os.path.join(AFDB_DIR, rec + '.dat')):
            continue
        sig, _ = wfdb.rdsamp(os.path.join(AFDB_DIR, rec))
        ann = wfdb.rdann(os.path.join(AFDB_DIR, rec), 'atr')
        s = np.asarray(ann.sample)
        l = [a.strip('\x00').strip() for a in ann.aux_note]
        x = sig[:, 0].astype(np.float32)
        for st in range(0, len(x) - WIN + 1, WIN):
            mid = st + WIN // 2
            lm = label_at(mid, s, l)
            if lm < 0: continue
            if label_at(st, s, l) != lm or label_at(st+WIN-1, s, l) != lm: continue
            seg = x[st:st+WIN]; mu, sd = seg.mean(), seg.std()
            X.append((seg-mu)/(sd+1e-6)); y.append(lm)
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


# ---------- PTB-XL test data ----------
def load_ptbxl():
    df = pd.read_csv(os.path.join(PTBXL_DIR, 'ptbxl_database.csv'))
    df['codes'] = df.scp_codes.apply(ast.literal_eval)
    df['is_af'] = df.codes.apply(lambda d: any(c in PTBXL_AF for c in d))
    X, y = [], []
    for _, row in df.iterrows():
        rec = row.filename_hr   # 500 Hz
        try:
            sig, _ = wfdb.rdsamp(os.path.join(PTBXL_DIR, rec))
        except Exception:
            continue
        lead2 = sig[:, 1].astype(np.float32)        # lead II
        lead2 = lead2[::2][:WIN]                      # decimate 500->250, first 10s
        if len(lead2) < WIN:
            continue
        mu, sd = lead2.mean(), lead2.std()
        X.append((lead2-mu)/(sd+1e-6)); y.append(int(row.is_af))
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


class ECG_4488_2(nn.Module):
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
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading AFDB (train)...")
    Xa, ya = load_afdb()
    print(f"  AFDB: AF={int((ya==1).sum())} non-AF={int((ya==0).sum())}")
    print("Loading PTB-XL (test, zero-shot)...")
    Xp, yp = load_ptbxl()
    print(f"  PTB-XL: AF={int((yp==1).sum())} non-AF={int((yp==0).sum())}  ({100*yp.mean():.1f}% AF)")

    rng = np.random.default_rng(SEED)
    # train on ALL afdb, oversample balanced
    n_count = max(int((ya==0).sum()), int((ya==1).sum()))
    pX, pY = [], []
    for c in (0, 1):
        ci = np.where(ya == c)[0]
        pick = rng.choice(ci, size=n_count, replace=True)
        pX.append(Xa[pick]); pY.append(ya[pick])
    Xtr = torch.tensor(np.concatenate(pX)).to(dev)
    ytr = torch.tensor(np.concatenate(pY)).to(dev)
    Xte = torch.tensor(Xp).to(dev)

    model = ECG_4488_2().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    bs, epochs = 128, 40
    n = len(ytr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            b = perm[i:i+bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), ytr[b]); loss.backward(); opt.step()
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  ep {ep+1}")

    model.eval()
    with torch.no_grad():
        logits = model(Xte).cpu()
        prob_af = torch.softmax(logits, 1)[:, 1].numpy()
        pred = logits.argmax(1).numpy()

    acc = (pred == yp).mean()
    f1m = f1_score(yp, pred, average='macro', zero_division=0)
    f1_af = f1_score(yp, pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(yp, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp+fn) else 0.0          # recall AF
    spec = tn / (tn + fp) if (tn+fp) else 0.0
    try:
        auc = roc_auc_score(yp, prob_af)
    except Exception:
        auc = float('nan')

    L = []
    L.append("# PROBE G — Cross-dataset zero-shot AF: AFDB -> PTB-XL\n")
    L.append(f"- Train: AFDB 10s/2500@250Hz, AF=AFIB+AFL vs non-AF=N+J, lead ECG1, balanced\n")
    L.append(f"- Test : PTB-XL zero-shot, lead II, 500->250Hz decimate, first 10s; AF=AFIB|AFLT\n")
    L.append(f"- Model 4-4-8-8 FC 8->2 (622 params); imbalanced test -> report F1/Sens/Spec/AUC\n")
    L.append(f"- NOTE: Lead mismatch (AFDB ECG1 vs PTB-XL II) + Holter vs resting = inherent shift\n")
    L.append(f"\n## PTB-XL test set\n- AF={int((yp==1).sum())} / non-AF={int((yp==0).sum())} ({100*yp.mean():.1f}% AF)\n")
    L.append(f"\n## Zero-shot results\n")
    L.append(f"- Accuracy: {acc:.4f}  (misleading under {100*yp.mean():.1f}% imbalance)\n")
    L.append(f"- **Macro-F1: {f1m:.4f}**\n")
    L.append(f"- **AF F1: {f1_af:.4f}**\n")
    L.append(f"- **Sensitivity (AF recall): {sens:.4f}**\n")
    L.append(f"- **Specificity: {spec:.4f}**\n")
    L.append(f"- **ROC-AUC: {auc:.4f}**\n")
    L.append(f"\n### Confusion (rows=true [non-AF, AF], cols=pred)\n```\n{cm}\n```\n")
    L.append(f"\n### sklearn report\n```\n{classification_report(yp, pred, target_names=['non-AF','AF'], zero_division=0)}\n```\n")
    report = "".join(L)
    with open(os.path.join(OUT_DIR, 'REPORT_zeroshot_ptbxl.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print("\n" + report)
    print(f"Saved -> {os.path.join(OUT_DIR, 'REPORT_zeroshot_ptbxl.md')}")


if __name__ == "__main__":
    main()
