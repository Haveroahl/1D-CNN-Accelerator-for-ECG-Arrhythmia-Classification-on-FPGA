"""
PROBE J — Train Chapman AF/non-AF, zero-shot cross-check -> PTB-XL AND AFDB.
================================================================================
Source: Chapman (train split, ~8500 patients), AF = orig label 0 (AFIB+AF) vs non-AF.
Model: deployed 4-4-8-8, FC 8->2, 622 params. Lead II, 2500@250Hz. Oversample TRAIN.
Cross-check (zero-shot, model never sees these):
  - PTB-XL: lead II, 500->250 decimate, first 10s; AF = AFIB|AFLT
  - AFDB:   ECG1, 10s segments@250Hz; AF = AFIB+AFL ; non-AF = N+J
Also report Chapman in-dist test for reference.
Imbalanced -> F1 / sensitivity / specificity / AUC (not accuracy alone).

CAVEAT: AFDB uses ECG1 (Holter), PTB-XL & Chapman use lead II (resting). Lead/context
mismatch to AFDB is an inherent cross-dataset shift, stated honestly.
"""
import os, ast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.dataset import ChapmanECGDataset

BASE = os.path.dirname(__file__)
CHAP_DIR  = os.path.join(BASE, '..', '..', 'data', 'Chapman')
PTBXL_DIR = os.path.join(BASE, '..', '..', 'data', 'ptbxl')
AFDB_DIR  = os.path.join(BASE, '..', '..', 'data', 'afdb')
OUT_DIR   = os.path.join(BASE, 'results', 'probe_chapman_af')
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42; WIN = 2500
PTBXL_AF = {'AFIB', 'AFLT'}
AF_SET = {'(AFIB', '(AFL'}; NON_SET = {'(N', '(J'}


def chap_xy(ds):
    X = np.stack(ds.records).astype(np.float32)
    y = (np.asarray(ds.labels, np.int64) == 0).astype(np.int64)   # 1=AF
    return X, y


def load_ptbxl():
    df = pd.read_csv(os.path.join(PTBXL_DIR, 'ptbxl_database.csv'))
    df['codes'] = df.scp_codes.apply(ast.literal_eval)
    df['is_af'] = df.codes.apply(lambda d: any(c in PTBXL_AF for c in d))
    X, y = [], []
    for _, r in df.iterrows():
        try: sig, _ = wfdb.rdsamp(os.path.join(PTBXL_DIR, r.filename_hr))
        except Exception: continue
        s = sig[:, 1].astype(np.float32)[::2][:WIN]   # lead II, 500->250
        if len(s) < WIN: continue
        mu, sd = s.mean(), s.std(); X.append((s-mu)/(sd+1e-6)); y.append(int(r.is_af))
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


def _lab(i, s, l):
    p = np.searchsorted(s, i, side='right') - 1
    if p < 0: return -1
    x = l[p]; return 1 if x in AF_SET else (0 if x in NON_SET else -1)

def load_afdb():
    recs = [r.strip() for r in open(os.path.join(AFDB_DIR, 'RECORDS'))]
    X, y = [], []
    for rec in recs:
        if not os.path.exists(os.path.join(AFDB_DIR, rec + '.dat')): continue
        sig, _ = wfdb.rdsamp(os.path.join(AFDB_DIR, rec))
        ann = wfdb.rdann(os.path.join(AFDB_DIR, rec), 'atr')
        s = np.asarray(ann.sample); l = [a.strip('\x00').strip() for a in ann.aux_note]
        x = sig[:, 0].astype(np.float32)
        for st in range(0, len(x) - WIN + 1, WIN):
            m = _lab(st + WIN//2, s, l)
            if m < 0 or _lab(st, s, l) != m or _lab(st+WIN-1, s, l) != m: continue
            seg = x[st:st+WIN]; mu, sd = seg.mean(), seg.std()
            X.append((seg-mu)/(sd+1e-6)); y.append(m)
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


class ECG_4488_2(nn.Module):
    def __init__(s, nc=2):
        super().__init__()
        s.conv1=nn.Conv1d(1,4,5,padding=2,bias=True); s.pool1=nn.MaxPool1d(5)
        s.conv2=nn.Conv1d(4,4,5,padding=2,bias=True); s.pool2=nn.MaxPool1d(5)
        s.conv3=nn.Conv1d(4,8,5,padding=2,bias=True); s.pool3=nn.MaxPool1d(5)
        s.conv4=nn.Conv1d(8,8,5,padding=2,bias=True); s.pool4=nn.MaxPool1d(5)
        s.gap=nn.AdaptiveAvgPool1d(1); s.fc=nn.Linear(8,nc,bias=True)
    def forward(s,x):
        if x.dim()==2: x=x.unsqueeze(1)
        x=s.pool1(s.conv1(x)); x=s.pool2(s.conv2(x))
        x=s.pool3(s.conv3(x)); x=s.pool4(F.relu(s.conv4(x)))
        return s.fc(s.gap(x).squeeze(-1))


def metrics(model, X, y, dev):
    with torch.no_grad():
        lg = model(torch.tensor(X).to(dev)).cpu()
        prob = torch.softmax(lg, 1)[:, 1].numpy(); pred = lg.argmax(1).numpy()
    acc = (pred == y).mean()
    f1m = f1_score(y, pred, average='macro', zero_division=0)
    f1a = f1_score(y, pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0,1]); tn,fp,fn,tp = cm.ravel()
    sens = tp/(tp+fn) if (tp+fn) else 0; spec = tn/(tn+fp) if (tn+fp) else 0
    auc = roc_auc_score(y, prob) if len(set(y))>1 else float('nan')
    return dict(acc=acc,f1m=f1m,f1a=f1a,sens=sens,spec=spec,auc=auc,cm=cm,
                af=int((y==1).sum()),non=int((y==0).sum()))


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(SEED)

    print("Loading Chapman train/test...")
    Xtr, ytr = chap_xy(ChapmanECGDataset(CHAP_DIR, split='train', seed=SEED))
    Xcte, ycte = chap_xy(ChapmanECGDataset(CHAP_DIR, split='test', seed=SEED))
    print(f"Chapman train: AF={int((ytr==1).sum())} non-AF={int((ytr==0).sum())}")

    # oversample train balanced
    nc = max(int((ytr==0).sum()), int((ytr==1).sum())); pX, pY = [], []
    for c in (0,1):
        ci = np.where(ytr==c)[0]; pick = rng.choice(ci, size=nc, replace=True)
        pX.append(Xtr[pick]); pY.append(ytr[pick])
    Xb = torch.tensor(np.concatenate(pX)).to(dev); yb = torch.tensor(np.concatenate(pY)).to(dev)

    model = ECG_4488_2().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
    bs, epochs = 128, 40; n = len(yb)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=dev)
        for i in range(0,n,bs):
            b=perm[i:i+bs]; opt.zero_grad(); lossf(model(Xb[b]), yb[b]).backward(); opt.step()
        if (ep+1)%10==0 or ep==0: print(f"  ep {ep+1}")
    model.eval()

    print("Loading PTB-XL...");  Xp, yp = load_ptbxl()
    print("Loading AFDB...");    Xa, ya = load_afdb()

    R = {
        'Chapman (in-dist test)': metrics(model, Xcte, ycte, dev),
        'PTB-XL (zero-shot)':     metrics(model, Xp,   yp,   dev),
        'AFDB (zero-shot)':       metrics(model, Xa,   ya,   dev),
    }

    L = ["# PROBE J - Train Chapman AF/non-AF -> zero-shot PTB-XL & AFDB\n",
         "- Model 4-4-8-8 FC 8->2 (622 params), lead II train, 2500@250Hz, oversample TRAIN\n",
         "- AF: Chapman=AFIB+AF, PTB-XL=AFIB|AFLT, AFDB=AFIB+AFL\n",
         "- CAVEAT: AFDB uses ECG1 (Holter) vs lead II resting -> inherent shift\n\n",
         "| target | %AF | acc | macroF1 | AF-F1 | sens | spec | AUC |\n",
         "|---|---|---|---|---|---|---|---|\n"]
    for name, m in R.items():
        pct = 100*m['af']/(m['af']+m['non'])
        L.append(f"| {name} | {pct:.1f}% | {m['acc']:.4f} | {m['f1m']:.4f} | {m['f1a']:.4f} | "
                 f"{m['sens']:.3f} | {m['spec']:.3f} | {m['auc']:.4f} |\n")
    for name, m in R.items():
        L.append(f"\n### {name}  (AF={m['af']}, non-AF={m['non']})\n```\n{m['cm']}\n```\n")
    report = "".join(L)
    with open(os.path.join(OUT_DIR, 'REPORT_chapman2cross.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print("\n" + report)
    print(f"Saved -> {os.path.join(OUT_DIR, 'REPORT_chapman2cross.md')}")


if __name__ == "__main__":
    main()
