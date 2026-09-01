"""
PROBE I — Chapman AF vs non-AF (2-class), reuse existing loader split.
================================================================================
- AF = original 4-class label 0 (AFIB+AF) ; non-AF = labels 1/2/3 (GSVT/SB/SR).
- Reuses ChapmanECGDataset 80/10/10 split (per-record random => patient-independent,
  since each Chapman record is one patient). Lead II, 250Hz, 2500 samples.
- Model: deployed 4-4-8-8, FC 8->2, 622 params. Oversample TRAIN only.
- Imbalance ~21% AF => report F1 / sensitivity / specificity / AUC, not just accuracy.
"""
import os, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, classification_report

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.dataset import ChapmanECGDataset

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Chapman')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'results', 'probe_chapman_af')
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42


def to_xy(ds):
    X = np.stack(ds.records).astype(np.float32)        # (N, 2500)
    y4 = np.asarray(ds.labels, np.int64)
    y = (y4 != 0).astype(np.int64)                     # 0=AF (label0), 1=non-AF  -> flip below
    # define AF as positive=1 for clarity
    y_af = (y4 == 0).astype(np.int64)                  # 1 = AF, 0 = non-AF
    return X, y_af


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


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(SEED)

    print("Loading Chapman (train/test) ...")
    tr_ds = ChapmanECGDataset(DATA_DIR, split='train', seed=SEED)
    te_ds = ChapmanECGDataset(DATA_DIR, split='test',  seed=SEED)
    Xtr, ytr = to_xy(tr_ds)
    Xte, yte = to_xy(te_ds)
    print(f"Train: AF={int((ytr==1).sum())} non-AF={int((ytr==0).sum())}")
    print(f"Test : AF={int((yte==1).sum())} non-AF={int((yte==0).sum())} ({100*yte.mean():.1f}% AF)")

    # oversample TRAIN only to balance
    n_count = max(int((ytr==0).sum()), int((ytr==1).sum()))
    pX, pY = [], []
    for c in (0, 1):
        ci = np.where(ytr == c)[0]
        pick = rng.choice(ci, size=n_count, replace=True)
        pX.append(Xtr[pick]); pY.append(ytr[pick])
    Xtr_b = torch.tensor(np.concatenate(pX)).to(dev)
    ytr_b = torch.tensor(np.concatenate(pY)).to(dev)
    Xte_t = torch.tensor(Xte).to(dev)

    model = ECG_4488_2().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
    bs, epochs = 128, 40; n = len(ytr_b)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            b = perm[i:i+bs]; opt.zero_grad()
            lossf(model(Xtr_b[b]), ytr_b[b]).backward(); opt.step()
        if (ep+1) % 10 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xte_t).argmax(1).cpu().numpy() == yte).mean()
            print(f"  ep {ep+1}  test_acc {acc:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(Xte_t).cpu()
        prob = torch.softmax(logits, 1)[:, 1].numpy()
        pred = logits.argmax(1).numpy()
    acc = (pred == yte).mean()
    f1m = f1_score(yte, pred, average='macro', zero_division=0)
    f1_af = f1_score(yte, pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    sens = tp/(tp+fn) if (tp+fn) else 0.0; spec = tn/(tn+fp) if (tn+fp) else 0.0
    auc = roc_auc_score(yte, prob)

    L = []
    L.append("# PROBE I - Chapman AF vs non-AF (2-class), 4-4-8-8 FC 8->2\n")
    L.append("- AF = AFIB+AF (orig label 0); non-AF = GSVT/SB/SR; patient-independent 80/10/10\n")
    L.append("- Lead II, 250Hz, 2500 samples, 622 params, oversample TRAIN balanced, 40 ep\n")
    L.append(f"\n## Test set\n- AF={int((yte==1).sum())} / non-AF={int((yte==0).sum())} ({100*yte.mean():.1f}% AF)\n")
    L.append(f"\n## Results\n- Accuracy: {acc:.4f}\n- **Macro-F1: {f1m:.4f}**\n- AF F1: {f1_af:.4f}\n")
    L.append(f"- **Sensitivity (AF recall): {sens:.4f}**\n- **Specificity: {spec:.4f}**\n- **ROC-AUC: {auc:.4f}**\n")
    L.append(f"\n### Confusion (rows=true [non-AF, AF])\n```\n{cm}\n```\n")
    L.append(f"\n### sklearn report\n```\n{classification_report(yte, pred, target_names=['non-AF','AF'], zero_division=0)}\n```\n")
    L.append("\n## AF detection across datasets (same 8-PE arch, 622 params, input 2500)\n")
    L.append("| dataset | setup | macroF1 | AUC | sens | spec |\n|---|---|---|---|---|---|\n")
    L.append("| AFDB | inter-patient 5fold | 0.8201 | 0.9052 | 0.930 | 0.747 |\n")
    L.append("| AFDB->PTB-XL | zero-shot | 0.7674 | 0.9512 | 0.820 | 0.923 |\n")
    L.append(f"| Chapman | patient-indep | {f1m:.4f} | {auc:.4f} | {sens:.3f} | {spec:.3f} |\n")
    report = "".join(L)
    with open(os.path.join(OUT_DIR, 'REPORT.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print("\n" + report)
    print(f"Saved -> {os.path.join(OUT_DIR, 'REPORT.md')}")


if __name__ == "__main__":
    main()
