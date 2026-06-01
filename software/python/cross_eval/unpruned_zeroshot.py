"""Zero-shot PTB-XL eval for unpruned float32 model (channels 4,8,8,16)."""

import sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class ECG_CNN(nn.Module):
    def __init__(self, c1=4, c2=8, c3=8, c4=16, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1,  c1, 5, padding=2, bias=True)
        self.conv2 = nn.Conv1d(c1, c2, 5, padding=2, bias=True)
        self.conv3 = nn.Conv1d(c2, c3, 5, padding=2, bias=True)
        self.conv4 = nn.Conv1d(c3, c4, 5, padding=2, bias=True)
        self.pool  = nn.MaxPool1d(5)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(c4, num_classes, bias=True)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


classes = ['AFIB', 'GSVT', 'SB', 'SR']

ckpt = torch.load(r'd:\Thesis101\software\python\results\best_model.pth',
                  map_location='cpu', weights_only=False)
model = ECG_CNN(c1=4, c2=8, c3=8, c4=16)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

d = np.load(r'd:\Thesis101\data\ptbxl_processed\ptbxl_dataset.npz')
loader = DataLoader(
    TensorDataset(torch.tensor(d['X_test']), torch.tensor(d['y_test'])),
    batch_size=128, shuffle=False)

all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in loader:
        preds = model(xb).argmax(dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(yb.tolist())

acc = accuracy_score(all_labels, all_preds)
f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
f1p = f1_score(all_labels, all_preds, average=None, labels=[0,1,2,3], zero_division=0).tolist()
cm  = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3]).tolist()

print('=== Unpruned float32 zero-shot PTB-XL ===')
print('acc=' + str(round(acc,4)) + '  f1_macro=' + str(round(f1,4)))
print('Per-class F1: ' + '  '.join(classes[i]+':'+str(round(f1p[i],3)) for i in range(4)))
print('Confusion matrix (row=true, col=pred):')
print('       ' + '  '.join(c.rjust(5) for c in classes))
for i, row in enumerate(cm):
    print('  ' + classes[i].rjust(4) + ': ' + '  '.join(str(v).rjust(5) for v in row))

print()
print('--- So sanh voi pruned QAT-INT8 zero-shot (C2) ---')
print('Unpruned float32: acc=' + str(round(acc,4)) + '  f1=' + str(round(f1,4)))
print('Pruned  QAT-INT8: acc=0.7714  f1=0.6486')
print('Pruned  float32 (C6): acc=0.7714  f1=0.6486')
