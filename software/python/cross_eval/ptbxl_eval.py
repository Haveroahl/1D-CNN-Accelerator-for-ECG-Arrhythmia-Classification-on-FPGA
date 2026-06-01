"""Phase A — Cross-Dataset Evaluation: Chapman -> PTB-XL

5 evaluation modes (contribution C3):
  C1. Chapman  -> Chapman  (in-distribution baseline)     [from saved ckpt]
  C2. Chapman  -> PTB-XL   zero-shot
  C3. Chapman  -> PTB-XL   linear probe (freeze conv, retrain FC only)
  C4. Chapman  -> PTB-XL   full fine-tune
  C5. PTB-XL from scratch
  C6. Float32  -> PTB-XL   zero-shot (decompose quant vs distribution drop)

Usage:
    python cross_eval/ptbxl_eval.py \
        --ckpt    software/python/results/qat_int8/model_qat_int8.pth \
        --ptbxl   data/ptbxl_processed/ptbxl_dataset.npz \
        --chapman_dir data/Chapman \
        --output  software/python/results/cross_eval
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']


# ── Model definition (channels 4,4,8,8 — matches hardware V1) ─────────────

class ECG_CNN(nn.Module):
    def __init__(self, c1=4, c2=4, c3=8, c4=8, num_classes=4):
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

    def freeze_conv(self):
        for name, param in self.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


def load_qat_checkpoint(ckpt_path, device):
    """Load QAT checkpoint into ECG_CNN float model (for eval/finetune)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c1 = ckpt.get('c1_out', 4)
    c2 = ckpt.get('c2_out', 4)
    c3 = ckpt.get('c3_out', 8)
    model = ECG_CNN(c1=c1, c2=c2, c3=c3, c4=ckpt.get('c4_out', 8))
    # Load float weights from model_state_dict (strip fq_ keys)
    sd = ckpt['model_state_dict']
    float_sd = {k: v for k, v in sd.items()
                if not k.startswith('fq_') and k in model.state_dict()}
    model.load_state_dict(float_sd, strict=False)
    return model.to(device)


# ── Metrics ───────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(yb.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_per = f1_score(all_labels, all_preds, average=None,
                      labels=[0,1,2,3], zero_division=0).tolist()
    cm  = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3]).tolist()
    return {'acc': acc, 'f1_macro': f1, 'f1_per_class': f1_per,
            'confusion_matrix': cm, 'n': len(all_labels)}


# ── Training loop ──────────────────────────────────────────────────────────

def finetune(model, train_loader, val_loader, device,
             epochs=20, lr=1e-3, label='finetune'):
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc, best_sd = 0.0, None

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        val_m = evaluate(model, val_loader, device)
        if val_m['acc'] > best_val_acc:
            best_val_acc = val_m['acc']
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 5 == 0:
            print(f"    [{label}] ep {ep+1}/{epochs}  val_acc={val_m['acc']:.4f}")

    if best_sd:
        model.load_state_dict(best_sd)
    return model


# ── Dataset helpers ───────────────────────────────────────────────────────

def npz_loaders(npz_path, batch_size=128):
    d = np.load(npz_path)
    def make(X, y):
        return DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(y)),
            batch_size=batch_size, shuffle=False, num_workers=0)
    train = DataLoader(
        TensorDataset(torch.tensor(d['X_train']), torch.tensor(d['y_train'])),
        batch_size=batch_size, shuffle=True, num_workers=0)
    val  = make(d['X_val'],  d['y_val'])
    test = make(d['X_test'], d['y_test'])
    return train, val, test


def chapman_test_loader(data_dir, batch_size=128):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.dataset import ChapmanECGDataset
    ds = ChapmanECGDataset(data_dir, split='test')
    # Chapman returns (ecg, label, hr) — wrap to (ecg, label) only
    from torch.utils.data import Dataset
    class Wrapper(Dataset):
        def __init__(self, d): self.d = d
        def __len__(self): return len(self.d)
        def __getitem__(self, i):
            ecg, label, _ = self.d[i]
            return ecg, label
    return DataLoader(Wrapper(ds), batch_size=batch_size, shuffle=False, num_workers=0)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',         default=r'software/python/results/qat_int8/model_qat_int8.pth')
    p.add_argument('--ptbxl',        default=r'data/ptbxl_processed/ptbxl_dataset.npz')
    p.add_argument('--chapman_dir',  default=r'data/Chapman')
    p.add_argument('--output',       default=r'software/python/results/cross_eval')
    p.add_argument('--finetune_epochs', type=int, default=20)
    p.add_argument('--batch_size',   type=int, default=128)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    ptbxl_train, ptbxl_val, ptbxl_test = npz_loaders(args.ptbxl, args.batch_size)
    results = {}

    # ── C1: Chapman -> Chapman (in-distribution) ────────────────────────
    print("\n[C1] Chapman -> Chapman (in-distribution)")
    model_c1 = load_qat_checkpoint(args.ckpt, device)
    try:
        chap_test = chapman_test_loader(args.chapman_dir, args.batch_size)
        m = evaluate(model_c1, chap_test, device)
        results['C1_chapman_indist'] = m
        print(f"     acc={m['acc']:.4f}  f1={m['f1_macro']:.4f}")
    except Exception as e:
        print(f"     [SKIP] Chapman loader failed: {e}")
        results['C1_chapman_indist'] = {'error': str(e)}

    # ── C2: Chapman -> PTB-XL zero-shot ─────────────────────────────────
    print("\n[C2] Chapman -> PTB-XL zero-shot")
    model_c2 = load_qat_checkpoint(args.ckpt, device)
    m = evaluate(model_c2, ptbxl_test, device)
    results['C2_zeroshot'] = m
    print(f"     acc={m['acc']:.4f}  f1={m['f1_macro']:.4f}")

    # ── C3: Linear probe (freeze conv, retrain FC on PTB-XL) ────────────
    print("\n[C3] Linear probe (freeze conv, retrain FC)")
    model_c3 = load_qat_checkpoint(args.ckpt, device)
    model_c3.freeze_conv()
    model_c3 = finetune(model_c3, ptbxl_train, ptbxl_val, device,
                        epochs=args.finetune_epochs, lr=1e-3, label='linear_probe')
    m = evaluate(model_c3, ptbxl_test, device)
    results['C3_linear_probe'] = m
    print(f"     acc={m['acc']:.4f}  f1={m['f1_macro']:.4f}")
    # Save adapter weights
    torch.save(model_c3.fc.state_dict(),
               os.path.join(args.output, 'ptbxl_fc_adapter.pth'))

    # ── C4: Full fine-tune ───────────────────────────────────────────────
    print("\n[C4] Full fine-tune (unfreeze all)")
    model_c4 = load_qat_checkpoint(args.ckpt, device)
    model_c4.unfreeze_all()
    model_c4 = finetune(model_c4, ptbxl_train, ptbxl_val, device,
                        epochs=args.finetune_epochs, lr=5e-4, label='full_finetune')
    m = evaluate(model_c4, ptbxl_test, device)
    results['C4_full_finetune'] = m
    print(f"     acc={m['acc']:.4f}  f1={m['f1_macro']:.4f}")
    torch.save(model_c4.state_dict(),
               os.path.join(args.output, 'ptbxl_finetuned.pth'))

    # ── C5: PTB-XL from scratch ──────────────────────────────────────────
    print("\n[C5] PTB-XL from scratch")
    model_c5 = ECG_CNN().to(device)
    model_c5 = finetune(model_c5, ptbxl_train, ptbxl_val, device,
                        epochs=args.finetune_epochs, lr=1e-3, label='from_scratch')
    m = evaluate(model_c5, ptbxl_test, device)
    results['C5_from_scratch'] = m
    print(f"     acc={m['acc']:.4f}  f1={m['f1_macro']:.4f}")

    # ── C6: Float32 zero-shot (decompose quant vs distribution drop) ─────
    print("\n[C6] Float32 zero-shot (no quantization)")
    model_c6 = load_qat_checkpoint(args.ckpt, device)
    # Use float weights as-is (QAT checkpoint stores float weights post-training)
    m = evaluate(model_c6, ptbxl_test, device)
    results['C6_float_zeroshot'] = m
    print(f"     acc={m['acc']:.4f}  f1={m['f1_macro']:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    row_fmt = "{:<30} {:>8} {:>10}"
    print(row_fmt.format("Mode", "Acc", "F1-macro"))
    print("-"*50)
    for key, val in results.items():
        if 'acc' in val:
            print(row_fmt.format(key, f"{val['acc']:.4f}", f"{val['f1_macro']:.4f}"))

    out_path = os.path.join(args.output, 'ptbxl_cross_eval.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved: {out_path}")


if __name__ == '__main__':
    main()
