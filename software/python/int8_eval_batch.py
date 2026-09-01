"""
Batch INT8 (int8×int8) evaluation that is BIT-EXACT with the RTL pipeline.

Unlike quantization.qat_int8.int8_forward (which uses AdaptiveAvgPool1d = float
average at GAP), this reproduces the hardware GAP exactly: integer floor(sum/4).
So the argmax here equals what the RTL produces per sample — this is the number
we compare the hardware full-test-set golden against.

Also reports float32 accuracy from the same checkpoint's dequantized path for
the side-by-side float-vs-INT8 tables.

Data sources:
  --npz PATH         : pre-split .npz, uses X_test/y_test
  --byclass DIR      : folder-per-class tree of .npy (e.g. georgia_by_class);
                       whole tree is the test set (zero-shot cross-dataset)

Usage:
  python int8_eval_batch.py --checkpoint results/ningba/qat_int8/model_qat_int8.pth \
      --npz ../../data/ningba_processed/ningbo_dataset.npz --out results/ningba/int8_eval
"""
import os, sys, json, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_recall_fscore_support)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantization.qat_int8 import round_shift, ECG_1DCNN_QAT

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']


def int8_forward_bitexact(qat_model, x, w_int8, b_int8, nb, w_shift, input_shift):
    """Batch INT8 forward matching RTL exactly (integer floor GAP)."""
    device = next(qat_model.parameters()).device
    if x.dim() == 2:
        x = x.unsqueeze(1)
    x = x.to(device)
    x = torch.clamp(torch.round(x * (2.0 ** input_shift)), -127, 127)

    def conv(x_in, name, relu=False):
        w = torch.tensor(w_int8[name].astype(np.float32)).to(device)
        n = nb[name]
        b = torch.tensor(np.round(b_int8[name] * (2.0 ** n)).astype(np.float32)).to(device)
        layer = getattr(qat_model, name)
        acc = F.conv1d(x_in, w, b, padding=layer.padding)
        out = torch.clamp(round_shift(acc, n), -127, 127)
        if relu:
            out = torch.clamp(out, min=0)
        return out

    x = qat_model.pool1(conv(x, 'conv1'))
    x = qat_model.pool2(conv(x, 'conv2'))
    x = qat_model.pool3(conv(x, 'conv3'))
    x = qat_model.pool4(conv(x, 'conv4', relu=True))    # (B, 8, 4)

    gap = torch.floor(x.sum(dim=-1) / 4.0)              # RTL [9:2] slice, floor
    w_fc = torch.tensor(w_int8['fc'].astype(np.float32)).to(device)
    b_fc = torch.tensor(np.round(b_int8['fc'] * (2.0 ** w_shift['fc'])).astype(np.float32)).to(device)
    return F.linear(gap, w_fc, b_fc)                    # (B, 4) INT32 logits


def load_data(args):
    # Input outlier clip (±args.clip) applied consistently with training preprocess
    # so the input_shift calibration domain matches. npz test is already clipped if
    # it was built clipped; re-clipping is idempotent. By-class .npy (Georgia) is
    # clipped here to the same bound.
    if args.npz:
        d = np.load(args.npz)
        X, y = d['X_test'].astype(np.float32), d['y_test'].astype(np.int64)
    else:
        X, y = [], []
        for cidx, cname in enumerate(CLASS_NAMES):
            for f in sorted(glob.glob(os.path.join(args.byclass, cname, '*.npy'))):
                X.append(np.load(f)); y.append(cidx)
        X, y = np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)
    if args.clip > 0:
        X = np.clip(X, -args.clip, args.clip).astype(np.float32)
    return X, y


def plot_cm(cm, title, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i][j], ha='center', va='center',
                    color='white' if cm[i][j] > cm.max()/2 else 'black', fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def plot_roc(y_true, logits, title, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    probs = torch.softmax(torch.from_numpy(logits.astype(np.float32)), dim=1).numpy()
    yb = np.eye(4)[y_true]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    aucs = {}
    all_fpr = np.unique(np.concatenate([roc_curve(yb[:, i], probs[:, i])[0] for i in range(4)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(4):
        fpr, tpr, _ = roc_curve(yb[:, i], probs[:, i])
        aucs[i] = auc(fpr, tpr)
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        ax.plot(fpr, tpr, lw=1, label=f"{CLASS_NAMES[i]} (AUC={aucs[i]:.3f})")
    mean_tpr /= 4
    macro = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, 'k--', lw=2, label=f"macro (AUC={macro:.3f})")
    ax.plot([0, 1], [0, 1], ':', color='gray')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(title); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    return {str(i): float(aucs[i]) for i in range(4)}, float(macro)


def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro')
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2,3], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    print(f"\n=== {name}: acc={acc:.4f}  F1-macro={f1m:.4f} ===")
    print(f"{'class':<6}{'prec':>8}{'rec':>8}{'f1':>8}{'supp':>8}")
    for i, c in enumerate(CLASS_NAMES):
        print(f"{c:<6}{p[i]:>8.4f}{r[i]:>8.4f}{f[i]:>8.4f}{s[i]:>8d}")
    print("CM (rows=true, cols=pred):")
    print("        " + "".join(f"{c:>7}" for c in CLASS_NAMES))
    for i, c in enumerate(CLASS_NAMES):
        print(f"{c:<6}" + "".join(f"{cm[i][j]:>7d}" for j in range(4)))
    return dict(acc=float(acc), f1_macro=float(f1m),
                precision=p.tolist(), recall=r.tolist(), f1=f.tolist(),
                support=s.tolist(), confusion=cm.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--npz', default=None)
    ap.add_argument('--byclass', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='eval')
    ap.add_argument('--cm-title', default=None,
                    help='CM title prefix, e.g. "Test" or "Cross-data"; '
                         'renders as "Confusion Matrix Float32/Int8 <prefix>"')
    ap.add_argument('--clip', type=float, default=16.0,
                    help='Clip input to ±clip (match training preprocess); 0=off')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    qat = ECG_1DCNN_QAT(c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
                        c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'])
    qat.load_state_dict(ckpt['model_state_dict'])
    qat.eval()

    w_int8 = {k: np.array(v, dtype=np.int8)    for k, v in ckpt['w_int8'].items()}
    b_int8 = {k: np.array(v, dtype=np.float32) for k, v in ckpt['b_int8'].items()}
    nb, w_shift, ishift = ckpt['nb'], ckpt['w_shift'], ckpt['input_shift_bits']

    X, y = load_data(args)
    print(f"[INFO] {args.tag}: {len(y)} samples, dist={np.bincount(y, minlength=4).tolist()}")

    # ---- INT8 bit-exact + float32 forward ----
    int8_logits, float_logits = [], []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i+256])
            int8_logits.append(int8_forward_bitexact(qat, xb, w_int8, b_int8, nb, w_shift, ishift).numpy())
            float_logits.append(qat(xb.unsqueeze(1), quantize=False).numpy())
    int8_logits = np.concatenate(int8_logits)
    float_logits = np.concatenate(float_logits)
    int8_preds = int8_logits.argmax(1)
    float_preds = float_logits.argmax(1)

    res = {}
    res['float32'] = report(f"{args.tag} FLOAT32", y, float_preds)
    res['int8']    = report(f"{args.tag} INT8 (bit-exact GAP)", y, int8_preds)
    res['int8_vs_float_agree'] = float((int8_preds == float_preds).mean())
    print(f"\n[INFO] INT8 vs float32 prediction agreement: {res['int8_vs_float_agree']:.4f}")

    # ---- plots (CM + ROC) for both float32 and INT8 ----
    suf = f" {args.cm_title}" if args.cm_title else ''
    plot_cm(np.array(res['float32']['confusion']), f"Confusion Matrix Float32{suf}",
            os.path.join(args.out, f'{args.tag}_cm_float32.png'))
    plot_cm(np.array(res['int8']['confusion']), f"Confusion Matrix Int8{suf}",
            os.path.join(args.out, f'{args.tag}_cm_int8.png'))
    res['float32']['roc_auc'], res['float32']['macro_auc'] = plot_roc(
        y, float_logits, f"{args.tag} float32 ROC", os.path.join(args.out, f'{args.tag}_roc_float32.png'))
    res['int8']['roc_auc'], res['int8']['macro_auc'] = plot_roc(
        y, int8_logits, f"{args.tag} INT8 ROC", os.path.join(args.out, f'{args.tag}_roc_int8.png'))
    print(f"[INFO] macro-AUC  float32={res['float32']['macro_auc']:.4f}  int8={res['int8']['macro_auc']:.4f}")

    # save per-sample int8 argmax = the full-test golden the RTL is checked against
    np.save(os.path.join(args.out, f'{args.tag}_int8_argmax.npy'), int8_preds.astype(np.uint8))
    np.save(os.path.join(args.out, f'{args.tag}_labels.npy'), y.astype(np.uint8))
    with open(os.path.join(args.out, f'{args.tag}_metrics.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(f"[INFO] wrote {args.out}/{args.tag}_metrics.json + argmax/labels npy")


if __name__ == '__main__':
    main()
