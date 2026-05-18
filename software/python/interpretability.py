"""
Interpretability Analysis — ECG_1DCNN
======================================
Three types of analysis for sequential CNN:

  Layer 1 — Layer-wise Ablation:
    Kill từng Conv layer output → layer nào critical?

  Layer 2 — Confidence Calibration:
    Phân phối softmax confidence khi đúng vs sai.
    Model có well-calibrated không?

  Layer 3 — Noise Robustness:
    Accuracy vs Gaussian noise level (SNR: inf → 10 dB).
    Model có fragile với input perturbation không?

Run:
    cd /home/duc/Thesis/software/python
    python interpretability.py
"""

import sys, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.dataset  import get_dataloaders
from utils.evaluate import compute_metrics, evaluate_model
from model.model import ECG_1DCNN, build_model

DATA_DIR   = "/home/duc/Thesis/data/Chapman"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "../results/best_model.pth")
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "../results")
CLASS_NAMES = ["AFIB", "GSVT", "SB", "SR"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    model = build_model(num_classes=4).to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


# ── Inference helpers ─────────────────────────────────────────────────────────

def run_inference(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for ecg, lbl, _ in loader:
            ecg = ecg.unsqueeze(1).float().to(DEVICE)
            preds.extend(model(ecg).argmax(1).cpu().numpy())
            labels.extend(lbl.numpy())
    return np.array(preds), np.array(labels)


def run_inference_with_confidence(model, loader):
    """Returns preds, labels, max softmax confidence per sample."""
    model.eval()
    preds, labels, confs = [], [], []
    with torch.no_grad():
        for ecg, lbl, _ in loader:
            ecg    = ecg.unsqueeze(1).float().to(DEVICE)
            logits = model(ecg)
            probs  = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(lbl.numpy())
            confs.extend(conf.cpu().numpy())
    return np.array(preds), np.array(labels), np.array(confs)


# ── Layer 1: Layer-wise ablation ──────────────────────────────────────────────

class LayerKilledV3(nn.Module):
    """Forward but zero-out output of a specific Conv layer."""

    def __init__(self, base: ECG_1DCNN, kill_layer: int):
        """kill_layer: 1..4 corresponding to conv1..conv4"""
        super().__init__()
        self.base       = base
        self.kill_layer = kill_layer

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Conv1
        x = self.base.conv1(x)
        if self.kill_layer == 1:
            x = torch.zeros_like(x)
        x = self.base.pool1(x)

        # Conv2
        x = self.base.conv2(x)
        if self.kill_layer == 2:
            x = torch.zeros_like(x)
        x = self.base.pool2(x)

        # Conv3
        x = self.base.conv3(x)
        if self.kill_layer == 3:
            x = torch.zeros_like(x)
        x = self.base.pool3(x)

        # Conv4 + ReLU
        x = F.relu(self.base.conv4(x))
        if self.kill_layer == 4:
            x = torch.zeros_like(x)
        x = self.base.pool4(x)

        x = self.base.gap(x).squeeze(-1)
        return self.base.fc(x)


def run_layer_ablation(base_model, loader, labels):
    print("\n" + "="*60)
    print("LAYER 1: LAYER-WISE ABLATION")
    print("="*60)

    layer_info = {
        1: "Conv1 (1→4,  K=5) RF=20ms  [beat onset]",
        2: "Conv2 (4→8,  K=5) RF=120ms [QRS complex]",
        3: "Conv3 (8→8,  K=5) RF=620ms [multi-beat]",
        4: "Conv4 (8→16, K=5) RF=3.1s  [full window]",
    }

    results = {}

    # Full model baseline
    full_preds, _ = run_inference(base_model, loader)
    full_metrics  = compute_metrics(full_preds, labels, CLASS_NAMES)
    results["Full"] = full_metrics

    print(f"\n  {'Variant':<45} {'Accuracy':>10}  {'F1-macro':>10}")
    print("  " + "-"*68)
    print(f"  {'Full model':<45} {full_metrics['accuracy']*100:>9.2f}%  "
          f"{full_metrics['f1_macro']:>10.4f}")

    for k in range(1, 5):
        patched = LayerKilledV3(base_model, kill_layer=k).to(DEVICE)
        preds, _ = run_inference(patched, loader)
        m = compute_metrics(preds, labels, CLASS_NAMES)
        results[f"kill_conv{k}"] = m
        print(f"  Kill {layer_info[k]:<40} {m['accuracy']*100:>9.2f}%  "
              f"{m['f1_macro']:>10.4f}")

    # Per-class F1 drop table
    print(f"\n  Per-Class F1 drop when Conv layer is killed:")
    header = f"  {'Class':<8}" + "".join(f"  {'No C'+str(k):>10}" for k in range(1,5))
    print(header)
    print("  " + "-"*52)
    for cls in CLASS_NAMES:
        f_full = full_metrics["per_class"][cls]["f1"]
        row = f"  {cls:<8}"
        for k in range(1, 5):
            drop = f_full - results[f"kill_conv{k}"]["per_class"][cls]["f1"]
            row += f"  {drop:>+10.4f}"
        print(row)

    print()
    print("  Diễn giải:")
    print("    Drop lớn → layer đó mang thông tin discriminative cho class")
    print("    Drop đều nhau → không có layer đặc biệt nào dominant")
    print("    (So sánh: v7_nodiff AFIB drop 0.84 khi kill rhythm branch)")

    return results


# ── Layer 2: Confidence calibration ──────────────────────────────────────────

def run_confidence_analysis(base_model, loader, labels):
    print("\n" + "="*60)
    print("LAYER 2: CONFIDENCE CALIBRATION")
    print("="*60)

    preds, _, confs = run_inference_with_confidence(base_model, loader)

    correct = (preds == labels)
    conf_correct   = confs[correct]
    conf_incorrect = confs[~correct]

    print(f"\n  Correct predictions   (n={correct.sum():4d}): "
          f"mean conf = {conf_correct.mean():.4f}  "
          f"min = {conf_correct.min():.4f}")
    print(f"  Incorrect predictions (n={(~correct).sum():4d}): "
          f"mean conf = {conf_incorrect.mean():.4f}  "
          f"min = {conf_incorrect.min():.4f}")

    # Expected Calibration Error (ECE) — 10 bins
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n   = len(confs)

    print(f"\n  ECE calibration (10 bins):")
    print(f"  {'Bin':>16}  {'Count':>7}  {'Avg Conf':>10}  {'Accuracy':>10}  {'|gap|':>8}")
    print("  " + "-"*60)

    for i in range(n_bins):
        lo, hi   = bin_edges[i], bin_edges[i+1]
        mask     = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        acc_bin  = correct[mask].mean()
        conf_bin = confs[mask].mean()
        gap      = abs(conf_bin - acc_bin)
        ece     += gap * mask.sum() / n
        print(f"  [{lo:.1f}, {hi:.1f})  {mask.sum():>7d}  "
              f"{conf_bin:>10.4f}  {acc_bin:>10.4f}  {gap:>8.4f}")

    print(f"\n  Expected Calibration Error (ECE) = {ece:.4f}")
    print()
    print("  Diễn giải:")
    print("    ECE < 0.05 → well-calibrated")
    print("    ECE > 0.10 → overconfident (data-driven models thường bị)")
    print("    v7_nodiff để so sánh: [chạy tương tự nếu cần]")

    return preds, confs, correct, ece


def plot_confidence(confs, correct, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Confidence Analysis — ECG_1DCNN", fontsize=12)

    # Left: confidence histogram correct vs incorrect
    ax = axes[0]
    ax.hist(confs[correct],  bins=20, alpha=0.6, color="#2ca02c", label=f"Correct (n={correct.sum()})")
    ax.hist(confs[~correct], bins=20, alpha=0.6, color="#d62728", label=f"Incorrect (n={(~correct).sum()})")
    ax.set_xlabel("Max Softmax Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)

    # Right: reliability diagram
    ax = axes[1]
    n_bins     = 10
    bin_edges  = np.linspace(0, 1, n_bins + 1)
    bin_accs   = []
    bin_confs  = []
    bin_counts = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask   = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        bin_accs.append(correct[mask].mean())
        bin_confs.append(confs[mask].mean())
        bin_counts.append(mask.sum())

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.6, color="#1f77b4", label="Actual accuracy")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved confidence figure → {out_path}")


# ── Layer 3: Noise robustness ─────────────────────────────────────────────────

def run_noise_robustness(base_model, loader, labels):
    print("\n" + "="*60)
    print("LAYER 3: NOISE ROBUSTNESS (Gaussian noise)")
    print("="*60)

    # SNR in dB: inf (no noise) → 10 dB (heavy noise)
    snr_db_list = [None, 40, 30, 20, 15, 10]
    results     = {}

    print(f"\n  {'SNR':>10}  {'Accuracy':>10}  {'F1-macro':>10}  {'Drop':>8}")
    print("  " + "-"*44)

    # Baseline (no noise)
    base_preds, _ = run_inference(base_model, loader)
    base_metrics  = compute_metrics(base_preds, labels, CLASS_NAMES)
    base_acc      = base_metrics["accuracy"]
    print(f"  {'No noise':>10}  {base_acc*100:>9.2f}%  "
          f"{base_metrics['f1_macro']:>10.4f}  {'—':>8}")
    results["no_noise"] = base_metrics

    for snr_db in snr_db_list[1:]:
        noisy_preds, noisy_labels = run_noisy_inference(base_model, loader, snr_db)
        m    = compute_metrics(noisy_preds, noisy_labels, CLASS_NAMES)
        drop = base_acc - m["accuracy"]
        print(f"  {f'{snr_db} dB':>10}  {m['accuracy']*100:>9.2f}%  "
              f"{m['f1_macro']:>10.4f}  {drop*100:>+7.2f}%")
        results[f"snr_{snr_db}db"] = m

    print()
    print("  Diễn giải:")
    print("    Drop < 2% at 20dB → robust (clinical ECG SNR ~25–30 dB)")
    print("    Drop > 5% at 20dB → fragile (data-driven overfitting to clean data)")
    print("    ECG acquisition noise ~ 20–25 dB (muscle artifact, baseline wander)")

    return results


def run_noisy_inference(model, loader, snr_db):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for ecg, lbl, _ in loader:
            ecg   = ecg.unsqueeze(1).float().to(DEVICE)   # (B, 1, 2500)
            # Compute signal power per sample
            sig_power   = ecg.pow(2).mean(dim=-1, keepdim=True)   # (B, 1, 1)
            noise_power = sig_power / (10 ** (snr_db / 10.0))
            noise       = torch.randn_like(ecg) * noise_power.sqrt()
            ecg_noisy   = ecg + noise
            preds.extend(model(ecg_noisy).argmax(1).cpu().numpy())
            labels.extend(lbl.numpy())
    return np.array(preds), np.array(labels)


def plot_noise_robustness(noise_results, out_path):
    snr_labels = ["No noise", "40 dB", "30 dB", "20 dB", "15 dB", "10 dB"]
    keys       = ["no_noise"] + [f"snr_{s}db" for s in [40, 30, 20, 15, 10]]
    accs       = [noise_results[k]["accuracy"] * 100 for k in keys]
    f1s        = [noise_results[k]["f1_macro"] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Noise Robustness — ECG_1DCNN", fontsize=12)

    x = np.arange(len(snr_labels))
    ax.plot(x, accs, "o-", color="#1f77b4", label="Accuracy (%)")
    ax.plot(x, [f*100 for f in f1s], "s--", color="#ff7f0e", label="F1-macro (%)")
    ax.axvline(3, color="green", linestyle=":", linewidth=1.2,
               label="Clinical ECG SNR (~20 dB)")
    ax.set_xticks(x)
    ax.set_xticklabels(snr_labels)
    ax.set_ylabel("Performance (%)")
    ax.set_xlabel("Noise Level")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Annotate drop at 20dB
    drop = accs[0] - accs[3]
    ax.annotate(f"Δ={drop:.1f}%",
                xy=(3, accs[3]), xytext=(3.3, accs[3] - 5),
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved noise figure → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    _, _, test_loader = get_dataloaders(DATA_DIR, batch_size=128)

    model = load_model()
    print(model.get_layer_info())

    # Ground-truth labels
    _, labels = run_inference(model, test_loader)

    # ── Full model baseline ────────────────────────────────────────
    full_preds, _   = run_inference(model, test_loader)
    full_metrics    = compute_metrics(full_preds, labels, CLASS_NAMES)
    print("\n" + "="*60)
    print("BASELINE — ECG_1DCNN (float, no quantization)")
    print("="*60)
    print(f"  Accuracy : {full_metrics['accuracy']:.4f}")
    print(f"  F1-macro : {full_metrics['f1_macro']:.4f}")
    print(f"\n  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-"*42)
    for cls in CLASS_NAMES:
        m = full_metrics["per_class"][cls]
        print(f"  {cls:<8} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    # ── Layer 1: Ablation ──────────────────────────────────────────
    ablation_results = run_layer_ablation(model, test_loader, labels)

    # ── Layer 2: Confidence ────────────────────────────────────────
    preds, confs, correct, ece = run_confidence_analysis(model, test_loader, labels)
    plot_confidence(confs, correct,
                    os.path.join(OUT_DIR, "confidence_analysis.png"))

    # ── Layer 3: Noise robustness ──────────────────────────────────
    noise_results = run_noise_robustness(model, test_loader, labels)
    plot_noise_robustness(noise_results,
                          os.path.join(OUT_DIR, "noise_robustness.png"))

    # ── Save JSON ──────────────────────────────────────────────────
    out = {
        "baseline": {
            "accuracy": full_metrics["accuracy"],
            "f1_macro": full_metrics["f1_macro"],
            "per_class_f1": {c: full_metrics["per_class"][c]["f1"] for c in CLASS_NAMES},
        },
        "layer_ablation": {
            name: {
                "accuracy": m["accuracy"],
                "per_class_f1": {c: m["per_class"][c]["f1"] for c in CLASS_NAMES},
            }
            for name, m in ablation_results.items()
        },
        "confidence": {
            "ece": ece,
            "mean_conf_correct":   float(confs[correct].mean()),
            "mean_conf_incorrect": float(confs[~correct].mean()),
        },
        "noise_robustness": {
            k: {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"]}
            for k, m in noise_results.items()
        },
    }
    out_json = os.path.join(OUT_DIR, "interpretability.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_json}")


if __name__ == "__main__":
    main()
