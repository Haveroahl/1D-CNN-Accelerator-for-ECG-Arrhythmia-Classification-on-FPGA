"""
ANN -> SNN conversion feasibility study (verify-before-build)
=============================================================
Goal: BEFORE writing any LIF RTL, measure on the *deployed* pruned ECG model:
  (1) SNN accuracy as a function of T (rate-coding timesteps), vs the float ANN.
  (2) Average spike rate (sparsity) per layer  -> decides the energy story.

Why this matters for the ICDV "unified CNN/SNN core" pitch:
  - If accuracy needs large T, latency = T x CNN-latency -> SNN must win on energy
    via sparsity, not latency. Spike rate tells us if that is plausible.
  - If spike rate is dense (~>0.5 events/neuron/step), the event-driven energy
    advantage collapses and we should re-pitch as "flexibility" not "low-energy".

Conversion scheme (hand-written LIF/IF, no external SNN lib):
  - Constant-current ("analog") input: layer-1 reads the real INT8-domain input
    every timestep (standard, accuracy-friendly first-layer encoding).
  - Each conv layer's output current i_L = pool_L(conv_L(s_{L-1})) drives an
    integrate-and-fire neuron with soft reset:
        V += i ;  fire when |V| crosses Vth ;  V -= sign*Vth
  - Conv1-3 have NO ReLU in this architecture -> SIGNED IF (spikes in {-1,0,+1}).
  - Conv4 has ReLU -> positive-only IF (spikes in {0,+1}).  relu(max)=max(relu).
  - GAP+FC are linear -> applied once to the accumulated Conv4 spike counts.
  - Vth per layer = threshold-balanced to the calibrated max |current| (no saturation).

This is a feasibility measurement, NOT the final bit-exact model. It tells us
whether the SNN direction is worth building in RTL.

Run:
    cd d:\\Thesis101\\software\\python
    .\\.venv\\Scripts\\Activate.ps1
    python snn\\ann2snn_feasibility.py
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prune_finetune import ECG_1DCNN_Pruned
from utils.dataset import get_dataloaders

PAD = 2  # all conv layers padding=2


# ----------------------------------------------------------------------------
# IF neuron firing (soft reset)
# ----------------------------------------------------------------------------
def signed_fire(V, vth):
    """Signed integrate-and-fire: emit +1/-1 when |V|>=vth, soft-reset. Returns spikes."""
    pos = (V >= vth).float()
    neg = (V <= -vth).float()
    return pos - neg  # in {-1, 0, +1}


def pos_fire(V, vth):
    """Positive-only IF (ReLU equivalent)."""
    return (V >= vth).float()


# ----------------------------------------------------------------------------
# Sequential data-based threshold calibration in the SPIKING regime.
#   Vth_L = max |i_L| where i_L is driven by the ACTUAL spikes of layer L-1
#   (not the ANN's continuous activations). This avoids firing-rate death:
#   downstream currents driven by unit spikes are much smaller than the ANN's
#   continuous-valued currents, so ANN-calibrated thresholds never fire.
# Calibrate each layer with the upstream layers already thresholded, over T_cal
# timesteps, on one calibration batch.
# ----------------------------------------------------------------------------
def _pct(t, pct):
    """Robust upper percentile of |t| (subsample if large to bound memory)."""
    a = t.abs().flatten()
    if a.numel() > 200000:
        a = a[torch.randperm(a.numel(), device=a.device)[:200000]]
    return torch.quantile(a, pct).item()


@torch.no_grad()
def calibrate_vth(model, loader, device, T_cal=32, pct=0.999):
    model.eval()
    x = next(iter(loader))[0].to(device)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    vth = {}

    # Layer 1: analog constant-current input -> ANN current is correct
    i1 = model.pool1(model.conv1(x))
    vth[1] = _pct(i1, pct)

    # produce s1 sequence, measure layer-2 spiking currents, set vth[2], etc.
    def run_layer(i_const_or_seq, vth_l, signed=True):
        """Run one IF layer for T_cal steps, return list of spike tensors."""
        V = torch.zeros_like(i_const_or_seq[0] if isinstance(i_const_or_seq, list) else i_const_or_seq)
        out = []
        for t in range(T_cal):
            i = i_const_or_seq[t] if isinstance(i_const_or_seq, list) else i_const_or_seq
            V = V + i
            s = signed_fire(V, vth_l) if signed else pos_fire(V, vth_l)
            V = V - s * vth_l
            out.append(s)
        return out

    # graded spikes: transmit s*vth (the reset amount) so downstream currents and
    # conv biases stay consistent with the ANN (value transmission, not unit rate).
    s1_seq = run_layer(i1, vth[1], signed=True)

    i2_seq = [model.pool2(model.conv2(s * vth[1])) for s in s1_seq]
    vth[2] = _pct(torch.stack(i2_seq), pct)
    s2_seq = run_layer(i2_seq, vth[2], signed=True)

    i3_seq = [model.pool3(model.conv3(s * vth[2])) for s in s2_seq]
    vth[3] = _pct(torch.stack(i3_seq), pct)
    s3_seq = run_layer(i3_seq, vth[3], signed=True)

    i4_seq = [model.pool4(model.conv4(s * vth[3])) for s in s3_seq]
    vth[4] = _pct(torch.stack(i4_seq), pct)
    return vth


# ----------------------------------------------------------------------------
# Spiking forward pass
# ----------------------------------------------------------------------------
@torch.no_grad()
def snn_forward(model, x, vth, T):
    """
    Run T-timestep spiking inference. Returns (logits, spike_events_per_layer).
    spike_events: total |spike| count summed over the batch & T, per layer.
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)
    B = x.shape[0]

    # membrane potentials (lazy-init on first current to get shapes)
    V1 = V2 = V3 = V4 = None
    s4_acc = None  # accumulated Conv4 spike counts -> rate readout
    events = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    numel = {}

    for t in range(T):
        # Layer 1: constant-current analog input
        i1 = model.pool1(model.conv1(x))
        if V1 is None:
            V1 = torch.zeros_like(i1); numel[1] = i1[0].numel()
        V1 = V1 + i1
        s1 = signed_fire(V1, vth[1]); V1 = V1 - s1 * vth[1]
        events[1] += s1.abs().sum().item()

        # Layer 2 (signed, no ReLU) — receives graded spikes s1*vth1
        i2 = model.pool2(model.conv2(s1 * vth[1]))
        if V2 is None:
            V2 = torch.zeros_like(i2); numel[2] = i2[0].numel()
        V2 = V2 + i2
        s2 = signed_fire(V2, vth[2]); V2 = V2 - s2 * vth[2]
        events[2] += s2.abs().sum().item()

        # Layer 3 (signed, no ReLU)
        i3 = model.pool3(model.conv3(s2 * vth[2]))
        if V3 is None:
            V3 = torch.zeros_like(i3); numel[3] = i3[0].numel()
        V3 = V3 + i3
        s3 = signed_fire(V3, vth[3]); V3 = V3 - s3 * vth[3]
        events[3] += s3.abs().sum().item()

        # Layer 4 (positive-only IF == ReLU)
        i4 = model.pool4(model.conv4(s3 * vth[3]))
        if V4 is None:
            V4 = torch.zeros_like(i4); numel[4] = i4[0].numel(); s4_acc = torch.zeros_like(i4)
        V4 = V4 + i4
        s4 = pos_fire(V4, vth[4]); V4 = V4 - s4 * vth[4]
        events[4] += s4.abs().sum().item()
        s4_acc = s4_acc + s4

    # Rate readout: average Conv4 spike rate -> scale back by vth -> GAP+FC (linear)
    rate4 = (s4_acc / T) * vth[4]           # approx mean post-ReLU activation
    g = model.gap(rate4).squeeze(-1)        # (B, c4_out)
    logits = model.fc(g)                    # (B, 4)

    # normalize events -> events per neuron per timestep
    spike_rate = {k: events[k] / (B * numel[k] * T) for k in events}
    return logits, spike_rate


# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_ann(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
    return correct / total


@torch.no_grad()
def eval_snn(model, loader, device, vth, T, max_batches=None):
    model.eval()
    correct = total = 0
    rate_sum = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}; nb = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        logits, sr = snn_forward(model, x, vth, T)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
        for k in rate_sum:
            rate_sum[k] += sr[k]
        nb += 1
    avg_rate = {k: rate_sum[k] / nb for k in rate_sum}
    return correct / total, avg_rate


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='results/qat_int8/model_qat_int8.pth')
    ap.add_argument('--data_dir', default='../../data/Chapman')
    ap.add_argument('--T_list', default='1,2,4,8,16,32,64,128,256')
    ap.add_argument('--vth_pct', type=float, default=0.999,
                    help='percentile for threshold balancing (1.0 = max-norm)')
    ap.add_argument('--max_batches', type=int, default=None,
                    help='limit test batches for a quick run (None = full test set)')
    ap.add_argument('--out', default='snn/ann2snn_feasibility.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- load deployed pruned model (float / fake-quant weights) ----
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = ECG_1DCNN_Pruned(ck['c1_out'], ck['c2_out'], ck['c3_out'], ck['c4_out'])
    missing, unexpected = model.load_state_dict(ck['model_state_dict'], strict=False)
    # only fq_* (fake-quant observers) should be unexpected; conv/fc must all load
    assert not missing, f"missing weights: {missing}"
    model.to(device).eval()
    print(f"Loaded {args.checkpoint}  (channels {ck['c1_out']},{ck['c2_out']},{ck['c3_out']},{ck['c4_out']})")

    _, _, test_loader = get_dataloaders(args.data_dir, batch_size=128)

    # ---- ANN baseline ----
    ann_acc = eval_ann(model, test_loader, device)
    print(f"\nANN float baseline accuracy: {ann_acc*100:.2f}%")

    # ---- threshold balancing ----
    vth = calibrate_vth(model, test_loader, device, pct=args.vth_pct)
    print(f"Calibrated Vth per layer: " +
          ", ".join(f"L{k}={vth[k]:.3f}" for k in sorted(vth)))

    # ---- sweep T ----
    T_list = [int(t) for t in args.T_list.split(',')]
    results = {'ann_acc': ann_acc, 'vth': vth, 'sweep': []}
    print(f"\n{'T':>4} | {'SNN acc':>8} | {'drop vs ANN':>11} | "
          f"{'rate L1':>8} {'L2':>6} {'L3':>6} {'L4':>6} | {'mean rate':>9}")
    print("-" * 78)
    for T in T_list:
        acc, rate = eval_snn(model, test_loader, device, vth, T, args.max_batches)
        mean_rate = sum(rate.values()) / len(rate)
        results['sweep'].append({'T': T, 'snn_acc': acc, 'rate': rate, 'mean_rate': mean_rate})
        print(f"{T:>4} | {acc*100:>7.2f}% | {(ann_acc-acc)*100:>+10.2f}% | "
              f"{rate[1]:>8.3f} {rate[2]:>6.3f} {rate[3]:>6.3f} {rate[4]:>6.3f} | "
              f"{mean_rate:>9.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")

    # ---- verdict hint ----
    best = max(results['sweep'], key=lambda r: r['snn_acc'])
    print(f"\nBest: T={best['T']}  acc={best['snn_acc']*100:.2f}%  "
          f"(drop {(ann_acc-best['snn_acc'])*100:+.2f}%)  mean spike rate={best['mean_rate']:.3f}")
    print("Energy story is plausible only if accuracy recovers at a T whose "
          "mean spike rate stays well below ~0.5 events/neuron/step.")


if __name__ == '__main__':
    main()
