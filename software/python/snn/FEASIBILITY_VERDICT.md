# ANN→SNN Conversion Feasibility — Verdict (verify-before-build)

**Date:** 2026-06-16
**Script:** `snn/ann2snn_feasibility.py` · **Data:** `snn/ann2snn_feasibility.json`
**Model:** deployed pruned ECG_1DCNN (4-4-8-8), float/fake-quant weights from `qat_int8/model_qat_int8.pth`

## Question
Before writing any LIF RTL for the proposed **unified CNN/SNN reconfigurable core** (ICDV
novelty), measure on the *actual* deployed model: (1) SNN accuracy vs T (rate-coding
timesteps), (2) spike rate (sparsity → energy story).

## Method
Hand-written LIF/IF (no SNN lib). Constant-current analog input; per-layer IF neurons;
sequential data-based threshold balancing in the spiking regime (99th-pct norm); **graded
spikes** (transmit `s·Vth`) so conv biases stay consistent; Conv1–3 **signed** IF (no ReLU
in this architecture), Conv4 positive-only IF (=ReLU); GAP+FC applied to accumulated Conv4
spike rate.

## Result (full Chapman test set, 1065 samples)
| T | SNN acc | drop vs ANN (94.08%) | mean spike rate |
|---|---|---|---|
| 4   | 25.5% | −68.6 | 0.05 |
| 16  | **58.6%** | −35.5 | 0.18 |
| 32  | 54.6% | −39.5 | 0.21 |
| 64  | 52.3% | −41.8 | 0.23 |
| 128 | 52.7% | −41.4 | 0.24 |

## Verdict: naive post-training conversion does NOT work for this model
Accuracy **peaks at T=16 (58.6%) then declines** — non-monotonic. If the only error were
rate-coding noise it would converge monotonically toward the ANN; the peak-then-decline is
the signature of a **systematic accumulating bias**.

Three structural reasons this model is hostile to free ANN→SNN:
1. **4× MaxPool.** Max of a noisy per-timestep spike-rate estimate is biased high and does
   not commute with temporal averaging. Documented hard case (Rueckauer 2017 → use avg-pool).
   This is the dominant error (explains the non-monotonic curve).
2. **ReLU only after Conv4.** Conv1–3 are linear → require *signed* spikes; rate coding of
   signed, bias-carrying linear layers is far less robust than ReLU homogeneity.
3. **Tiny feature maps** (Conv4 output = 8×4 → GAP). Few neurons × low rate → very coarse
   rate code; not enough events to represent logits accurately.

Spike rate ~0.18–0.24 events/neuron/step is **not extremely sparse**, so even a working
conversion would not obviously beat CNN-INT8 on energy (latency also ×T).

## Implication for the ICDV "unified core" plan
- **Free conversion is off the table.** SNN viability requires **retraining**:
  avg-pool instead of max-pool, and/or **direct surrogate-gradient training** (BPTT). That
  is a software scope addition (new training pipeline + new golden/bit-exact framework),
  not a drop-in.
- The **dual-mode datapath sketch still holds** (shared MAC/adder-tree/weight-RAM; LIF
  back-end replacing rescale; power-of-2 leak) — the *hardware* kinship is real. The risk
  is entirely on the **model/accuracy side**, now quantified.
- **Recommended next step before committing RTL:** retrain a small avg-pool variant with
  surrogate-gradient SNN training, re-measure accuracy/spike-rate. Only if that clears
  ~93%+ at a spike rate low enough to beat CNN-INT8 energy does the RTL build pay off.

## Reproduce
```
cd d:\Thesis101\software\python ; .\.venv\Scripts\Activate.ps1
python snn\ann2snn_feasibility.py --T_list 4,16,32,64,128 --vth_pct 0.99
```
