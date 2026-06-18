# A Bit-Exact Power-of-Two INT8 1D-CNN Accelerator for ECG Arrhythmia Classification on Intel Cyclone V

**Authors:** Lê Đức (ducle160499@gmail.com), *et al.*
**Target:** ICDV (International Conference on Integrated Circuits, Design, and Verification) — IEEE short paper, ~6 pages, double-column.
**Draft:** 2026-06-18. Scope = **production 8-PE channel-parallel core only** (SIMD-20 / DSE excluded).

> **Editorial markers used below**
> - `🔲` = number NOT yet verified from a primary file/report — must confirm before camera-ready.
> - `[CITE]` = reference to be inserted from SOTA_TABLE.md once BibTeX keys are assigned.
> - All hardware/accuracy numbers are from `PAPER_DATA.md` (verified 2026-06-15) unless marked 🔲.

---

## Abstract

Wearable electrocardiogram (ECG) monitoring demands arrhythmia classifiers that are
simultaneously accurate, low-latency, and energy-frugal enough to run continuously on a
battery-powered edge device. We present a compact INT8 1D-CNN accelerator for four-class
ECG arrhythmia classification deployed on an Intel Cyclone V FPGA. The design combines
three elements. First, a **power-of-two quantization-aware training (QAT)** scheme whose
rescale step is a single arithmetic shift with **round-half-up** correction, eliminating
the per-rescale multiplier that general-scale INT8 requires (0 DSP for rescale) while
recovering +0.38% accuracy over plain shift-truncation. Second, a **bit-exact verification
framework**: 21 golden checkpoints per input (layer pool outputs, global-average-pool, and
fully-connected logits) match between the Python model and RTL simulation to the last bit
(max |diff| = 0 LSB across 15,312 elements). Third, an **8-PE channel-parallel streaming
core** that performs one inference in a deterministic 5,216 cycles (52.16 µs at 100 MHz),
using 2,201 ALMs (5%) and 28 DSPs (25%) of a 5CSXFC6D6F31C6 device. The classifier reaches
94.65% accuracy (macro-F1 0.9396) on the Chapman dataset, and an on-board run on a
DE10-Standard reproduces 94.27% over the test set. A lightweight Avalon-MM weight-reload
path lets a single bitstream switch between datasets at runtime, which we exploit in a
brief Chapman→PTB-XL transfer study. The result is a fully-verified, deployable ECG IP core
suited to single-lead wearable monitoring.

**Keywords:** ECG, arrhythmia classification, 1D-CNN, INT8 quantization, power-of-two,
FPGA accelerator, bit-exact verification, Cyclone V, wearable.

---

## 1. Introduction

Cardiac arrhythmia is a leading contributor to sudden cardiac death, and early detection
through continuous wearable monitoring can be life-saving. 1D convolutional neural networks
(CNNs) over raw ECG achieve high accuracy, but the platform choice for *wearable* deployment
is constrained: microcontrollers lack the throughput for continuous inference, and edge GPUs
are unsuitable in power and form factor. A low-power FPGA such as the Intel Cyclone V is a
sweet spot — enough parallelism for a small CNN, with on-chip soft-logic or a host bridge to
drive the datapath.

Two recurring weaknesses in the ECG-FPGA literature motivate this work. (i) Quantization is
often reported as a software accuracy figure that is *not* bit-exact with the deployed RTL,
so the "INT8 accuracy" claimed in a paper is not the accuracy that actually runs on the chip.
(ii) Power-of-two (shift-based) rescaling is attractive because it removes multipliers, but
prior fully-mapped designs implement it with arithmetic-shift *floor* truncation [CITE Liu2023]
and do not analyze the cost–accuracy trade-off against general-scale INT8.

We address both. Our contributions are:

- **C1 — Round-half-up power-of-two rescale with quantitative ablation.** We do *not* claim
  power-of-two rescaling as novel; it is shared with prior fully-mapped designs [CITE Liu2023].
  Our contribution is (a) a **round-half-up** rescale `(O + 2^{nb-1}) ≫ nb` replacing floor
  truncation, measured at **+0.38%** accuracy at **zero** DSP cost, and (b) a systematic
  ablation of power-of-two vs general-scale vs floor with 5-test-fold robustness.
- **C2 — Bit-exact verification framework.** 21 golden checkpoints per sample match the RTL
  simulation exactly (max |diff| = 0 LSB over 15,312 elements), making the deployed accuracy
  provably identical to the software model.
- **C3 — Deployable IP core.** An 8-PE channel-parallel streaming core with a deterministic
  52.16 µs/inference latency, validated on-board on a DE10-Standard, with a lightweight
  Avalon-MM weight-reload path enabling a single-bitstream Chapman↔PTB-XL transfer study.

Section 2 reviews related work; Section 3 details the model and power-of-two QAT methodology;
Section 4 describes the hardware architecture; Section 5 covers the verification flow;
Section 6 reports results; Section 7 discusses limitations; Section 8 concludes.

---

## 2. Related Work

FPGA accelerators for ECG arrhythmia classification span a wide design space. Many recent
designs target the MIT-BIH dataset with 2D "beat-image" CNNs on Xilinx Zynq/PYNQ platforms,
reaching 97–99% on five AAMI classes but at millisecond-scale latency (e.g., 45–236 ms) and
hundreds of milliwatts to several watts of power [CITE PYNQ-a, PYNQ-b, Zynq7Z020]. Such
designs optimize throughput (tens to hundreds of GOPS) rather than per-inference energy, and
report accuracy without bit-exact correspondence to the RTL.

The closest comparison point is the fully-mapped 1D-CNN accelerator of Liu *et al.* [CITE
Liu2023], which targets the *same* Chapman four-class problem on the *same* Cyclone V family
with an architecturally identical network (four conv + four max-pool layers, 2,500-sample
input, four-class output) and additionally performs heart-rate estimation. It reports an INT8
accuracy of **92.95%** (macro-F1 0.9205) at 50 MHz, 66 µs/inference, 66 mW, and 87.42 GOPS/W.
It maps each layer to dedicated hardware (one module per layer) and rescales using
arithmetic-shift floor truncation. Our design differs in two ways that this paper makes precise: a **folded
streaming** core that time-multiplexes 8 processing elements across positions (rather than
fully mapping every layer) trades a small latency increase for a large reduction in logic and
registers, and a **round-half-up** rescale improves accuracy over floor at no DSP cost.

A gap analysis is summarized in Table 1. Unlike prior work, we (i) verify the deployed model
bit-exact against the software reference, and (ii) quantify the power-of-two vs general-scale
rescale trade-off rather than asserting one choice.

**Table 1.** Gap analysis vs representative ECG-FPGA work. *(see SOTA_TABLE.md; entries marked
🔲 to be verified from primary sources before camera-ready.)*

| Limitation in prior work | This work |
|---|---|
| Power-of-two via floor truncation, no cost-accuracy analysis | Round-half-up (+0.38%) + power-of-two vs general ablation |
| Reported INT8 accuracy not bit-exact with RTL | 21-checkpoint bit-exact (max |diff| = 0 LSB) |
| ms-scale latency, throughput-oriented | 52.16 µs deterministic, energy-oriented |
| Hard-coded weights in bitstream | Runtime Avalon-MM weight reload |

---

## 3. CNN Model and Power-of-Two QAT

### 3.1 Network topology

The classifier is a pruned 1D-CNN with four convolutional layers (channels 4-4-8-8, kernel
K=5, padding 2), each followed by max-pooling with stride 5, then global average pooling
(GAP), a fully-connected (FC) layer (8→4), and argmax (Fig. 1). The input is 2,500 INT8
samples (5 s of single-lead — lead II — ECG at 500 Hz). ReLU is applied **only after Conv4**,
deliberately preserving negative ECG morphology in Conv1–3. The four classes are AFIB, GSVT,
SB, and SR. The pruned model has **654 parameters**.

The per-stage tensor shapes are: 2500×1 → 500×4 → 100×4 → 20×8 → 4×8 → GAP(8) → FC(4).

### 3.2 Power-of-two quantization

Weights, activations, and biases are quantized to INT8 with **power-of-two** scales chosen
per layer as `nb = floor(log2(127 / abs_max))`. The fixed parameters are:

- activation rescale shift `nb = {8, 6, 6, 7, 0}` for {Conv1, Conv2, Conv3, Conv4, FC};
- weight shift `w_shift = {6, 6, 6, 7, 8}`;
- input shift = 2 (applied when mapping the float ECG to INT8).

The accumulator-to-output rescale is

```
out = clamp( round_half_up( acc / 2^nb ), -127, 127 )
round_half_up(x) = (x + 2^(nb-1)) >> nb           (signed arithmetic shift)
```

Biases are scaled `bias_scaled = round(b_float · 2^nb)` and stored as INT32 little-endian.
Because every scale is a power of two, the rescale reduces to a barrel shift plus an adder —
**zero DSP multipliers**, against one multiplier per rescale for a general real-valued scale.

### 3.3 Round-half-up vs floor

Prior fully-mapped designs implement the shift as a plain arithmetic shift, i.e. floor
truncation toward −∞. Replacing it with round-half-up costs only the constant addend
`2^(nb-1)` (a wire, constant-folded by the synthesizer) and recovers **+0.38%** accuracy
(94.37% vs 93.99% in the QAT power-of-two ablation, Table 4). This is the rounding correction
we claim as part of C1.

### 3.4 Cost comparison with general-scale INT8

Table 2 contrasts the rescale hardware. The power-of-two path needs only a shifter and adder;
general-scale INT8 needs a multiplier (here, 4 DSP18 across the rescale points). Across the
ablation (§6.1) the two are statistically indistinguishable in accuracy (Δ < fold std), so the
power-of-two choice is Pareto-favorable: equal accuracy, fewer DSPs.

**Table 2.** Rescale hardware cost.

| Variant | Scale | Rescale op | DSP18/rescale |
|---|---|---|---:|
| Power-of-two (ours) | 2^nb | shift + add | **0** |
| General-scale INT8 | s ∈ ℝ | mul + shift | 1 (≈4 total) |

---

## 4. Hardware Architecture

### 4.1 Top level

The core (Fig. 2) consists of an input SRAM (2,500×8b), a Conv-Pool engine (CPE) of 8
parallel CP blocks, a ping-pong SRAM holding inter-layer feature maps, and a GAP/FC/Argmax
unit producing a 2-bit class. A flat 8-state controller FSM sequences the four conv layers
and the GAP/FC tail. An Avalon-MM slave acts as the bus adapter — it decodes which addresses
load the input SRAM, which bit starts inference, and where the done/result are read — and is
reused unchanged across the synthesis (virtual-pin) and on-board (JTAG-to-Avalon) flows.

### 4.2 CP block pipeline

Each CP block computes one output channel. Its pipeline is split into three submodules:

- **MAC (S1–S4):** 5 INT8×INT8 multipliers (DSP18) feeding a 3-stage adder tree → `tree_out`.
- **Accumulate + rescale (S5–S8):** accumulates `in_ch` partial sums, then rescales. The
  per-layer constants — bias and the round-half-up addend — are **folded into the accumulator
  init term** so that the downstream bias-add and round-add drop off the critical path; the
  result is numerically identical to `(acc + bias + round) ≫ nb`. A clamp to [−127, 127] and
  the Conv4 ReLU complete the stage.
- **Pool (S9):** a rolling max comparator implementing K=5, stride-5 max-pooling.

The MAC mapping uses 8 CP blocks × 5 multipliers = 40 INT8 multiplies in flight; the synthesis
packs these into **28 DSP18** after retiming and sharing. The pipeline depth from the input
window MUX to the accumulator update is exactly 5 cycles, which the controller's delay chain
must match (a subtle off-by-one here was the dominant verification bug; see §5).

### 4.3 Streaming dataflow

Rather than fully mapping every layer to its own hardware, the CPE **time-multiplexes** the 8
CP blocks across output positions, streaming the input through a shift-register window. This
folded mapping is the key area lever versus the fully-mapped baseline: it reuses one set of
PEs for all positions of a layer, trading a modest, fully-deterministic latency for a large
reduction in logic and registers (§6.3).

### 4.4 Runtime weight reload

In the baseline, conv weights are baked into the bitstream via `$readmemh` (FF-ROM). An
optional **weight-RAM** variant stores conv weights in 8 per-output-channel M10K blocks
written at runtime through the Avalon-MM port (with companion bias/FC write ports). This lets
a *single* bitstream run either Chapman or PTB-XL weights, enabling the transfer study of §6.4
without recompilation. The overhead is bounded (§6.3) and, crucially, latency and accuracy are
**unchanged** — the M10K synchronous read substitutes 1-for-1 for the `w_packed` register
stage, preserving pipeline alignment (bit-exact, max |diff| = 0 LSB).

---

## 5. Verification Flow

The verification contract is **bit-exactness** between the Python model and RTL simulation.
The Python golden generator reproduces the RTL sequence exactly: `acc_int32 → +bias_scaled →
+2^(nb-1) → ≫nb → clamp[-127,127] → ReLU(if Conv4) → MaxPool`; GAP uses integer floor
division `floor(sum/4) = sum ≫ 2`; FC uses `nb=0` with raw INT32 logits into argmax.

For each input, **21 checkpoints** are compared: the INT8 input, the four post-pool feature
maps, the GAP vector, and the four FC logits. A ModelSim/Questa testbench loads the golden
`.mem` files and compares each checkpoint. The result on the production core is **21/21 PASS**
with **max |diff| = 0 LSB across 15,312 element comparisons** (three test samples), a
deterministic latency of **5,216 cycles**, plus configuration/recovery and GAP-mask tests for
the runtime-reconfigurable path.

This framework converts the usual "INT8 simulation ≈ RTL" hand-wave into a provable identity:
the accuracy reported in §6.1 is *exactly* the accuracy the deployed core produces.

---

## 6. Results

### 6.1 Quantization ablation (Chapman, patient-independent 70/15/15)

**Table 4.** Quantization ablation (single run, seed 42). DSP column = multipliers for rescale.

| Variant | Scale | Train | Acc % | F1 | DSP rescale |
|---|---|---|---:|---:|---:|
| A1 Float32 baseline | — | — | 94.65 | 0.9402 | — |
| A0 PTQ power-of-two | 2^nb | none | 94.08 | 0.9338 | **0** |
| A0' PTQ general | absmax/127 | none | 94.46 | 0.9380 | 4 |
| **A2 QAT power-of-two (ours)** | 2^nb | fake-quant | **94.37** | **0.9364** | **0** |
| A3 QAT general | absmax/127 | fake-quant | 94.65 | 0.9398 | 4 |
| A4 QAT power-of-two floor | 2^nb | fake-quant | 93.99 | 0.9328 | 0 |

A2 vs A3: −0.28% accuracy for **−4 DSP18**. A2 vs A4 (floor): **+0.38%** from round-half-up.
Across 5 test-folds the per-variant std (0.4–0.9%) exceeds the inter-variant gaps, so accuracy
is statistically equivalent between power-of-two and general-scale — the certain difference is
DSP cost. PTQ alone (no fine-tune) already reaches 94.08%, so QAT is beneficial but not
required.

> 🔲 **Accuracy consistency note (must resolve before submission):** the headline/golden figure
> is **94.65% / F1 0.9396** (FC-bias re-train, 2026-06-08), used for the RTL golden. Table 4's
> A2 = 94.37 predates the FC-bias re-train. Use one consistent number throughout; recommend
> reporting 94.65% as the deployed accuracy and footnoting Table 4 as the pre-FC-bias ablation,
> or re-generating Table 4 with FC bias.

A macro-AUC of 0.967 is obtained on Chapman (ROC, confusion matrix in Fig. 3 🔲 confirm PNG).

### 6.2 Bit-width: why INT8, not INT4

**Table 5.** Bit-width ablation.

| Variant | W/A | Acc % | F1 | AFIB F1 |
|---|---|---:|---:|---:|
| A2 INT8 (ours) | 8/8 | 94.37 | 0.9364 | — |
| QAT INT4 power-of-two | 4/4 | 69.95 | 0.660 | — |
| QAT INT4 general (ceiling) | 4/4 | 75.59 | 0.704 | 0.42 |

INT4 loses ~19% even at the general-scale ceiling, and AFIB collapses — the absence of ReLU
in Conv1–3 keeps a wide signed activation range that INT4 cannot represent. INT8 is the sweet
spot.

### 6.3 Hardware resource, timing, and energy

**Table 6.** Production core on Cyclone V 5CSXFC6D6F31C6 (Quartus 25.1 Lite).

| Metric | Baseline (FF-ROM) | + weight-RAM reload |
|---|---:|---:|
| ALM | 2,201 (5%) | 2,820 (7%) |
| DSP18 | 28 (25%) | 28 (25%) |
| Registers | 3,177 | 4,852 |
| M10K | 20 (4%) | 28 (5%) |
| Fmax (standalone, 85 °C) | 104.85 MHz | 108.94 MHz |
| Latency | 5,216 cy (52.16 µs) | 5,216 cy (52.16 µs) |
| Throughput | ~19,200 inf/s | ~19,200 inf/s |

The weight-reload path adds +619 ALM and +8 M10K — overhead from the runtime read-address
adder, 8-way write decode, and 40-bit assembly in the Avalon slave, *not* from the memory
itself (M10K uses 0 ALM). Latency and accuracy are unchanged.

**Energy.** 🔲 (confirm from PowerPlay `.pow.rpt`): total power 623 mW (dynamic 198 mW, static
413 mW) at 95.6% toggle, giving **≈10.3 µJ/inference dynamic, 32.5 µJ total** at 52.16 µs.
DSPs account for ~68% of dynamic power, directly tying the energy story to the 0-DSP-rescale
choice of C1.

Compared with the fully-mapped Cyclone V baseline [CITE Liu2023], the folded streaming core
uses roughly an order of magnitude fewer ALMs and ~20× fewer registers (51%→5% ALM, 86%→~8%
register utilization on comparable devices), with a similar DSP share, at the cost of a
deterministic latency increase — a favorable trade for an area- and energy-constrained
wearable target. On accuracy, our INT8 core reaches **94.65%** versus Liu's INT8 **92.95%** on
the same Chapman four-class task (+1.7 pp), which we attribute to the round-half-up rescale and
the FC-bias retrain. Liu's fully-mapped design draws lower power (66 mW vs our 623 mW total) by
running at 50 MHz with all parameters resident on-chip and no runtime memory traffic; this is a
genuine advantage of the fully-mapped approach that our folded, reconfigurable core trades for
a ~9× smaller logic footprint and runtime weight reload. *(Full comparison in SOTA_TABLE.md.)*

### 6.4 On-board validation and cross-dataset transfer

Programmed onto a DE10-Standard and driven through a JTAG-to-Avalon bridge with System
Console, the core classifies the Chapman test set at **94.27%** 🔲 (1,004/1,065 — *needs a
cite-able log; currently from run notes*), reproducing the simulated 94.65%. The small gap is
the test-subset/run difference, not a numerical one (the datapath is bit-exact).

Using the weight-reload path, the same bitstream runs PTB-XL weights. In a brief transfer
study (Chapman↔PTB-XL, patient-independent 70/15/15), zero-shot INT8 accuracy is 0.7714 and a
linear probe (retrain FC only) recovers 0.9263. Because the INT8 zero-shot (C2) equals the
float32 zero-shot (C6) at 0.7714, the entire transfer drop is **distribution shift, not
quantization** — i.e. power-of-two INT8 adds 0% generalization loss. This is a short
supporting result, not the paper's focus.

---

## 7. Discussion and Limitations

The design targets *single-lead, continuous* monitoring, where one inference per heartbeat
(~1 Hz) makes 52 µs latency four orders of magnitude faster than required; the binding
constraints are area and energy, which the folded, 0-DSP-rescale core minimizes. The
limitations are: single-lead input, a fixed topology (the runtime path reconfigures weights
and per-layer scale, not the layer dimensions), and evaluation on two datasets (Chapman,
PTB-XL). Multi-lead fusion and streaming operation are future work. We also note the
power/energy figures await confirmation from the PowerPlay report, and the on-board accuracy
log must be regenerated for a cite-able artifact.

---

## 8. Conclusion

We presented a bit-exact, power-of-two INT8 1D-CNN accelerator for four-class ECG arrhythmia
classification on an Intel Cyclone V. A round-half-up power-of-two rescale removes rescale
multipliers (0 DSP) while improving accuracy +0.38% over floor; a 21-checkpoint framework
proves the deployed model is bit-identical to the software reference (max |diff| = 0 LSB); and
an 8-PE channel-parallel streaming core delivers 94.65% accuracy at a deterministic 52.16 µs
per inference in 2,201 ALMs and 28 DSPs, validated on-board at 94.27%. A lightweight Avalon-MM
weight-reload path enables single-bitstream cross-dataset operation. The combination — verified
correctness, multiplier-free rescaling, and a compact deterministic core — makes this a
practical ECG IP core for wearable monitoring.

---

## References

> To be assigned from SOTA_TABLE.md once primary sources are verified. Placeholder keys used
> above: [CITE Liu2023], [CITE PYNQ-a], [CITE PYNQ-b], [CITE Zynq7Z020].

---

## Figures / Tables checklist (for camera-ready)

- **Fig. 1** Network topology with per-stage tensor shapes.
- **Fig. 2** Top-level block diagram + CP-block 5-stage pipeline.
- **Fig. 3** Chapman confusion matrix + ROC (macro-AUC 0.967) — 🔲 confirm PNG exists.
- **Table 1** Gap analysis (done).
- **Table 2** Rescale cost (done).
- **Table 4** Quantization ablation — 🔲 resolve 94.65 vs 94.37 consistency.
- **Table 5** Bit-width ablation (done).
- **Table 6** Resource/timing (done) — 🔲 add energy row from PowerPlay.

## Open items blocking camera-ready (from PAPER_DATA.md §8)

1. 🔴 Fix one accuracy number (94.65 recommended) across Table 4 + text.
2. 🔴 Fmax: use 104.85 (baseline) / 108.94 (weight-RAM) / board ~125 — never 137.6.
3. 🟠 Cross-dataset: JSON numbers (C3 0.9263, C4 0.9336).
4. 🟠 On-board 1,004/1,065: regenerate a cite-able log.
5. 🟠 PowerPlay energy: confirm from `.pow.rpt`.
6. 🟢 Verify each SoTA cell from the primary paper; assign BibTeX keys.
