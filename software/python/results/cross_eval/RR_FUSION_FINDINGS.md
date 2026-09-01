# RR / HR Fusion Study — Findings for Paper Discussion & Limitations

**Date:** 2026-06-05
**Model:** ECG_1DCNN Pruned (4,4,8,8), QAT-INT8 power-of-2 (model_qat_int8.pth)
**Question:** Does the CNN learn RR-interval / heart-rate features, and can an
explicit HR signal fix the SB/SR cross-dataset confusion?

> Supports Paper_Proposal_Q3 Section 7 (Discussion) and RQ3 (quant vs
> distribution decomposition). All numbers reproducible from scripts listed
> at the bottom. These are exploratory analyses, not part of the shipped
> hardware pipeline.

---

## 1. What features does the CNN actually use? (probe_rr_morph.py)

Two probes on Chapman test (n=1065), float pruned model:

**Test 1 — HR correlation (does the CNN encode heart rate?)**
- Best single GAP feature vs ground-truth VentricularRate: **|r| = 0.74**
- Linear regression GAP(all dims) -> HR: **R² = 0.71**
- logit[GSVT] vs HR: ρ = +0.83 ; logit[SB] vs HR: ρ = -0.77
- **Conclusion:** the CNN encodes heart rate strongly, via spike density in the
  pooling windows (GAP removes absolute timing but NOT spike count).

**Test 2 — beat-shuffle (does it use RR-interval / rhythm timing?)**
Shuffle beat order (destroy RR-interval, keep per-beat morphology), per-class
accuracy baseline -> shuffled:

| Class | base -> shuffle | drop | uses |
|-------|-----------------|------|------|
| AFIB  | 92.7% -> 91.3%  | -1.4 | morphology (shuffle-immune) |
| GSVT  | 91.3% -> 83.1%  | -8.3 | mostly morphology |
| SB    | 99.5% -> 53.7%  | -45.8| **RR-interval / rhythm timing** |
| SR    | 92.0% -> 46.7%  | -45.3| **RR-interval / rhythm timing** |

- **Conclusion:** HYBRID model. AFIB/GSVT decided by morphology; SB/SR decided
  by RR/rate. This is WHY SB/SR is the cross-dataset failure mode — it depends
  on a rate boundary, and the implicit rate threshold the CNN learned does not
  transfer.
- Caveat: beat-shuffle introduces splice artifacts, but the per-class CONTRAST
  (AFIB -1.4 vs SB -46) is artifact-immune — uniform artifact would hurt all
  classes equally. Test 1 (correlation) has no such caveat and confirms
  independently.

---

## 2. Can explicit HR fix SB/SR? (rr_fusion_probe, compare_detectors, gated_fusion)

Fusion rule: if argmax in {SB,SR}: class = SB if HR_bpm < THR else SR;
else keep argmax (AFIB/GSVT untouched).

**Cross-dataset (PTB-XL zero-shot), THR calibrated on val:**

| Detector | THR | acc | f1-macro | vs baseline 0.7714 |
|----------|-----|-----|----------|--------------------|
| baseline (CNN only)     | —  | 0.7714 | 0.6486 | — |
| scipy find_peaks        | 50 | 0.9046 | 0.7514 | **+13.3 pp** |
| Pan-Tompkins (Liu 2023) | 50 | 0.9072 | 0.7689 | **+13.6 pp** |

**In-distribution (Chapman), same fusion:**

| Detector | THR | acc | f1-macro | vs baseline 0.9446 |
|----------|-----|-----|----------|--------------------|
| baseline (CNN only)     | —  | 0.9446 | 0.9379 | — |
| scipy find_peaks        | 60 | 0.8808 | 0.8810 | **-6.4 pp** |
| Pan-Tompkins            | 60 | 0.9089 | 0.9055 | **-3.6 pp** |
| **ground-truth HR**     | 60 | 0.9465 | 0.9398 | **+0.19 pp** |

---

## 3. Why fusion is NOT shipped — three hard findings

1. **Fusion helps when the CNN is wrong, hurts when it is right.**
   PTB-XL (CNN poor, SB F1=0.305) gains +13pp; Chapman (CNN near-perfect,
   SB F1=0.986) loses 3.6pp. Blanket fusion is the wrong policy.

2. **The bottleneck is R-peak detection error, not the fusion design.**
   With ground-truth HR, Chapman fusion is +0.19pp (harmless). With detected
   HR, MAE ≈ 8 bpm (scipy) / 7.8 bpm (Pan-Tompkins) on z-scored 250-Hz signals.
   Even Liu's fully-mapped Pan-Tompkins leaves MAE ~8 bpm here -> an RTL
   hr_estimator.v would STILL hurt Chapman if fused unconditionally.

3. **Optimal THR_BPM depends on the dataset, and the system is THR-sensitive.**
   PTB-XL optimal = 50 bpm; Chapman optimal = 60 bpm. On PTB-XL, THR 50->60
   drops acc 0.907 -> 0.802 (-10.6pp). The "fixed clinical 60 bpm" threshold is
   not optimal for either, because detector bias + HR distribution differ. This
   is distribution shift re-expressed as a threshold parameter.

4. **Confidence-gated fusion does not work.**
   Hypothesis: only fuse when CNN is unsure (|logit_SB - logit_SR| small).
   But the SB/SR logit gap is identical across datasets (median 10, p25 6,
   p75 13-15) — the CNN is equally unsure on SB/SR in BOTH. No gate separates
   "Chapman keep" from "PTB-XL fix". Dead end.

---

## 4. Takeaway for the paper

- The CNN is a hybrid morphology + implicit-rate classifier (Section 1 evidence).
- SB/SR is a clinical rate-boundary (60 bpm) the CNN learns implicitly and which
  does NOT transfer — confirmed quantitatively (Section 1 + 2).
- Explicit HR fusion CAN recover cross-dataset SB/SR (+13pp) but trades off
  in-distribution accuracy, bounded by R-peak detection error and dataset-
  dependent threshold. This is a quantified demonstration that SB/SR ambiguity
  is inherent and not fully solvable by post-processing — strengthens RQ3.
- Liu 2023 (the SoTA reference) outputs HR in PARALLEL (dual-function), never
  fuses it into the class decision — consistent with our finding that fusion is
  not unconditionally safe. Our study quantifies WHY.

---

## Scripts (exploratory; reproduce numbers above)

- `software/python/probe_rr_morph.py` — HR correlation + beat-shuffle (Section 1)
- `software/python/cross_eval/pan_tompkins.py` — Pan-Tompkins HR (golden ref, matches Liu 2023)
- `software/python/cross_eval/rr_fusion_probe.py` — PTB-XL fusion, val-calibrated THR
- `software/python/cross_eval/rr_fusion_chapman.py` — Chapman fusion + GT-HR upper bound
- `software/python/cross_eval/compare_detectors.py` — scipy vs Pan-Tompkins, both datasets
- `software/python/cross_eval/gated_fusion.py` — confidence-gated fusion (negative result)
