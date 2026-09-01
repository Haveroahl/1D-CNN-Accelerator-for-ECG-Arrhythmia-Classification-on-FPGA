# Ningba (Chapman-Ningbo extended) — Software Evaluation Tables

Dataset: `ningbo_dataset_clip16.npz` (input clipped ±16 → input_shift=2), test = 4973 records.
Model: ECG_1DCNN pruned (4,4,8,8), 640 params. QAT-INT8 power-of-2, nb=[8,7,6,7,0].
INT8 numbers use bit-exact GAP (integer floor sum/4) = what the RTL ROM build emits.

Pipeline provenance: train (dense 1244p) → prune+finetune (640p) → QAT-INT8 →
export → RTL/ ROM (conv*_w.hex + cnn_controller.v nb hard-code = 8/7/6/7).

## Table A — Float32

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| AFIB  | 0.9151 | 0.9442 | 0.9294 | 1130 |
| GSVT  | 0.9122 | 0.9321 | 0.9220 | 869  |
| SB    | 0.9735 | 0.9844 | 0.9789 | 1791 |
| SR    | 0.9801 | 0.9180 | 0.9481 | 1183 |
| **Overall** | — | — | **F1-macro 0.9446** | 4973 |

Accuracy **0.9503** · macro-AUC **0.9938**

## Table B — INT8 (power-of-2, bit-exact with RTL)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| AFIB  | 0.9267 | 0.9062 | 0.9163 | 1130 |
| GSVT  | 0.8698 | 0.9459 | 0.9063 | 869  |
| SB    | 0.9756 | 0.9810 | 0.9783 | 1791 |
| SR    | 0.9670 | 0.9172 | 0.9414 | 1183 |
| **Overall** | — | — | **F1-macro 0.9356** | 4973 |

Accuracy **0.9427** · macro-AUC **0.9712**

## Float32 → INT8 summary

| Metric | Float32 | INT8 | Δ |
|--------|---------|------|---|
| Accuracy | 0.9503 | 0.9427 | −0.76 pp |
| F1-macro | 0.9446 | 0.9356 | −0.90 pp |
| macro-AUC | 0.9938 | 0.9712 | −0.0226 |
| INT8↔Float32 prediction agreement | — | — | 0.9761 |

Plots: `int8_eval/ningba_cm_float32.png`, `ningba_cm_int8.png`,
`ningba_roc_float32.png`, `ningba_roc_int8.png`.
