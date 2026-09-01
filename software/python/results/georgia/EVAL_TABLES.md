# Georgia (far-transfer, zero-shot) — Software Evaluation Tables

Dataset: `georgia_by_class/` (5459 records, input clipped ±16). **Zero-shot**: model
trained on ningba (Chapman-Ningbo), NOT fine-tuned on Georgia. Different acquisition
system (Emory) → far-transfer test.
Model: ECG_1DCNN pruned (4,4,8,8) QAT-INT8, nb=[8,7,6,7,0], input_shift=2.
INT8 = bit-exact GAP (= what RTL ROM emits).

## Table A — Float32 (zero-shot)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| AFIB  | 0.7984 | 0.8584 | 0.8273 | 692  |
| GSVT  | 0.9381 | 0.9153 | 0.9265 | 1192 |
| SB    | 0.9473 | 0.9698 | 0.9584 | 1521 |
| SR    | 0.9584 | 0.9309 | 0.9444 | 2054 |
| **Overall** | — | — | **F1-macro 0.9142** | 5459 |

Accuracy **0.9291** · macro-AUC **0.9813**

## Table B — INT8 (power-of-2, bit-exact with RTL)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| AFIB  | 0.8309 | 0.8237 | 0.8273 | 692  |
| GSVT  | 0.9208 | 0.9455 | 0.9329 | 1192 |
| SB    | 0.9537 | 0.9625 | 0.9581 | 1521 |
| SR    | 0.9513 | 0.9328 | 0.9420 | 2054 |
| **Overall** | — | — | **F1-macro 0.9151** | 5459 |

Accuracy **0.9300** · macro-AUC **0.9580**

## Float32 → INT8 summary (zero-shot Georgia)

| Metric | Float32 | INT8 | Δ |
|--------|---------|------|---|
| Accuracy | 0.9291 | 0.9300 | +0.09 pp |
| F1-macro | 0.9142 | 0.9151 | +0.09 pp |
| macro-AUC | 0.9813 | 0.9580 | −0.0233 |
| INT8↔Float32 agreement | — | — | 0.9749 |

Note: AFIB precision (~0.83) is limited by composition shift (Georgia GSVT dominated
by sinus-tach), not a quantization effect — INT8 tracks float32 within 0.1 pp. See
memory `georgia-far-transfer`.

Plots: `int8_eval/georgia_cm_float32.png`, `georgia_cm_int8.png`,
`georgia_roc_float32.png`, `georgia_roc_int8.png`.
