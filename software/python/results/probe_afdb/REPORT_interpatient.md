# PROBE H — AFDB AF/non-AF, INTER-PATIENT (record-level 5-fold)
- Entire records held out per fold (no patient in both train/test)
- Model 4-4-8-8 FC 8->2 (622 params), 10s/2500@250Hz, oversample TRAIN balanced

## Pooled across 5 folds (n=83776)
- Accuracy: 0.8209
- **Macro-F1: 0.8201**
- AF F1: 0.8079
- **Sensitivity (AF recall): 0.9297**
- **Specificity: 0.7468**
- **ROC-AUC: 0.9052**

### Confusion (rows=true [non-AF, AF])
```
[[37218 12620]
 [ 2386 31552]]
```

### Per-fold
| fold | n | macroF1 | AUC |
|---|---|---|---|
| 0 | 18217 | 0.9323 | 0.9856 |
| 1 | 18368 | 0.7580 | 0.8419 |
| 2 | 18374 | 0.5704 | 0.8162 |
| 3 | 14344 | 0.9090 | 0.9634 |
| 4 | 14473 | 0.9361 | 0.9706 |

## vs intra-patient (PROBE F)
- Intra-patient (segment split): 0.9848 acc / 0.9843 macroF1
- **Inter-patient (this):       0.8209 acc / 0.8201 macroF1 / AUC 0.9052**
