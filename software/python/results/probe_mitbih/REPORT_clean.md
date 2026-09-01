# PROBE B (CLEAN, no leakage) — MIT-BIH 5-class beat morphology
- Window 256 samples, split FIRST 80/20 then oversample TRAIN ONLY to 59612/class
- Test keeps REAL imbalanced distribution
- Model 4-4-8-8 (same as deployed), FC 8->5, float32, 30 ep

## Results
- **Test accuracy: 0.9742**
- **Macro-F1: 0.9354**
- Test dist: {'N': 14903, 'L': 1615, 'R': 1452, 'A': 510, 'V': 1381}

### Per-class F1
- N: 0.9842
- L: 0.9802
- R: 0.9739
- A: 0.8020
- V: 0.9366

### Confusion matrix (rows=true, cols=pred; order ['N', 'L', 'R', 'A', 'V'])
```
[[14500    26    59   190   128]
 [   14  1588     1     1    11]
 [    3     1  1439     7     2]
 [   24     1     3   478     4]
 [   21     9     1     6  1344]]
```

### sklearn report
```
              precision    recall  f1-score   support

           N       1.00      0.97      0.98     14903
           L       0.98      0.98      0.98      1615
           R       0.96      0.99      0.97      1452
           A       0.70      0.94      0.80       510
           V       0.90      0.97      0.94      1381

    accuracy                           0.97     19861
   macro avg       0.91      0.97      0.94     19861
weighted avg       0.98      0.97      0.98     19861

```

## vs leaky probe
- Leaky (oversample-before-split): 0.9770 acc / 0.9771 macro-F1
- This (clean): 0.9742 acc / 0.9354 macro-F1
