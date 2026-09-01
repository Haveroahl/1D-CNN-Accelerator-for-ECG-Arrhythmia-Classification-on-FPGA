# PROBE D — MIT-BIH 5-class, WIDER conv 8-8-16-16, FC 16->5, clean split
- Window 256 samples, split FIRST 80/20 then oversample TRAIN ONLY to 59612/class
- Test keeps REAL imbalanced distribution
- Model 8-8-16-16 conv (2x wider), FC 16->5 single layer, float32, 30 ep

## Results
- **Test accuracy: 0.9831**
- **Macro-F1: 0.9521**
- Test dist: {'N': 14903, 'L': 1615, 'R': 1452, 'A': 510, 'V': 1381}

### Per-class F1
- N: 0.9898
- L: 0.9862
- R: 0.9904
- A: 0.8307
- V: 0.9636

### Confusion matrix (rows=true, cols=pred; order ['N', 'L', 'R', 'A', 'V'])
```
[[14652    25    11   154    61]
 [    5  1603     1     0     6]
 [    5     1  1444     2     0]
 [   23     1     5   476     5]
 [   17     6     3     4  1351]]
```

### sklearn report
```
              precision    recall  f1-score   support

           N       1.00      0.98      0.99     14903
           L       0.98      0.99      0.99      1615
           R       0.99      0.99      0.99      1452
           A       0.75      0.93      0.83       510
           V       0.95      0.98      0.96      1381

    accuracy                           0.98     19861
   macro avg       0.93      0.98      0.95     19861
weighted avg       0.98      0.98      0.98     19861

```

## Comparison (all clean-split, FC single layer unless noted)
- 4-4-8-8  FC 8->5 :   0.9742 acc / 0.9354 macro-F1 / A-F1 0.802
- 4-4-8-8  FC 8->16->5: 0.9715 acc / 0.9263 macro-F1 / A-F1 0.752
- 8-8-16-16 FC 16->5 (this): 0.9831 acc / 0.9521 macro-F1 / A-F1 0.831
- params this model: 2413
