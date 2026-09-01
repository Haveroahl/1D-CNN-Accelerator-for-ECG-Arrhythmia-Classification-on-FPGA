# PROBE C — MIT-BIH 5-class, 4-4-8-8 + FC HIDDEN (8->16->5), clean split
- Window 256 samples, split FIRST 80/20 then oversample TRAIN ONLY to 59612/class
- Test keeps REAL imbalanced distribution
- Model 4-4-8-8 conv + FC 8->16(ReLU)->5, float32, 30 ep

## Results
- **Test accuracy: 0.9715**
- **Macro-F1: 0.9263**
- Test dist: {'N': 14903, 'L': 1615, 'R': 1452, 'A': 510, 'V': 1381}

### Per-class F1
- N: 0.9822
- L: 0.9827
- R: 0.9759
- A: 0.7518
- V: 0.9387

### Confusion matrix (rows=true, cols=pred; order ['N', 'L', 'R', 'A', 'V'])
```
[[14449    32    47   251   124]
 [    8  1595     0     1    11]
 [    3     1  1438     6     4]
 [   37     1     4   465     3]
 [   22     2     6     4  1347]]
```

### sklearn report
```
              precision    recall  f1-score   support

           N       1.00      0.97      0.98     14903
           L       0.98      0.99      0.98      1615
           R       0.96      0.99      0.98      1452
           A       0.64      0.91      0.75       510
           V       0.90      0.98      0.94      1381

    accuracy                           0.97     19861
   macro avg       0.90      0.97      0.93     19861
weighted avg       0.98      0.97      0.97     19861

```

## Comparison (all clean-split unless noted)
- Leaky FC 8->5 (oversample-before-split): 0.9770 acc / 0.9771 macro-F1
- Clean FC 8->5 (no hidden):              0.9742 acc / 0.9354 macro-F1
- Clean FC 8->16->5 (this):               0.9715 acc / 0.9263 macro-F1
