# PROBE — MIT-BIH 5-class beat morphology (N,L,R,A,V)
- Window: 256 samples (+/-128 around R-peak), 360 Hz
- Split: intra-patient random 80/20 (probe only)
- Imbalance: random oversample TRAIN ONLY to 74515/class
- Model: 4-4-8-8 conv (same as deployed), FC 8->5, float32, 30 ep

## Results
- **Test accuracy: 0.9770**
- **Macro-F1: 0.9771**
- Test-set class distribution: {'A': 15072, 'L': 14681, 'N': 14928, 'R': 14973, 'V': 14861}

### Per-class F1
- N: 0.9555
- L: 0.9921
- R: 0.9918
- A: 0.9642
- V: 0.9818

### Confusion matrix (rows=true, cols=pred; order ['N', 'L', 'R', 'A', 'V'])
```
[[14199    30     7   548   144]
 [   67 14531     1    15    67]
 [   84     7 14766    93    23]
 [  296     0    28 14738    10]
 [  147    44     0   104 14566]]
```

### sklearn report
```
              precision    recall  f1-score   support

           N       0.96      0.95      0.96     14928
           L       0.99      0.99      0.99     14681
           R       1.00      0.99      0.99     14973
           A       0.95      0.98      0.96     15072
           V       0.98      0.98      0.98     14861

    accuracy                           0.98     74515
   macro avg       0.98      0.98      0.98     74515
weighted avg       0.98      0.98      0.98     74515

```
