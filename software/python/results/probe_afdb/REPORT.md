# PROBE F — AFDB AF vs non-AF, 10s segments (2500@250Hz), 4-4-8-8 FC 8->2
- Segment 2500 samples (= Chapman input len), lead ECG1, clean split, oversample TRAIN to 39870/class
- AF = AFIB+AFL ; non-AF = N+J ; boundary-straddling segments dropped
- params 622, 40 ep

## Results
- **Accuracy 0.9848 / Macro-F1 0.9843**
- Test: AF=6788 non-AF=9968
- Per-class F1: non-AF=0.9871, AF=0.9814

### Confusion (rows=true [non-AF, AF], cols=pred)
```
[[9778  190]
 [  65 6723]]
```

### sklearn report
```
              precision    recall  f1-score   support

      non-AF       0.99      0.98      0.99      9968
          AF       0.97      0.99      0.98      6788

    accuracy                           0.98     16756
   macro avg       0.98      0.99      0.98     16756
weighted avg       0.98      0.98      0.98     16756

```
