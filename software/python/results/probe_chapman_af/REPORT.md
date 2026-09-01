# PROBE I - Chapman AF vs non-AF (2-class), 4-4-8-8 FC 8->2
- AF = AFIB+AF (orig label 0); non-AF = GSVT/SB/SR; patient-independent 80/10/10
- Lead II, 250Hz, 2500 samples, 622 params, oversample TRAIN balanced, 40 ep

## Test set
- AF=218 / non-AF=847 (20.5% AF)

## Results
- Accuracy: 0.9324
- **Macro-F1: 0.9008**
- AF F1: 0.8448
- **Sensitivity (AF recall): 0.8991**
- **Specificity: 0.9410**
- **ROC-AUC: 0.9805**

### Confusion (rows=true [non-AF, AF])
```
[[797  50]
 [ 22 196]]
```

### sklearn report
```
              precision    recall  f1-score   support

      non-AF       0.97      0.94      0.96       847
          AF       0.80      0.90      0.84       218

    accuracy                           0.93      1065
   macro avg       0.88      0.92      0.90      1065
weighted avg       0.94      0.93      0.93      1065

```

## AF detection across datasets (same 8-PE arch, 622 params, input 2500)
| dataset | setup | macroF1 | AUC | sens | spec |
|---|---|---|---|---|---|
| AFDB | inter-patient 5fold | 0.8201 | 0.9052 | 0.930 | 0.747 |
| AFDB->PTB-XL | zero-shot | 0.7674 | 0.9512 | 0.820 | 0.923 |
| Chapman | patient-indep | 0.9008 | 0.9805 | 0.899 | 0.941 |
