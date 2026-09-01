# PROBE G — Cross-dataset zero-shot AF: AFDB -> PTB-XL
- Train: AFDB 10s/2500@250Hz, AF=AFIB+AFL vs non-AF=N+J, lead ECG1, balanced
- Test : PTB-XL zero-shot, lead II, 500->250Hz decimate, first 10s; AF=AFIB|AFLT
- Model 4-4-8-8 FC 8->2 (622 params); imbalanced test -> report F1/Sens/Spec/AUC
- NOTE: Lead mismatch (AFDB ECG1 vs PTB-XL II) + Holter vs resting = inherent shift

## PTB-XL test set
- AF=1570 / non-AF=20229 (7.2% AF)

## Zero-shot results
- Accuracy: 0.9152  (misleading under 7.2% imbalance)
- **Macro-F1: 0.7674**
- **AF F1: 0.5820**
- **Sensitivity (AF recall): 0.8197**
- **Specificity: 0.9226**
- **ROC-AUC: 0.9512**

### Confusion (rows=true [non-AF, AF], cols=pred)
```
[[18663  1566]
 [  283  1287]]
```

### sklearn report
```
              precision    recall  f1-score   support

      non-AF       0.99      0.92      0.95     20229
          AF       0.45      0.82      0.58      1570

    accuracy                           0.92     21799
   macro avg       0.72      0.87      0.77     21799
weighted avg       0.95      0.92      0.93     21799

```
