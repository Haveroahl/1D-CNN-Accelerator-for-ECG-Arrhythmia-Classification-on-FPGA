# PROBE J - Train Chapman AF/non-AF -> zero-shot PTB-XL & AFDB
- Model 4-4-8-8 FC 8->2 (622 params), lead II train, 2500@250Hz, oversample TRAIN
- AF: Chapman=AFIB+AF, PTB-XL=AFIB|AFLT, AFDB=AFIB+AFL
- CAVEAT: AFDB uses ECG1 (Holter) vs lead II resting -> inherent shift

| target | %AF | acc | macroF1 | AF-F1 | sens | spec | AUC |
|---|---|---|---|---|---|---|---|
| Chapman (in-dist test) | 20.5% | 0.9324 | 0.9008 | 0.8448 | 0.899 | 0.941 | 0.9805 |
| PTB-XL (zero-shot) | 7.2% | 0.9170 | 0.7815 | 0.6094 | 0.899 | 0.918 | 0.9677 |
| AFDB (zero-shot) | 40.5% | 0.7686 | 0.7656 | 0.7391 | 0.809 | 0.741 | 0.8620 |

### Chapman (in-dist test)  (AF=218, non-AF=847)
```
[[797  50]
 [ 22 196]]
```

### PTB-XL (zero-shot)  (AF=1570, non-AF=20229)
```
[[18579  1650]
 [  159  1411]]
```

### AFDB (zero-shot)  (AF=33938, non-AF=49838)
```
[[36935 12903]
 [ 6483 27455]]
```
