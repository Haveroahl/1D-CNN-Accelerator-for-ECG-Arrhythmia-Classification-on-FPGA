# PROBE E — MIT-BIH 5-class, window 250, pool 5/5/5, 3 CONV (L=10 SIMD-friendly)
- Window 250, split-first 80/20, oversample TRAIN to 59613/class, FC single layer, 30 ep
- Test dist: {'N': 14904, 'L': 1615, 'R': 1452, 'A': 510, 'V': 1381}

## Config narrow (4,8,16) — params 933
- **Acc 0.9807 / Macro-F1 0.9497**
- Per-class F1: N=0.989, L=0.986, R=0.985, A=0.848, V=0.941
- Confusion (rows=true ['N', 'L', 'R', 'A', 'V']):
```
[[14631    23    19   120   111]
 [    6  1600     0     1     8]
 [    6     1  1436     7     2]
 [   29     0     4   470     7]
 [   26     7     5     1  1342]]
```

## Config wide (8,16,16) — params 2085
- **Acc 0.9790 / Macro-F1 0.9414**
- Per-class F1: N=0.987, L=0.990, R=0.990, A=0.776, V=0.964
- Confusion (rows=true ['N', 'L', 'R', 'A', 'V']):
```
[[14560    19    13   239    73]
 [    6  1607     1     0     1]
 [    5     0  1441     4     2]
 [   25     0     4   479     2]
 [   16     4     0     3  1358]]
```

## Comparison vs 4-conv (256-window) probes — A-F1 is the key metric
| config | conv | window | pool | acc | macroF1 | A-F1 | SIMD L |
|---|---|---|---|---|---|---|---|
| 4-4-8-8 | 4 | 256 | 5/5/2/2 | 0.9742 | 0.9354 | 0.802 | none (mixed) |
| 8-8-16-16 | 4 | 256 | 5/5/2/2 | 0.9866 | 0.9619 | 0.874 | none (mixed) |
| (4, 8, 16) | 3 | 250 | 5/5/5 | 0.9738 | 0.9322 | 0.770 | **10** |
| (8, 16, 16) | 3 | 250 | 5/5/5 | 0.9708 | 0.9205 | 0.687 | **10** |
