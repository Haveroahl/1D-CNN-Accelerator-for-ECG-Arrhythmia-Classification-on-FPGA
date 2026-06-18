# Elastic-Pareto (capacity-isolation): one bitstream, consistent from-scratch recipe

All points run **bit-exact on ONE bitstream** (tb_topo_sweep 5/5 PASS); INT8 = power-of-2 QAT on Chapman test. Latency MEASURED; energy = P×latency (Phase-C PowerPlay, baseline power held constant — conservative for small topologies).

| Topology | Float acc | INT8 acc | INT8 F1 | Conv w | Latency (µs) | E_total (µJ) | E_dyn (µJ) |
|---|---|---|---|---|---|---|---|
| (2, 2, 2, 2) | 0.7840 | 0.7850 | 0.7346 | 70 | 38.44 | 23.95 | 7.61 |
| (2, 2, 4, 4) | 0.8779 | 0.8817 | 0.8649 | 150 | 38.94 | 24.26 | 7.71 |
| (4, 4, 4, 4) | 0.9221 | 0.9108 | 0.9005 | 260 | 51.14 | 31.86 | 10.13 |
| (4, 4, 8, 8) † | 0.9192 | 0.9183 | 0.9099 | 580 | 52.14 | 32.48 | 10.32 |
| (8, 8, 8, 8) | 0.9239 | 0.9286 | 0.9211 | 1000 | 76.54 | 47.68 | 15.15 |

All points use ONE consistent **from-scratch** recipe (50 float + 30 QAT ep, seed 42) so accuracy reflects capacity, not training effort. Frontier monotone in capacity; (8,8,8,8) is NOT worse than (4,4,8,8) at equal recipe. INT8 ≈ float at every point.

† **Production-recipe lever** (orthogonal to capacity): the shipped (4,4,8,8) (train-full → prune-transfer → finetune + tuned QAT) reaches **94.65% INT8** vs 91.83% from-scratch — +2.8pp available at any operating point, NOT a capacity effect.
