# Elastic-Pareto (deployment): one bitstream, best-deployable per operating point

All points run **bit-exact on ONE bitstream** (tb_topo_sweep 5/5 PASS); INT8 = power-of-2 QAT on Chapman test. Latency MEASURED; energy = P×latency (Phase-C PowerPlay, baseline power held constant — conservative for small topologies).

| Topology | INT8 acc | INT8 F1 | Conv w | Latency (µs) | E_total (µJ) | E_dyn (µJ) | Method |
|---|---|---|---|---|---|---|---|
| (2, 2, 2, 2) | 0.8075 | 0.7647 | 70 | 38.44 | 23.95 | 7.61 | prune-transfer |
| (2, 2, 4, 4) | 0.9146 | 0.9042 | 150 | 38.94 | 24.26 | 7.71 | prune-transfer |
| (4, 4, 4, 4) | 0.9277 | 0.9189 | 260 | 51.14 | 31.86 | 10.13 | prune-transfer |
| (4, 4, 8, 8) | 0.9465 | 0.9396 | 580 | 52.14 | 32.48 | 10.32 | production-anchor |
| (8, 8, 8, 8) | 0.9286 | 0.9211 | 1000 | 76.54 | 47.68 | 15.15 | from-scratch (wider than parent) |

Each point = **best-deployable** model at that topology via the same prune-transfer recipe as the shipped (4,4,8,8)=94.65%. (8,8,8,8) is from-scratch (conv1=8 exceeds the (4,8,8,16) parent → not prunable). INT8 ≈ float at every point (power-of-2 0-DSP rescale generalises). (8,8,8,8) costs 2× weights + 47% latency/energy for ≤(4,4,8,8) accuracy → (4,4,8,8) is the knee; (2,2,*) are low-power screening points (38 µs, ~24 µJ, −26% energy).
