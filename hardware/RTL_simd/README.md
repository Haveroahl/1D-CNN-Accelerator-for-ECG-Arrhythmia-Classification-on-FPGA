# SIMD-20 Position-Parallel Variant — RTL

A **second dataflow** for the ECG CNN accelerator, separate from production
(`hardware/RTL/` = channel-parallel 8-PE). Specified in [`/SIMD.md`](../../SIMD.md).
Material for Design-Space Exploration (area↔latency axis) in the Q3 paper.

- **Dataflow:** SIMD position-parallel — L=20 lanes each compute one consecutive
  output **position** of the same output channel; weight broadcast, data differs.
  (Production: channel-parallel — 8 PEs = 8 output channels, input broadcast.)
- **Status:** ✅ **bit-exact** (93384/93384 tol-0, 6 samples, every stage/channel),
  ✅ synthesized (Fmax 116.9 MHz, ALM 14 %), ✅ **latency-optimized 2755 cy (1.89× vs production)**
  — see *Latency* below.

## Modules

| File | Role |
|---|---|
| `simd_lane_array.v` | 20 lanes × 5-tap MAC + 20 accumulators + 20 requantize. Bit-exact copy of `cp_block.v` arithmetic (power-of-2 round-half-up, bias+round folded into acc init = production Fix B). Weight fan-out W1 (4 group copies; `W3_REPLICATE` param for 20-copy fallback). |
| `simd_pool.v` | MaxPool K5/S5 over the 20 synchronous lanes → 4 pooled values. Combinational 4-tree, 2-stage pipeline. |
| `line_buffer_engine.v` | 8 line-buffers (1/in-channel), depth L+4=24, shift-register. Static 5-tap window wiring (`slot[23-l-k]`) — no dynamic channel MUX. PP4/PP1 padding via `srw_rst` + `pad_zero`. |
| `pong_sram_wide.v` | Wide inter-layer feature memory, 32-bit word = 4 positions (SIMD.md §6 Plan B). 8 ch × 2 banks. Lets the 4-pooled/cycle pool result land in 1 write/cycle (Conv1 n=1 would otherwise overflow). |
| `simd_weight_rom.v` | Per-(oc,a) weight + bias, combinational. Hex layout identical to production. |
| `gap_fc_argmax_simd.v` | GAP reads the wide word (4 pos) and sums in one read; FC/Argmax identical to production. |
| `simd_controller.v` | FSM: block-outer / oc-inner loop. Per layer: PRIME first block → for each block { for each oc: blk_rst → sweep a → drain → write } → SLIDE. |
| `input_buffer.v` | 2500×8b input buffer at the wrapper (SIMD.md §3b Model 2); core streams 1 sample/cycle. |
| `ecg_core_simd.v` | Wires the datapath; source-data 1-cy alignment; pong write packing; read-addr MUX (Conv↔GAP). |
| `ecg_simd_top.v` | Thin wrapper: `avalon_slave` (reused verbatim) + `ecg_core_simd`. Port-identical to production top. |

## Bit-exact contract (SIMD.md §12 — identical to production)
`nb={8,6,6,7,0}`, `w_shift={6,6,6,7,8}`, `input_shift=2`, round-half-up
`(acc+2^(nb-1))>>nb` → clamp[-127,127] → ReLU(Conv4 only); bias+round folded into acc
init at a==0; GAP `floor(sum/4)`; FC nb=0. Weight hex byte/addr layout unchanged.

## Verification
Extended golden (`software/python/generate_golden_simd.py`) adds **pre-pool raw conv**
checkpoints (`after_conv1..4`, all channels) on top of the production post-pool ones —
required because a SIMD boundary/alignment bug can be hidden by MaxPool. 6 class-balanced
Chapman samples, tolerance 0 LSB.

```
# golden (from software/python, venv active)
python generate_golden_simd.py --checkpoint ./results/qat_int8/model_qat_int8.pth \
    --data_dir D:/Thesis101/data/Chapman --num_samples 6
# unit tests + full system (ModelSim/Questa)
cd hardware/fpga/simulation/questa_simd
vsim -c -do run_tb_lane.do     # 20-lane MAC vs after_conv4 (160/160)
vsim -c -do run_tb_pool.do     # pool vs after_pool4 (32/32)
vsim -c -do run_tb_block4.do   # lbuf+lane+pool block, Conv4
vsim -c -do run_tb_simd.do     # full inference, all stages/channels, tol-0 → 93384/93384
```

## Synthesis (5CSXFC6D6F31C6, `hardware/fpga/simd_synth/`)
| Metric | SIMD-20 | Production (ref) |
|---|---|---|
| DSP | **64 / 112 (57%)** | 28 (25%) |
| ALM | **5,948 / 41,910 (14%)** | 2,261 (5%) |
| M10K | 20 (4%) | ~11 |
| Registers | 7,780 | 3,196 |
| Fmax | **116.9 MHz** (timing PASS @100 MHz) | ~137.6 MHz |

Weight fan-out W1 (4 group copies) closed timing — `W3_REPLICATE` not needed.

**ALM by entity** (fit.rpt §17): `simd_lane_array` 3,108 (20-lane MAC, the legitimate
SIMD cost) · `line_buffer_engine` 1,991 (8×24 shift-reg) · `gap_fc` 422 · `controller`
253 · pool/wrom/pong/avs <350 total · `input_buffer` **100** (4 M10K banks).

> ⚠️ The wide-4 `input_buffer` initially stored 625×32-bit with a **byte-granular
> read-modify-write** write port — Quartus couldn't infer RAM and mapped all 20,000 bits
> to flip-flops → 10,799 ALM (64% of the design, 41% of the chip). Fix: split into **4
> byte banks** of 625×8b keyed by `pos%4`; each write touches one bank's full 8-bit word
> (RAM-inferable) and the read assembles the 32-bit wide word from all 4 banks. Same
> ports, same 1-cy timing, same byte layout → bit-exact unchanged. ALM 16,976 → **5,948**
> (−65%), +4 M10K, Fmax 104.85 → **116.9 MHz**.

## Latency — ✅ optimized (pipelined + wide-4 + tight settle)
Measured **2755 cycles** (deterministic, all 6 samples), bit-exact 93384/93384 tol-0.
**1.89× faster than production** (5216 cy). Three changes from the 13827-cy
correctness-first base:

1. **Wide-4 input/feature load** — `input_buffer.v` stores 625×32-bit words (4 INT8
   positions/word), the line-buffer loads 4 positions/cycle (`wide_load`). Cuts the
   per-block slide/prime from ~20-24 cy to ~7-8 cy for **all** layers (Conv1 input
   stream included — the earlier "~2500 input floor" only held for the 1-byte/cy path).
   This alone: 13827 → 11935 cy.
2. **Pipelined issue↔writeback** — `PH_SWEEP` issues (oc,a) continuously (pipeline
   stays full); results emerge ~10 cy later as `pooled_valid` IN ISSUE ORDER, and an
   independent writeback path (`wb_oc`/`wb_block` counters) replays the same order to
   the pong write address — no FIFO, no per-oc drain stall. `in_flight` gates the layer
   transition (`PH_TRANS`) until all in-flight results drain. 11935 → 3223 cy.
3. **Tight settle** — the post-load `PH_*_SETTLE` wait was trimmed 5→2 cy (the minimum
   that stays bit-exact; 1 cy fails — data needs 1 cy `shift_d` + 1 cy line-buffer write
   before taps are valid). 3223 → **2755 cy**.

Per-layer breakdown (measured): CONV1 1762, CONV2 662, CONV3 222, CONV4 86, GAP/FC 22.
Conv1 (IN_CH=1) still dominates (~64 %) but no longer from drain — it's the fixed
per-block overhead (load ~8 cy + settle 2 cy) × 125 blocks; useful compute is only
4 cy/block. The remaining lever is overlapping the next block's load into the previous
block's writeback drain (no second line-buffer needed) — left as the next step.

The SIMD.md §10 estimate (~1284 cy) is still optimistic (it omits per-layer pipeline
fill + slide), but the achieved 2755 cy realizes the variant's design goal: **lower
latency than production on the area↔latency axis** (at higher ALM/DSP — see below).

> ⚠️ tb capture note: `lane_valid` (lane-array S8) leads `pooled_valid` (after the
> 2-stage pool) by exactly 2 cy. The tb captures `lane_out` at `lane_valid` using its
> OWN replica counters (`cap_oc`/`cap_block`), NOT the controller's `wb_oc`/`wb_block`
> (those track `pooled_valid` and would be 2 results stale at `lane_valid`).

vs production (8-PE): production is smaller+faster-clock (ALM ~5 %, DSP 25 %, Fmax
~137 MHz) but higher latency (5216 cy). SIMD-20 trades area for latency — the intended
DSE data point for the Q3 paper (area↔latency axis). After the input_buffer M10K fix
the area cost is modest (ALM 14 %), so the trade is now favorable: ~1.9× lower latency
for ~2.6× the ALM and ~2.3× the DSP.
