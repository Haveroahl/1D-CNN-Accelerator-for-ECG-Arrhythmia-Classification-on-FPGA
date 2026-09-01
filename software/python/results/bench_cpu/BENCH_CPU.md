# CPU baseline vs FPGA accelerator — latency comparison

Measured 2026-07-30. Scope: three platforms running the **same bit-exact INT8
pipeline** — PyTorch on CPU, portable C on CPU, and the RTL accelerator.

Reproduce with:

```powershell
cd d:\Thesis101\software\python
..\..\.venv\Scripts\python.exe bench_sw_vs_hw.py `
    --checkpoint .\results\qat_int8\model_qat_int8.pth `
    --data_dir ..\..\data\Chapman --num_samples 500 --threads 1
```

The C row needs a working `gcc` on PATH (see "C toolchain" under Limitations).

## Platform

| Item | Value |
|---|---|
| CPU | Intel Core i7-11850H, 8C/16T, 2.50 GHz base |
| OS | Windows 11 Pro 26200 |
| Framework | PyTorch 2.12.0+cpu, Python 3.14.4 |
| C compiler | mingw-w64 GCC 14.2.0, `-O2` |
| Accelerator | Cyclone V 5CSXFC6D6F31C6 @ 100 MHz |
| Workload | 1 record = 2500 samples (batch = 1, matching the accelerator) |
| Arithmetic | **identical** bit-exact INT8 on all three rows |

## Table — latency / throughput

CPU rows: median over the same 500 Chapman test records, warm-up before timing,
compute-only (input already resident — the same boundary the accelerator's
5216-cycle figure is measured at). Single-threaded.

| Platform | Latency median | min | p95 | Throughput | vs accelerator |
|---|---:|---:|---:|---:|---|
| CPU — PyTorch INT8 (framework) | 458–470 µs | 443 | — | ~2,150 inf/s | ~8.8× slower |
| CPU — portable C `-O2` | 84–89 µs | 78.1 | ~96 | ~11,800 inf/s | **~1.65× slower** |
| **Cyclone V accelerator** | **52.16 µs** (5216 cy) | 52.16 (fixed) | 52.16 | 19,172 inf/s | **1×** |

Both C and PyTorch rows verified **bit-exact: 0/500 mismatches** against the
Python golden classes (`BIT-EXACT PASS`) — the same computation, not an
approximation. The C row is the spread over 5 independent runs
(84.0 / 84.2 / 84.8 / 86.5 / 88.7 µs).

### Why the C row is the reference, and what it represents

`hardware/fpga/sw/niosv/cnn_sw.c` (117 lines) is **portable scalar C written for
the Nios V/m RISC-V soft-core** — the same source file the Nios V firmware
compiles, so the software and firmware paths stay in sync. It uses plain nested
loops (`oc → p → j → ic → kk`) with no SIMD intrinsics, no `restrict`, no
threading, and a per-tap bounds check in the innermost loop.

It is therefore a baseline for the **embedded / soft-core device class** — the
class the wearable use case targets — compiled at `-O2`, the conventional
optimisation level for embedded builds. It is *not* a hand-tuned desktop kernel,
and this document does not present it as one; a baseline tuned for x86
specifically (AVX2 INT8 intrinsics, cache blocking, multi-threading) would be
faster, so the latency margin reported here should be read as applying to the
portable-C/embedded class rather than to desktop CPUs in general.

## Findings

1. **Against the same bit-exact algorithm in portable C, the accelerator is
   ~1.65× faster** (52.16 µs vs 84–89 µs) — while clocked at **100 MHz against a
   2.5 GHz core**. Normalised per clock cycle that is ~40× more work per cycle,
   which is the architecturally meaningful statement.

2. **The framework-level comparison inflates the hardware by ~5×.** The commonly
   used PyTorch/Keras-vs-FPGA method gives 8.8× here, but ~380 µs of PyTorch's
   458 µs is dispatch overhead on a 640-parameter network, not arithmetic. The C
   row, doing identical work single-threaded, is 5.4× faster than PyTorch.
   Corroboration: thread count barely moves the PyTorch row (458 → 456 → 525 µs
   for 1/4/8 threads — more threads is *worse*), impossible for compute-bound work.

3. **Determinism is a categorical win.** The accelerator is fixed at 5216 cycles
   for *every* input: p95 = median = max, spread 1.00×. The C row spreads
   78 → 222 µs (2.8×) and PyTorch 2.0–4.4×, from OS scheduling and cache effects.
   For continuous wearable monitoring the hard worst-case bound is what sizes the
   system, and only the accelerator has one.

4. **Energy is the accelerator's measured strength.** **42.02 µJ/inference** on
   DE10 (Cyclone V) and **12.84 µJ/inference** on the DE0-Nano Cyclone IV port
   (gate-level SDF, the higher-confidence run). No CPU-relative energy ratio is
   claimed — CPU package power could not be measured on this machine.

5. **Same arithmetic on all rows.** Unlike Liu 2023 (float Keras on CPU vs INT8 on
   FPGA), every row here runs the *same* bit-exact INT8 pipeline. The comparison
   therefore isolates the architecture and is not confounded with the quantization
   change — a direct payoff of the C2 bit-exact framework.

6. **Workload context.** Continuous single-lead ECG monitoring needs ~0.2
   inference/s; all three platforms exceed that by ~4–5 orders of magnitude. For
   the target application the binding constraints are energy per inference and
   worst-case latency bound, not median latency.

## Accelerator power (source: Quartus PowerPlay reports)

| | DE10-Standard (Cyclone V) | DE0-Nano (Cyclone IV E) |
|---|---:|---:|
| Device | 5CSXFC6D6F31C6 | EP4CE22F17C6 |
| **Total thermal power** | **805.53 mW** | **246.24 mW** |
| Core dynamic | 377.72 mW | 134.20 mW |
| Core static | 413.84 mW | 79.68 mW |
| I/O | 13.97 mW | 32.36 mW |
| **Energy / inference (total)** | **42.02 µJ** | **12.84 µJ** |
| Energy / inference (dynamic) | 19.70 µJ | 7.00 µJ |
| PowerPlay confidence | **Low** (22.4% toggle, 2.9% unknown) | Medium (86.9% toggle, 0.4% unknown) |
| VCD source | RTL simulation | **gate-level, back-annotated SDF** |

Energy = power × 52.16 µs (5216 cycles, identical and bit-exact on both ports).
Sources: `hardware/fpga/output_files/ecg_accelerator_top.pow.rpt`,
`hardware/fpga_de0/output_files/ecg_de0_100.pow.rpt`.

**Quote the DE0-Nano figure as the primary energy result** — it is the better
measurement (gate-level SDF, 86.9% toggle coverage vs RTL VCD at 22.4%). Report
the DE10 number with its `Low`-confidence caveat attached.

**Static power dominates on DE10** (413.84 of 805.53 mW = 51%): the Cyclone V SoC
die carries a hard ARM Cortex-A9 subsystem that leaks whether used or not, while
the small Cyclone IV E die leaks 79.68 mW — 81% less for the *same* RTL. For
continuous low-duty-cycle monitoring, choosing the right device matters more than
optimising the logic.

## Comparison with Liu 2023 (same reference method)

| | Liu 2023 | This work |
|---|---|---|
| FPGA | Cyclone V @ 50 MHz, 66 µs | Cyclone V @ 100 MHz, 52.16 µs |
| CPU baseline | Keras float, i7-8700 → 553 µs | PyTorch INT8, i7-11850H → 458 µs |
| Speedup vs PC | 8.38× | 8.8× (framework) / **1.65× (portable C)** |
| Baseline arithmetic | float (≠ FPGA) | INT8 bit-exact (= FPGA) |
| Compiled-C baseline | not reported for the CNN | **reported** |

Our framework-level number reproduces Liu's almost exactly (8.8× vs 8.38×),
cross-validating both measurements. The difference is that we also report a
compiled-C row, which shows the framework-level method — standard in this
literature — inflates the hardware advantage by ~5× on this workload. Liu's
8.38× is likely subject to the same inflation, since a Keras baseline on a 1-D
CNN of this size is dominated by the same dispatch overhead.

## Limitations

- **Compiler optimisation level.** The C row is `-O2`. Higher levels enable
  GCC's auto-vectorizer on this kernel and reduce CPU latency substantially, so
  the 1.65× margin is specific to `-O2` / embedded-class builds and does not hold
  against an aggressively optimised desktop build. State the flag with the number.
- **Single-threaded C.** Comparing 1 CPU thread against 8 PEs favours the
  accelerator. Not measured multi-threaded because at 2500 samples the
  per-inference work is too small to amortise thread dispatch — the PyTorch rows
  demonstrate exactly this.
- **No ARM / soft-core baseline yet.** No Cortex-A board available, so Liu's
  Raspberry-Pi row has no counterpart. `main.c:112-132` already instruments
  `cnn_sw.c` on Nios V/m with the RISC-V `mcycle` counter, but the on-board
  figure has not been captured. That would give a same-fabric, same-clock
  accelerator-vs-soft-core comparison — the strongest available latency claim in
  the correct device class, and the highest-value remaining measurement.
- **CPU power not measured.** No HWiNFO/LibreHardwareMonitor install, no battery
  to derive discharge from, and RAPL needs a kernel-mode driver. Nameplate TDP is
  a thermal design figure, not a measurement of this workload, so no CPU energy
  number is reported.
- **DE10 PowerPlay confidence is `Low`.** Only 22.4% of signals had toggle rates
  from simulation and 2.9% were unknown, so 805.53 mW carries real uncertainty.
  The DE0-Nano run is the defensible figure.
- **Earlier notes quoted 623 mW / 198 mW / 32.5 µJ with "95.6% toggle" for DE10.**
  Those values are **not in the report** and were carried from memory;
  `PAPER_DATA.md` had flagged the row `NEEDS-SOURCE`. The report says
  805.53 / 377.72 mW at `Low` confidence (22.4% toggle). Use the report values.
- **C toolchain.** The Quartus-bundled mingw
  (`questa_fse/gcc-7.4.0-.../gcc.exe`) has a non-functional `cc1.exe` stub: it
  exists and exits 0 but compiles nothing. `find_gcc()` now probe-compiles a test
  program rather than trusting file existence, and prefers PATH. Install
  mingw-w64 and put it on PATH.
- **Windows Application Control** blocked Python from spawning the freshly built
  `bench_c.exe` (`WinError 4551`), so the C row was run directly from the shell on
  the exact `inp.bin`/`pred.bin` the script emitted; the bit-exact check still
  passed. On a locked-down machine the automated path needs the build directory
  allow-listed.
