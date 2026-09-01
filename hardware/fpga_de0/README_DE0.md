# DE0-Nano port — ECG accelerator (low-power measurement)

Retarget of the production `ecg_accelerator_top` core from the DE10-Standard
(Cyclone V `5CSXFC6D6F31C6`) to the **DE0-Nano** (Cyclone IV E `EP4CE22F17C6`).

## Why this exists

The DE10-Standard SoC die dissipates **~412 mW static** (large 28 nm die + an
unused hard ARM core) — that leakage dominates total power and is a *device*
property, not a property of this small design. The DE0-Nano's 60 nm Cyclone IV E
idles far lower, so its total power reflects the accelerator itself. This project
exists to obtain an **honest low-power number** for the thesis.

The functional on-board demo is already covered on the DE10 (JTAG, 94.27%); this
port targets **resource + Fmax + power**, not a second functional demo.

## What was changed vs the Cyclone V project

| Item | Cyclone V (`../fpga/`) | DE0-Nano (here) |
|---|---|---|
| Device | 5CSXFC6D6F31C6 | EP4CE22F17C6 |
| Clock | PIN_AF14 (FPGA_CLK1_50), 100 MHz SDC | PIN_R8 (CLOCK_50), 50 MHz SDC |
| Block RAM | M10K (ping_pong, w_ram) | **M9K** (forked RTL) |
| Bias `b_store` | MLAB | LEs (hint dropped) |
| Avalon bus pins | real (measurement top) | **virtual pins** (83→ only clk+rst_n real) |

### Forked RTL (the only RTL difference)
Two memory modules are forked into `rtl_de0/` because Cyclone IV E has no
M10K/MLAB primitive:
- `rtl_de0/ping_pong_sram.v` — ramstyle `"M10K"` → `"M9K"` (16 banks ×512×8)
- `rtl_de0/cp_engine.v` — `w_ram0..7` `"M10K"` → `"M9K"`; `b_store` MLAB hint dropped

Everything else is **shared unchanged** from `../RTL/` (listed in
`ecg_de0_common.qsf`). If you edit the shared RTL, only re-sync these two forks
if the originals' memory declarations change.

## Results so far (compile 2026-06-20, EP4CE22F17C6, slow 1200mV 85C)

Two revisions:
- **`ecg_de0_top`** — demo config, 50 MHz SDC (board CLOCK_50). Setup slack **+7.14 ns**.
- **`ecg_de0_100`** — timing-feasibility, 100 MHz SDC. Setup slack **+0.44 ns → PASS**.

| Metric | DE10 (Cyclone V) | DE0-Nano (Cyclone IV E) |
|---|---:|---:|
| Logic | 2,973 ALM (7%) | **8,035 LE (36%)** *(ALM≠LE)* |
| Block RAM | 28 M10K | **456 M9K seg, 95,776 bit (16%)** ✅ in block RAM |
| Multipliers | 28 DSP | **44 / 132 (33%)** |
| Fmax @85C (fit-for-100) | 104.85 MHz | **104.6 MHz** |
| **Total power** | 598 mW | **243 mW (−59%)** |
| └ Core dynamic | 171 mW | **130 mW** |
| └ Core static | **412 mW** | **80 mW (−81%)** |
| └ I/O | 14 mW | 33 mW |

The big win is **static power: 412 → 80 mW**. The DE10's huge SoC die (+ unused
hard ARM core) leaks ~5× more than this small Cyclone IV E. Power above used the
RTL `tb_top.vcd` → **confidence "Low"** (covers registers, combinational logic is
vectorless). The gate-level VCD flow (below) targets a HIGH-confidence number.

## Files
- `ecg_de0_top.qpf` / `.qsf` — the project (top = `ecg_accelerator_top`)
- `ecg_de0_common.qsf` — shared device/RTL/HEX list (sourced, never opened)
- `ecg_de0_50mhz.sdc` — 50 MHz constraint (DE0-Nano CLOCK_50)
- `rtl_de0/` — the two M9K-retargeted modules

## How to build (from `hardware/fpga_de0/`)

```sh
QPATH="D:/altera_lite/25.1std/quartus/bin64"

# 1. Analysis & Synthesis (fast — checks device fit + RAM inference)
"$QPATH/quartus_map.exe" ecg_de0_top

# 2. Full compile (fit + timing + assembler)
"$QPATH/quartus_sh.exe" --flow compile ecg_de0_top

# 3. Power (needs a real-activity VCD — see below), then re-run:
"$QPATH/quartus_pow.exe" ecg_de0_top
```

Reports land in `output_files/`:
- `ecg_de0_top.map.summary` / `.fit.summary` — resource (LE, M9K, multipliers)
- `ecg_de0_top.sta.rpt` — Fmax @ 50 MHz
- `ecg_de0_top.pow.summary` — power

## ⚠️ Verify after building
1. **RAM in M9K, not logic.** Check `.fit.summary` → "Total memory bits" and
   "Total RAM Blocks" are non-zero and the 16+8+1 SRAMs are M9K. If RAM fell to
   LEs, the LE count explodes and the hint didn't take.
2. **DSP/multipliers fit.** EP4CE22 has 132 embedded 9×9 multipliers. The INT8
   MACs (8 PE × 5 taps = 40 mults) should fit comfortably; confirm none spilled
   to logic.
3. **Timing @ 50 MHz closes** (slack ≥ 0). Cyclone IV E -6 is slower than
   Cyclone V -6; 50 MHz should still close with wide margin.

## Power: do NOT cite the default-toggle number
A PowerPlay run without an activity file reports **confidence "Low"** (default
12.5% toggle) — unusable for the thesis. Feed a VCD from a full inference:
- Reuse `../fpga/simulation/questa/tb_top.vcd` (already referenced in the .qsf),
  or regenerate it by simulating `tb_top.v` with VCD dump enabled.
- After a VCD is read, confidence rises to "High" and the **dynamic** figure is
  the design's real per-inference power.
- Report **dynamic** (and energy = P_dyn × latency); the lower static here makes
  the *total* defensible too, unlike on the DE10.
