# CP Submodule Cycle Timing — cp_mac / cp_accumulate_rescale / cp_pool

Cycle-accurate timing for the three submodules that `cp_block.v` was split into
(bit-exact structural split, no logic change):

```
cp_block = cp_mac (S1→S4) → cp_accumulate_rescale (S5→S8) → cp_pool (S9)
```

All values below are the **actual signed register values** from a Questa RTL
simulation on `ecg_sample0.hex`, captured by `testbench/tb_cpb_cycle_probe.v`
(channel 0). The combined cp_block view is in [cp_pipeline.md](cp_pipeline.md) and
[cp_block_datapath_tables.tex](cp_block_datapath_tables.tex); this file breaks the
timing out **per submodule**.

> Raw evidence (bit-exact, same run):
> [cp_block_conv1_probe_raw.txt](cp_block_conv1_probe_raw.txt) (Conv1, in_ch=1),
> [cp_block_conv4_probe_raw.txt](cp_block_conv4_probe_raw.txt) (Conv4, in_ch=8).
> Column order in the raw logs: `... mux_s1 a5 ce5 | tree_out acc accf av | shft clmp relu rv | pw pout`.

---

## 1. cp_mac (S1→S4) — MAC datapath

Pure feed-forward, **no control/valid** — a fixed 4-cycle pipeline. Dot-product of
5 taps × 5 weights. Widths grow by sign-extension 16→17→18→20 so summing 5 products
never overflows.

```
Latency: x_in / w  →  tree_out  =  4 cycles
```

| stage | register(s)              | combinational input                                              | width |
|-------|--------------------------|------------------------------------------------------------------|-------|
| S1    | `prod0..prod4`           | `prodk <= $signed(x_in[k*8+:8]) * $signed(w[k*8+:8])`            | 16b   |
| S2    | `sum01`,`sum23`,`p4_d1`  | `sum01<={prod0[15],prod0}+{prod1[15],prod1}` ; `sum23` sym. ; `p4_d1<=prod4` | 17b / 16b |
| S3    | `sum0123`,`p4_d2`        | `sum0123<={sum01[16],sum01}+{sum23[16],sum23}` ; `p4_d2<=p4_d1`  | 18b / 16b |
| S4    | `tree_out`               | `tree_out<={{2{sum0123[17]}},sum0123}+{{4{p4_d2[15]}},p4_d2}`    | 20b   |

`prod4` bypasses the 2-deep add tree of lanes 0–3 via `p4_d1`/`p4_d2` so it lands at
S4 aligned with `sum0123`.

**Real values (Conv4, `cp_block_conv4_probe_raw.txt`).** The `mux_s1` column shows the
packed 5-tap window arriving each cycle; `tree_out` is that window's MAC 4 cycles
later:

| cyc | mux_s1 (packed 5×8b) | → tree_out (4 cyc later) |
|-----|----------------------|--------------------------|
| 41  | `0b0c100000`         | 865  @cyc 45             |
| 42  | `141d1e0000`         | 1311 @cyc 46             |
| 43  | `2324fc0000`         | 694  @cyc 47             |
| 44  | `f702070000`         | 207  @cyc 48             |

Every cycle produces one `tree_out` — cp_mac itself has no stall; the per-output-cadence
comes entirely from the accumulator downstream.

---

## 2. cp_accumulate_rescale (S5→S8) — accumulate + bias + rescale + ReLU

Control-dependent. `a_in` (=`a_d5`, channel counter delayed 5 cycles) and
`compute_en_in` (=`ce_d5`) decide when the accumulator initialises and finalises.
Latency from a given `tree_out` to `relu_out` is fixed once the channel is the last
one, but the **spacing between outputs = `in_ch` cycles** (Conv1=1, Conv2/3=4, Conv4=8).

Combinational:
```verilog
out_valid = compute_en_in && (a_in == in_ch - 1);
tree_sext = {{12{tree_out[19]}}, tree_out};              // 20 → 32b
round_add = (nb > 0) ? (32'sd1 << (nb-1)) : 0;           // round-half-up, folded
```

| stage  | register(s)                 | behavior                                                                 |
|--------|-----------------------------|--------------------------------------------------------------------------|
| S5     | `acc`                       | `a_in==0` → `acc <= tree_sext + bias_in + round_add` (**init: bias+round folded in**); else `acc <= acc + tree_sext` |
| S5b    | `acc_final_r`,`acc_final_v` | `acc_final_v <= out_valid`; on `out_valid`, `acc_final_r` latches the full sum for **exactly 1 cycle** (adds the last channel's `tree_sext`, breaking the 2-adder critical path) |
| S_bias | `biased`,`bias_valid`       | **pure passthrough** (`biased <= acc_final_r`) — bias already folded; kept only for pipeline depth / valid timing |
| S6     | `shifted`,`rescale_v1`      | **pure arithmetic shift** `shifted <= biased >>> nb` (round already added at S5) |
| S7     | `clamped`,`rescale_v2`      | `clamped <= (shifted>127)?127 : (shifted<-127)?-127 : shifted[7:0]`      |
| S8     | `relu_out`,`relu_v`         | `relu_out <= (relu_en && clamped[7]) ? 0 : clamped` (ReLU active Conv4 only) |

**Valid chain (5 register hops):**
```
out_valid → acc_final_v → bias_valid → rescale_v1 → rescale_v2 → relu_v
```
so `relu_out` is valid **5 cycles after `out_valid`**.

> ⚠️ S6 is a *pure* `>>> nb` — the `+ round_add` is **folded into the S5/S5b acc-init
> term** (`a_in==0`). Numerically identical to `(acc + bias + round) >>> nb`
> (round-half-up, signed), but off the S6 critical path. Older `cp_pipeline.md`
> snippets that show `(biased + round_add) >>> nb` at S6 predate this fold.

### 2a. Conv4 (in_ch=8) — 8 channels accumulate → 1 output every 8 cycles

From `cp_block_conv4_probe_raw.txt`, one complete output (`a_d5` sweeps 0→7):

| cyc | a_d5 | tree_out | acc      | accf | av | shft | clmp | relu | rv |
|-----|------|----------|----------|------|----|------|------|------|----|
| 45  | 0    | 865      | 865 (init)| 0   | 0  |      |      |      |    |
| 46  | 1    | 1311     | 943      | 0    | 0  |      |      |      |    |
| 47  | 2    | 694      | 2254     | 0    | 0  |      |      |      |    |
| 48  | 3    | 207      | 2948     | 0    | 0  |      |      |      |    |
| 49  | 4    | 380      | 3155     | 0    | 0  |      |      |      |    |
| 50  | 5    | 1252     | 3535     | 0    | 0  |      |      |      |    |
| 51  | 6    | 823      | 4787     | 0    | 0  |      |      |      |    |
| 52  | 7    | 546      | 5610     | 0    | 0  |      |      |      |    |
| 53  | 0    | 1352     | **6156** | **6156** | **1** |  |  |  |    |
| 54  | 1    | 1861     | 1430     | 6156 | 0  |      |      |      |    |
| 55  | 2    | 852      | 3291     | 6156 | 0  | **48** |    |      |    |
| 56  | 3    | −231     | 4143     | 6156 | 0  | 48   | **48** |    |    |
| 57  | 4    | 42       | 3912     | 6156 | 0  | 48   | 48   | **48** | **1** |

- `a_d5=7` (cyc 52) is `in_ch-1` → `out_valid` → `acc_final_r=6156`, `av=1` one cycle
  later at cyc 53. Meanwhile `acc` at cyc 53 already restarts the *next* output (init).
- **Rescale:** `nb=7` (Conv4). `6156 >>> 7 = 48` — arithmetic shift drops the low 7 bits
  (floor); the round-half-up term was already added into `acc`, so this is one shift, no
  add. (`6156/128 = 48.09` decimal, but the shift yields the integer 48 directly.)
- `relu_out=48` at cyc 57 = 5 cycles after `out_valid` (cyc 52). Valid `rv` asserts once
  per 8 cycles — 8× sparser than Conv1.

### 2b. Conv1 (in_ch=1) — init and finalize every cycle → 1 output/cycle

From `cp_block_conv1_probe_raw.txt`. Because `in_ch=1`, `a_in==0` **every** cycle, so the
accumulator both initialises and finalises in the same cycle (`out_valid` is high every
cycle once the pipeline is primed):

| cyc | tree_out | acc / accf | av | shft | clmp | relu | rv |
|-----|----------|-----------|----|------|------|------|----|
| 10  | 6        | 0         | 0  |      |      |      |    |
| 11  | −158     | 0         | 1  |      |      |      |    |
| 12  | −191     | **−98**   | 1  |      |      |      |    |
| 13  | −353     | −131      | 1  |      |      |      |    |
| 14  | −429     | −293      | 1  | −1   |      |      |    |
| 15  | −391     | −369      | 1  | −1   | −1   |      |    |
| 16  | −301     | −331      | 1  | −2   | −1   | **−1** | 1 |

`nb=8` (Conv1). Values are negative (Conv1 has no ReLU), so after clamp they pass through
S8 unchanged. One `relu_out` every cycle → the accumulator is never a bottleneck for
Conv1.

---

## 3. cp_pool (S9) — MaxPool rolling comparator

Rolling max over a window of 5 valid samples. Gated by `relu_v && compute_en_in` so junk
from the SRW priming phase is never counted.

```verilog
if (rst || pool_rst) begin pool_cnt<=0; max_reg<=0; pool_write_r<=0; end
else if (relu_v && compute_en_in) begin
    if (pool_cnt==0)          max_reg <= relu_out;   // load first sample of window
    else if (relu_out>max_reg) max_reg <= relu_out;  // rolling max
    if (pool_cnt==4) begin pool_cnt<=0; pool_write_r<=1; end  // emit on 5th
    else               pool_cnt <= pool_cnt + 1;
end
pool_out  = max_reg;      // running max
pool_write = pool_write_r; // strobe, AND cp_en externally → pong_we[oc]
```

`pool_write` strobes for 1 cycle on every **5th** `relu_v`, then `pool_cnt` resets.
Latency: 1 cycle/hit; one `pool_write` per 5 accepted samples.

**Conv1 (`cp_block_conv1_probe_raw.txt`).** `relu_v` is high every cycle from cyc 16, so
`pool_write` (`pw`) fires every 5 cycles — cyc 21, 26, 31, 36, 41, 46, 51, 56 — with
`pool_out` (`pout`) `= −1`:

| cyc | relu | rv | pw | pout |
|-----|------|----|----|------|
| 17  | −1   | 1  | 0  | −1   |
| 21  | −1   | 1  | **1** | −1 |
| 26  | −1   | 1  | **1** | −1 |
| 31  | −1   | 1  | **1** | −1 |

**Conv4.** Each output arrives every 8 cycles, so a `pool_write` lands every 40 cycles —
outside the cyc 45–60 window in the raw log. Within that window only the first pooled
value propagates: `pool_out=48` appears from cyc 58 (`relu_out=48` at cyc 57 → registered
into `max_reg`).

---

## Cross-submodule cadence summary

| layer | in_ch | nb | cp_mac latency | acc spacing (1 output every) | pool_write every |
|-------|-------|----|----------------|-------------------------------|-------------------|
| Conv1 | 1     | 8  | 4 cy           | 1 cy                          | 5 cy              |
| Conv2 | 4     | 6  | 4 cy           | 4 cy                          | 20 cy             |
| Conv3 | 4     | 6  | 4 cy           | 4 cy                          | 20 cy             |
| Conv4 | 8     | 7  | 4 cy           | 8 cy                          | 40 cy             |

Only `a_in`/`in_ch`/`nb` (from `cnn_controller`) differ between layers — the datapath is
identical. cp_mac is always 4 cycles; the accumulator's init/finalize spacing is what
sets the per-layer throughput.
