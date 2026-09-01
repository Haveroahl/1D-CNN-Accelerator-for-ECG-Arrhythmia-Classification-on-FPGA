# Reset Safety — `cp_mac` Has No Reset (and Why That Is Correct)

> Scope: explains why the MAC datapath (`cp_mac.v`, S1–S4) carries no reset, and
> proves that any mid-pipeline `rst` followed by a fresh `start` still produces a
> bit-exact result. Companion to [cp_pipeline.md](cp_pipeline.md).

## Premise

`rst` is **synchronous, active-high**. It clears every register that holds
**accumulated / feedback state**:

| Stage | Module | Registers cleared by `rst \|\| pool_rst` |
|---|---|---|
| S5  | `cp_accumulate_rescale` | `acc` |
| S5b | `cp_accumulate_rescale` | `acc_final_r`, `acc_final_v` |
| S_bias | `cp_accumulate_rescale` | `biased`, `bias_valid` |
| S6  | `cp_accumulate_rescale` | `shifted`, `rescale_v1` |
| S7  | `cp_accumulate_rescale` | `clamped`, `rescale_v2` |
| S8  | `cp_accumulate_rescale` | `relu_out`, `relu_v` |
| S9  | `cp_pool` | `max_reg`, `pool_cnt`, `pool_write_r` |
| FSM | `cnn_controller` | state → IDLE, `compute_en`, `prefetch_cnt` |

`cp_mac` (S1–S4) is **NOT** in this list — `prod0..4`, `sum01`, `sum23`,
`sum0123`, `tree_out`, `p4_d1/d2` keep whatever they held. This is intentional.

After a fresh `start`, the controller re-primes the SRW and only raises
`compute_en` once `prefetch_cnt == 4`, i.e. after **5 real SRW shifts**
([cnn_controller.v:187-191](../RTL/cnn_controller.v#L187)). The MAC pipeline is
**4** stages deep. Margin = 5 − 4 = **+1 cycle**.

## Reset case table

| Case | Garbage location at `rst` | Reg reset? | Garbage reaches `tree_out` after | `compute_en` rises after | Garbage seen while `compute_en=1`? | Accumulated? | Result |
|---|---|---|---|---|---|---|---|
| **C0** | none (cold power-up, X-state) | — | — | ≥5 cy | No | No | Bit-exact |
| **C1** | S1 `prod0..4` | No (MAC) | +3 cy | ≥5 cy | No (3 < 5) | No | Bit-exact |
| **C2** | S2 `sum01/23` | No (MAC) | +2 cy | ≥5 cy | No (2 < 5) | No | Bit-exact |
| **C3** | S3 `sum0123` | No (MAC) | +1 cy | ≥5 cy | No (1 < 5) | No | Bit-exact |
| **C4** | S4 `tree_out` | No (MAC) | +0 cy (already at output) | ≥5 cy | No (0 < 5) | No | Bit-exact |
| **C5** | S5 `acc` (accumulating) | **Yes** → 0 | n/a | ≥5 cy | — | acc=0, then overwritten at `a_in==0` | Bit-exact |
| **C6** | S5b/S_bias/S6/S7/S8 | **Yes** → 0 + valid=0 | n/a | ≥5 cy | — | valid bit blocks | Bit-exact |
| **C7** | S9 `max_reg`/`pool_cnt` | **Yes** → 0 | n/a | ≥5 cy | — | overwritten at `pool_cnt==0` | Bit-exact |

The decisive column is **"Garbage seen while `compute_en=1`?"**. For every MAC
location (C1–C4), the time for garbage to flush out of `tree_out` (≤4 cy) is
always **less** than the time until `compute_en` rises (≥5 cy). Worst case is C4
(garbage already at the output), with the minimum +1 cy margin.

## Two invariants that guarantee correctness

- **INV-1 (flush):** `cp_mac` is feed-forward, depth 4. Any garbage is flushed
  out of `tree_out` within ≤4 cycles after `start`. Holds because
  `MAC_depth (4) < priming (5)` → +1 cy margin. The garbage *does* propagate
  through the MAC (it is not frozen) — it simply exits before `compute_en` rises.

- **INV-2 (overwrite):** Accumulation points are **assigned**, not summed onto
  stale data:
  - `acc <= tree_sext + bias_in + round_add` at `a_in==0`
    ([cp_accumulate_rescale.v:57](../RTL/cp_accumulate_rescale.v#L57))
  - `max_reg <= relu_out` at `pool_cnt==0`
    ([cp_pool.v:39-40](../RTL/cp_pool.v#L39))

  This is a second backstop: even if INV-1 ever fell short, the window boundary
  re-seeds `acc`/`max_reg` from scratch.

## Why omit reset on `cp_mac` (not laziness)

- Reset fan-out across 8 PE × 5 multipliers + adder tree is large and, per the
  table above, **useless** — correctness comes from valid-bit gating, not from
  clearing the MAC.
- DSP18 pipeline registers live **inside** the hard multiplier block. Forcing a
  reset on them can push the multiply out of the DSP hard block or add logic,
  hurting timing/area.
- Standard safe convention: reset only feedback / multi-cycle state; gate pure
  feed-forward pipelines with a valid bit instead. `cp_mac` follows this.

X-state note (sim only): right after power-up the MAC registers are `X` until 4
cycles of real data flow through. Because `compute_en_in=0` during that window
and `acc`/`max_reg` reset to 0
([cp_pool.v:35](../RTL/cp_pool.v#L35) — "clear max so X-state never propagates
post-reset"), no `X` ever reaches the result path. On silicon it is a defined
stale value, not `X`.

## ⚠️ Durability warning

The argument rests on a **+1 cy margin only** (priming 5 vs MAC depth 4). It
breaks if a future change does any of:

1. Shortens priming below 5 SRW shifts (changes the `prefetch_cnt==4` gate), or
2. Increases MAC depth beyond 4 (adds a pipeline stage inside `cp_mac`), or
3. Changes `acc` init from overwrite (`<=`) to accumulate (`+=`).

Any of these requires adding a reset to `cp_mac`. The margin is sufficient today
but thin — keep this note with the code.
