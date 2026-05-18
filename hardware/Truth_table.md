# Truth Table — CP Engine, per Conv Layer (v2 — packed-ROM architecture)

Bảng chân trị cycle-by-cycle cho mỗi layer Conv sau khi refactor sang **4 ROM packed per-layer**.
Pipeline stages: SRW shift → MUX → mux_s1 → **w_packed** (ROM read) → MULT(S1) → TREE(S2-S4) → ACC(S5) → BIAS → RESCALE(S6-S7) → RELU(S8) → POOL(S9) → Pong SRAM.

> **Thay đổi so với v1:**
> - Bỏ `mux_s2` (giữ `mux_s1` 1-stage).
> - Bỏ 2-stage weight pipeline (`w_base_oc`, `w_base_ic`, `w_cur_flat`).
> - Thay bằng 4 ROM packed 40-bit/word (`w_rom_conv1..4`), read combinational → register vào `w_packed[oc]` (1-stage).
> - Pipeline latency **giữ nguyên 6 cycles**: MUX_reg(1) + WROM_reg(1) + MULT(1) + TREE(3).
> - Delay chain: `a_d6, inch_d6, ce_d6` (đổi tên `a_d5` → `a_d6` để khớp số cycle thực).

---

## Ký hiệu chung

| Ký hiệu | Ý nghĩa |
|---------|---------|
| `a` | channel counter, 0 → IN_CH-1, tăng mỗi cycle |
| `t` | output position counter, tăng khi `a == IN_CH-1` |
| `a_d6` | `a` delay **6 cycles** (MUX_reg + WROM_reg + MULT + TREE×3) |
| `shift_en` | `a == IN_CH-1` → dịch tất cả SRW |
| `sram_addr_en` | `a == IN_CH-2` (Conv2/3/4); luôn `1` cho Conv1 (IN_CH=1) |
| `compute_en` | 0 trong pre-fetch, 1 sau khi pre-fetch xong (6 shifts) |
| `out_valid` | `ce_d6 && (a_d6 == in_ch_d6 - 1)` |
| `pool_write` | `pool_cnt == 4 && relu_v` |
| `rp` | read pointer = t − 2 (padding=2), tính địa chỉ SRAM |
| `srw_din` | 0 nếu `pad_zero=1`, ngược lại lấy từ input_sram/ping_dout |
| `NOP` | compute_en = 0, ACC không hoạt động |
| `RST` | `a_d6 == 0` → acc ← tree_sext |
| `ACC` | `a_d6 ∈ [1, IN_CH-2]` → acc += tree_sext |
| `OUT` | `a_d6 == IN_CH-1` → out_valid = 1 |
| `w_packed[oc]` | 40-bit word đọc từ `w_rom_conv{N}` tại addr `oc*IN_CH + a` |

---

## Cấu hình tĩnh per layer

| Parameter | Conv1 | Conv2 | Conv3 | Conv4 |
|-----------|-------|-------|-------|-------|
| IN_CH | 1 | 4 | 4 | 8 |
| OUT_CH | 4 | 4 | 8 | 8 |
| IN_LEN | 2500 | 500 | 100 | 20 |
| OUT_LEN | 500 | 100 | 20 | 4 |
| RELU_EN | 0 | 0 | 0 | 1 |
| NB | **NB1=8** | **NB2=6** | **NB3=6** | **NB4=7** |
| cp_en[0..3] | 1 | 1 | 1 | 1 |
| cp_en[4..7] | 0 | 0 | 1 | 1 |
| Ping banks active | 1 (Input SRAM) | 4 | 4 | 8 |
| Pong banks written | 4 | 4 | 8 | 8 |
| ROM source | `w_rom_conv1[0:3]` | `w_rom_conv2[0:15]` | `w_rom_conv3[0:31]` | `w_rom_conv4[0:63]` |
| ROM addr formula | `oc` | `oc*4 + a` | `oc*4 + a` | `oc*8 + a` |
| Pre-fetch shifts | 6 | 6 | 6 | 6 |
| Pre-fetch cycles | 6 cy | 24 cy | 24 cy | 48 cy |
| out_valid period | 1 cy | 4 cy | 4 cy | 8 cy |
| pool_write period | 5 cy | 20 cy | 20 cy | 40 cy |

> Pre-fetch shifts = 6 (kernel=5, padding=2, plus 1 to fill pipeline before first valid output).
> Pre-fetch cycles = 6 × IN_CH (1 shift = IN_CH cycles của vòng `a`).
> NB values updated từ python_log: NB1=8, NB2=6, NB3=6, NB4=7 (input_shift=2, w_shift conv1..4 = 6,6,6,7).

---

## Conv1 — IN_CH=1, OUT_CH=4, IN_LEN=2500, OUT_LEN=500

**Đặc điểm:**
- IN_CH=1 → `a` luôn = 0 = IN_CH-1. `shift_en` luôn = 1. `a_d6` luôn = 0.
- RST và OUT xảy ra cùng cycle. `out_valid` pulse mỗi cycle.
- `sram_addr_en` = 1 mọi cycle (Conv1 special case).
- Weight: `w_packed[oc] <= w_rom_conv1[oc]` cho oc=0..3 (a luôn 0).

### Pre-fetch (6 shift cycles)

`prefetch_cnt` đếm shift events: tăng từ 0 đến 5 (6 shifts), tại shift thứ 6 (prefetch_cnt==5) → `compute_en <= 1`.
Conv1 IN_CH=1 → mỗi cycle là 1 shift → pre-fetch = 6 cycles.

| Cycle | a | t (before update) | rp at shift | sram_addr_en | shift_en | srw_din | SRW after shift | prefetch_cnt | compute_en after |
|-------|---|------------------|-------------|--------------|----------|---------|-----------------|--------------|------------------|
| 0 | 0 | 0 | −2 | 1 | 1 | 0 (pad) | [0,0,0,0,0] | 0→1 | 0 |
| 1 | 0 | 1 | −1 | 1 | 1 | 0 (pad) | [0,0,0,0,0] | 1→2 | 0 |
| 2 | 0 | 2 | 0 | 1 | 1 | x[0] | [0,0,0,0,x[0]] | 2→3 | 0 |
| 3 | 0 | 3 | 1 | 1 | 1 | x[1] | [0,0,0,x[0],x[1]] | 3→4 | 0 |
| 4 | 0 | 4 | 2 | 1 | 1 | x[2] | [0,0,x[0],x[1],x[2]] | 4→5 | 0 |
| 5 | 0 | 5 | 3 | 1 | 1 | x[3] | [0,x[0],x[1],x[2],x[3]] | 5 (stay) | 0→1 |

> Cycle 6 trở đi: `compute_en=1`. SRW cho computation cycle 6 = `[0,x[0],x[1],x[2],x[3]]`.
> **Mapping out_pos**: mỗi cycle conv tính 1 output. Output position thứ K (K=0..2499) tương ứng SRW center x[K]. Cách đếm cụ thể của `t` và `out_pos` được normalize trong downstream (pong_addr).

### Compute (steady-state, out_pos=P, P=0..2499)

| Cycle | a | t | rp | shift_en | srw_din | a_d6 | ACC ctrl | out_valid | Pipeline output |
|-------|---|---|----|----------|---------|------|----------|-----------|-----------------|
| 6 | 0 | 6 | 4 | 1 | x[4] | 0 | RST+OUT | 1 | (out_pos=0 ở t=cycle6−6=0) |
| 7 | 0 | 7 | 5 | 1 | x[5] | 0 | RST+OUT | 1 | (out_pos=1) |
| 8 | 0 | 8 | 6 | 1 | x[6] | 0 | RST+OUT | 1 | (out_pos=2) |
| ... | 0 | t | t−2 | 1 | x[t−2] | 0 | RST+OUT | 1 | (out_pos = t−6) |
| 2505 | 0 | 2505 | 2503 | 1 | x[2503] | 0 | RST+OUT | 1 | (out_pos=2499, last) |

> Tổng compute cycles: 2500. out_valid pulse từ cycle 6 đến 2505.

### Pool → Pong SRAM (Conv1)

`pool_cnt` đếm 0..4, reset về 0 sau khi đạt 4. pool_write fire khi `pool_cnt == 4 && relu_v`.

| pool_cnt | out_pos | pool_write | pong_addr | Pong SRAM ghi |
|----------|---------|------------|-----------|---------------|
| 0 | 0 | 0 | − | − |
| 1 | 1 | 0 | − | − |
| 2 | 2 | 0 | − | − |
| 3 | 3 | 0 | − | − |
| 4 | 4 | **1** | 0 | SRAM[0..3][0] ← max(out[0..4]) |
| 0 | 5 | 0 | − | − |
| ... | ... | ... | ... | ... |
| 4 | 9 | **1** | 1 | SRAM[0..3][1] ← max(out[5..9]) |
| ... | ... | ... | ... | ... |
| 4 | 2499 | **1** | 499 | SRAM[0..3][499] ← max(out[2495..2499]) |

> SRAM[4..7] không ghi (cp_en[4..7]=0). Total pool_write = 500 = OUT_LEN.

### Conv1 Special Note — pad_zero logic

[cp_engine.v pad_zero]:
```
pad_zero = (in_ch == 1) ? (sram_rd_addr_in <= 2) : (sram_rd_addr_in < 2)
```

Lý do `<=` cho Conv1: vì `sram_addr_en=1` cùng cycle với `shift_en` → addr issue và shift xảy ra song song. SRAM 1-cycle latency → dout đến shift cycle kế tiếp khi `t` đã tăng. Vì vậy điều kiện padding cho Conv1 dịch lên 1.

---

## Conv2 — IN_CH=4, OUT_CH=4, IN_LEN=500, OUT_LEN=100

**Đặc điểm:**
- `a` cycle qua 0→3. `shift_en` tại `a=3`. `sram_addr_en` tại `a=2`.
- `out_valid` pulse mỗi 4 cycles.
- ROM: `w_rom_conv2[oc*4 + a]` (oc=0..3, a=0..3) — 16 entries × 40b.
- Pre-fetch: 6 shifts × 4 cy/shift = 24 cycles.

### Pre-fetch (24 cycles)

| Shift # | Cycles (a=0..3) | rp at shift | srw_din | SRW after shift | compute_en |
|---------|-----------------|-------------|---------|-----------------|------------|
| 0 | 0..3 | −2 | 0 (pad) | [−,−,−,−,0] | 0 |
| 1 | 4..7 | −1 | 0 (pad) | [−,−,−,0,0] | 0 |
| 2 | 8..11 | 0 | x[0] | [−,−,0,0,x[0]] | 0 |
| 3 | 12..15 | 1 | x[1] | [−,0,0,x[0],x[1]] | 0 |
| 4 | 16..19 | 2 | x[2] | [0,0,x[0],x[1],x[2]] | 0 |
| 5 | 20..23 | 3 | x[3] | [0,x[0],x[1],x[2],x[3]] | 0→1 |

> Sau cycle 23: `prefetch_cnt==5` → `compute_en=1`.

### Compute — 1 vòng a (4 cycles), out_pos=P (P=0..99)

| Cycle offset | a | a_d6 | sram_addr_en | shift_en | srw_din | ACC ctrl | out_valid |
|-------------|---|------|--------------|----------|---------|----------|-----------|
| 4P+0 | 0 | 0 | 0 | 0 | − | **RST** | 0 |
| 4P+1 | 1 | 1 | 0 | 0 | − | ACC | 0 |
| 4P+2 | 2 | 2 | **↑addr[P+4]** | 0 | − | ACC | 0 |
| 4P+3 | 3 | 3 | − | 1 | x[P+4] | **OUT** | **1** |

> Mỗi 4 cycles cho ra 1 output. Total compute = 100×4 = 400 cycles.

### Toàn bộ bảng (first 3 + last)

| out_pos | cycle out_valid | rp at shift | SRW tại compute |
|---------|-----------------|-------------|------------------|
| 0 | 27 (=24+3) | x[3] loaded | [0,x[0],x[1],x[2],x[3]] |
| 1 | 31 | x[4] | [x[0],x[1],x[2],x[3],x[4]] |
| 2 | 35 | x[5] | [x[1],x[2],x[3],x[4],x[5]] |
| ... | ... | ... | ... |
| 98 | 419 | 0 (OOB) | [x[96],x[97],x[98],x[99],0] |
| 99 | 423 | 0 (OOB) | [x[97],x[98],x[99],0,0] |

### Pool → Pong SRAM (Conv2)

| pool_cnt | out_pos range | pool_write cycle | pong_addr |
|----------|---------------|------------------|-----------|
| 0→4 | 0..4 | 43 (=27+4×4) | 0 |
| 0→4 | 5..9 | 63 | 1 |
| ... | ... | ... | ... |
| 0→4 | 95..99 | 423 | 19 |

> Total pool_write = 20 lần. SRAM[4..7] không ghi (cp_en[4..7]=0).

Wait — pong_addr range 0..19 nhưng OUT_LEN=100? Đúng là pong_addr = 0..99 (100 entries). Mỗi 5 output → 1 pool_write → 100/5 = 20 pool_writes? **Sai**.

Sửa: OUT_LEN = 100 là output sau pool. Conv2 output before pool = 500 values per channel (vì IN_LEN=500). Sau pool /5 = 100. Vậy pool_write = 100 (đúng OUT_LEN). Bảng cũ chia nhỏ chỉ là tóm tắt 20 dòng mẫu, không phải tổng.

| pool_cnt | conv_out range | pool_write count | pong_addr range |
|----------|----------------|------------------|-----------------|
| 0→4 cyclic | 0..499 | 100 lần | 0..99 |

---

## Conv3 — IN_CH=4, OUT_CH=8, IN_LEN=100, OUT_LEN=20

**Đặc điểm:**
- Timing giống Conv2 (IN_CH=4).
- Khác: `cp_en=0xFF` → tất cả 8 blocks active, Pong SRAM ghi đủ 8 banks.
- ROM: `w_rom_conv3[oc*4 + a]` (oc=0..7, a=0..3) — 32 entries × 40b.
- Pre-fetch: 24 cycles.

### Compute — giống Conv2 (4 cy/output)

| Cycle offset | a | a_d6 | ACC ctrl | out_valid |
|-------------|---|------|----------|-----------|
| 4P+0 | 0 | 0 | RST | 0 |
| 4P+1 | 1 | 1 | ACC | 0 |
| 4P+2 | 2 | 2 | ACC | 0 |
| 4P+3 | 3 | 3 | **OUT** | **1** |

> out_pos = 0..99 (100 conv outputs / channel) → 400 compute cycles. Sau pool /5 → 20 entries vào Pong.

### Pool → Pong SRAM (Conv3)

| pool_cnt | conv_out range | pool_write | pong_addr | Pong SRAM |
|----------|----------------|------------|-----------|-----------|
| 0→4 | 0..4 | 1 | 0 | SRAM[0..7][0] |
| 0→4 | 5..9 | 1 | 1 | SRAM[0..7][1] |
| ... | ... | ... | ... | ... |
| 0→4 | 95..99 | 1 | 19 | SRAM[0..7][19] |

> Total pool_write = 20 lần (= OUT_LEN). Tất cả 8 Pong banks được ghi mỗi pool_write.

---

## Conv4 — IN_CH=8, OUT_CH=8, IN_LEN=20, OUT_LEN=4

**Đặc điểm:**
- IN_CH=8. **RELU_EN=1**. `out_valid` mỗi 8 cycles. `pool_write` mỗi 40 cycles.
- Tổng pool_write = 4 (= OUT_LEN).
- ROM: `w_rom_conv4[oc*8 + a]` (oc=0..7, a=0..7) — 64 entries × 40b.
- Pre-fetch: 6 shifts × 8 cy/shift = 48 cycles.

### Pre-fetch (48 cycles)

| Shift # | Cycles (a=0..7) | rp at shift | srw_din | SRW after shift | compute_en |
|---------|----------------|-------------|---------|-----------------|------------|
| 0 | 0..7 | −2 | 0 (pad) | [−,−,−,−,0] | 0 |
| 1 | 8..15 | −1 | 0 (pad) | [−,−,−,0,0] | 0 |
| 2 | 16..23 | 0 | x[0] | [−,−,0,0,x[0]] | 0 |
| 3 | 24..31 | 1 | x[1] | [−,0,0,x[0],x[1]] | 0 |
| 4 | 32..39 | 2 | x[2] | [0,0,x[0],x[1],x[2]] | 0 |
| 5 | 40..47 | 3 | x[3] | [0,x[0],x[1],x[2],x[3]] | 0→1 |

> Cycle 47: `prefetch_cnt==5` → `compute_en=1`.

### Compute — 1 vòng a (8 cycles), out_pos=P (P=0..19)

| Cycle offset | a | a_d6 | sram_addr_en | shift_en | srw_din | ACC ctrl | out_valid | Downstream |
|-------------|---|------|--------------|----------|---------|----------|-----------|------------|
| 8P+0 | 0 | 0 | 0 | 0 | − | **RST** | 0 | − |
| 8P+1 | 1 | 1 | 0 | 0 | − | ACC | 0 | − |
| 8P+2 | 2 | 2 | 0 | 0 | − | ACC | 0 | − |
| 8P+3 | 3 | 3 | 0 | 0 | − | ACC | 0 | − |
| 8P+4 | 4 | 4 | 0 | 0 | − | ACC | 0 | − |
| 8P+5 | 5 | 5 | 0 | 0 | − | ACC | 0 | − |
| 8P+6 | 6 | 6 | **↑addr[P+4]** | 0 | − | ACC | 0 | − |
| 8P+7 | 7 | 7 | − | **1** | x[P+4] | **OUT** | **1** | S6: shift+round |

> P=0..19 → 160 compute cycles.

### Toàn bộ out_pos Conv4

| out_pos P | cycle out_valid | SRW tap[4..0] | Convolution |
|-----------|-----------------|---------------|-------------|
| 0 | 55 (=48+7) | [0,x[0],x[1],x[2],x[3]] | pad×w0 + x[0]×w1 + x[1]×w2 + x[2]×w3 + x[3]×w4 |
| 1 | 63 | [x[0],x[1],x[2],x[3],x[4]] | full (no pad) |
| 2 | 71 | [x[1]..x[5]] | full |
| ... | ... | ... | ... |
| 17 | 191 | [x[16],x[17],x[18],x[19],?] | edge |
| 18 | 199 | [x[17],x[18],x[19],0,0] | pad right |
| 19 | 207 | [x[18],x[19],0,0,0] | pad right |

### Pool → Pong SRAM (Conv4) — đầy đủ

| pool_cnt | out_pos (relu_v) | relu_out | max_reg | pool_write | pong_addr | Pong SRAM |
|----------|------------------|----------|---------|------------|-----------|-----------|
| 0 | 0 | v0 | v0 | 0 | − | − |
| 1 | 1 | v1 | max(v0,v1) | 0 | − | − |
| 2 | 2 | v2 | max(v0..v2) | 0 | − | − |
| 3 | 3 | v3 | max(v0..v3) | 0 | − | − |
| 4 | 4 | v4 | max(v0..v4) | **1** | 0 | SRAM[0..7][0] ← max(v0..v4) |
| 0 | 5 | v5 | v5 | 0 | − | − |
| ... | ... | ... | ... | ... | ... | ... |
| 4 | 9 | v9 | max(v5..v9) | **1** | 1 | SRAM[0..7][1] |
| 4 | 14 | v14 | max(v10..v14) | **1** | 2 | SRAM[0..7][2] |
| 4 | 19 | v19 | max(v15..v19) | **1** | 3 | SRAM[0..7][3] |

> **layer_done** fires khi `pool_write=1 && pong_addr == OUT_LEN-1 = 3`.
> Sau đó FSM chuyển sang `GAP_FC_S`.

---

## Weight ROM Access Pattern

Mỗi cycle, controller phát `a` → cp_engine đọc 8 weight words (1/oc) từ ROM theo `layer_state`:

```
case (layer_state):
  CONV1 (layer_idx=0):  w_packed[oc] <= w_rom_conv1[oc];                  // a=0 always
  CONV2 (layer_idx=1):  w_packed[oc] <= w_rom_conv2[oc*4 + a[1:0]];       // 4 oc × 4 ic
  CONV3 (layer_idx=2):  w_packed[oc] <= w_rom_conv3[oc*4 + a[1:0]];       // 8 oc × 4 ic
  CONV4 (layer_idx=3):  w_packed[oc] <= w_rom_conv4[oc*8 + a[2:0]];       // 8 oc × 8 ic
```

Mỗi `w_packed[oc]` là 40-bit = {tap4, tap3, tap2, tap1, tap0}, đến cp_block S1 MULT cùng cycle với `mux_s1`.

**Timing alignment:**
- `a` ở cycle N → `w_packed[oc]` register ở cycle N+1 (1-stage ROM read)
- `mux_comb` ở cycle N → `mux_s1` register ở cycle N+1
- Cả 2 vào MULT cycle N+2 → `prod` register

---

## Pipeline Valid Chain (giữ nguyên)

| Signal | Source | Trigger | Delay từ out_valid |
|--------|--------|---------|-------------------|
| `out_valid` | ACC (S5) | `a_d6 == in_ch_d6 - 1 && ce_d6` | 0 |
| `bias_valid` | S_bias register | out_valid | +1 cy |
| `rescale_v1` | S6 | bias_valid | +1 cy |
| `rescale_v2` | S7 | rescale_v1 | +1 cy |
| `relu_v` | S8 | rescale_v2 | +1 cy |
| `pool_write` | S9 | `relu_v && pool_cnt==4` | tới khi pool_cnt đạt 4 |
| Pong SRAM write | pool_write && cp_en[oc] | same cycle | 0 |

Total latency từ ACC OUT đến pool_write: 4 cycles (S6+S7+S8+pool register) + pool_cnt phase (0..4 cycles tùy vị trí trong window).

---

## Timing Tổng Thể (Cycles per Layer, post-refactor)

| Layer | Pre-fetch | Compute (conv) | Pool drain | Total ≈ | @ 100MHz |
|-------|-----------|---------------|------------|---------|----------|
| Conv1 | 6 | 2500 | ~5 (last pool flush) | ~2511 | 25.1 µs |
| Conv2 | 24 | 500×4=2000 | ~10 | ~2034 | 20.3 µs |
| Conv3 | 24 | 100×4=400 | ~10 | ~434 | 4.3 µs |
| Conv4 | 48 | 20×8=160 | ~10 | ~218 | 2.2 µs |
| **Total Conv** | | | | **~5197** | **~52.0 µs** |
| GAP/FC/Argmax | | 22 cy | | 22 | 0.2 µs |
| **Total** | | | | **~5219** | **~52.2 µs** |

> Pre-fetch = 6 × IN_CH (constant 6 shifts).
> Latency output đầu tiên = pre-fetch + 0 (compute fires ngay cycle sau pre-fetch).

---

## Ghi chú

- **Tất cả 8 CP blocks chạy song song** — bảng trên cho 1 block (1 output channel).
- **Conv1**: chỉ đọc Input SRAM (1 bank). MUX trong cp_engine chọn `input_sram_dout` thay vì `ping_dout[0]`.
- **Conv2/3/4**: đọc Ping SRAM 8 banks song song qua `ping_dout[ch*8 +: 8]`.
- **srw_din OOB**: khi `pad_zero=1` (rp < 0 hoặc rp ≥ IN_LEN_effective) → srw_din = 0.
- **acc_final**: combinational forward — khi `out_valid=1`, `acc` register chưa kịp update (non-blocking), nên dùng `acc + tree_sext` trực tiếp.
  - **Expected critical path**: `(acc + tree_sext) + bias_in` = 2 cascaded 32b adders trong 1 cycle. Verify TimeQuest; nếu fail tách stage (xem comment trong [cp_block.v](RTL/cp_block.v)).
- **Weight ROM mapping**:
  - `w_rom_conv1/2/3`: MLAB (160b, 640b, 1280b — small enough for distributed RAM)
  - `w_rom_conv4`: M10K (2560b — 1 block)
  - `b_store` (32 × INT32 = 1024b): MLAB
- **NB updated**: NB1=8, NB2=6, NB3=6, NB4=7 (input_shift=2 đã cộng vào NB1).
