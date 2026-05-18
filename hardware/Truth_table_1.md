# Truth Table v1 — CNN Accelerator Full Datapath

Tài liệu này mô tả **toàn bộ datapath** của CNN Accelerator: từ HPS load input → Conv1..4 → GAP → FC → Argmax → Result. Đầy đủ cycle-by-cycle, signal-by-signal, từng stage pipeline.

> **Phạm vi**: bao quát cả convolution engine (cp_engine + 8 cp_block) và post-processing (gap_fc_argmax). Mỗi layer trình bày đầy đủ: control signals, datapath stages, memory access, valid chain, pool sequence.

---

## 0. Tổng quan kiến trúc

```
HPS
 │ Avalon-MM (avs_*)
 ▼
avalon_slave ──────────► Input SRAM (2500×8b)
   │ start                  ▲
   │                        │ rd_addr
   ▼                        │
cnn_controller ──────► cp_engine ──┬─► 8 × cp_block
   │ FSM                            │     │
   │                                │     ▼
   │                                │   Pool out (INT8) + pool_write
   │                                ▼
   │                            Pong port write
   │                                │
   ▼                                ▼
Ping-Pong SRAM (2 banks × 8 ch × 500 entries)
   │ rd
   ▼
gap_fc_argmax ─► result[1:0]
```

**FSM states**: `IDLE → LOAD_INPUT → CONV1 → CONV2 → CONV3 → CONV4 → GAP_FC_S → DONE_S`
**GAP_FC sub-states**: `GAP_SUB → FC_SUB → FC_FLUSH_S → ARGMAX_SUB → DONE_SUB`

---

## 1. Cấu hình tĩnh per layer

| Parameter | Conv1 | Conv2 | Conv3 | Conv4 |
|-----------|-------|-------|-------|-------|
| IN_CH | 1 | 4 | 4 | 8 |
| OUT_CH | 4 | 4 | 8 | 8 |
| IN_LEN | 2500 | 500 | 100 | 20 |
| OUT_LEN (post-pool) | 500 | 100 | 20 | 4 |
| KERNEL | 5 | 5 | 5 | 5 |
| PADDING | 2 | 2 | 2 | 2 |
| POOL stride | 5 | 5 | 5 | 5 |
| RELU_EN | 0 | 0 | 0 | **1** |
| NB (rescale shift) | **8** | **6** | **6** | **7** |
| cp_en bitmask | 0x0F | 0x0F | 0xFF | 0xFF |
| Active OC blocks | 4 (oc=0..3) | 4 (oc=0..3) | 8 (oc=0..7) | 8 (oc=0..7) |
| Source SRAM | Input SRAM | Ping bank A | Ping bank B | Ping bank A |
| Dest SRAM | Pong bank B | Pong bank A | Pong bank B | Pong bank A |
| ROM source | `w_rom_conv1[0:3]` | `w_rom_conv2[0:15]` | `w_rom_conv3[0:31]` | `w_rom_conv4[0:63]` |
| ROM addr formula | `oc` | `oc*4 + a` | `oc*4 + a` | `oc*8 + a` |
| Bias source | `b_store[oc*4 + 0]` | `b_store[oc*4 + 1]` | `b_store[oc*4 + 2]` | `b_store[oc*4 + 3]` |
| `bank_sel` (during) | 0 | 1 | 0 | 1 |
| Cycles per output | 1 | 4 | 4 | 8 |
| `out_valid` period | 1 cy | 4 cy | 4 cy | 8 cy |
| `pool_write` period | 5 cy | 20 cy | 20 cy | 40 cy |
| Pre-fetch shifts | 6 | 6 | 6 | 6 |
| Pre-fetch cycles | 6 | 24 | 24 | 48 |
| Compute cycles | 2500 | 2000 | 400 | 160 |
| Total cycles per layer | ~2510 | ~2030 | ~430 | ~210 |

---

## 2. Pipeline timing — Datapath chung cho mọi Conv

### 2.1. Stage map (post-refactor)

```
cycle N    : a, layer_state, in_ch, nb              [from controller]
             ↓
             SRW[a] mux_comb (combinational)
             w_rom_convX[oc*IN_CH + a] read         (combinational)
             ↓
cycle N+1  : mux_s1   ← mux_comb                    [40-bit register]
             w_packed[oc] ← w_rom output           [40-bit per oc, 8 registers]
             a_d1, ce_d1, inch_d1                  [delay chain]
             ↓
cycle N+2  : prod[t] ← mux_s1[t*8+:8] × w[t*8+:8]   [S1 — 5 multipliers/oc]
             a_d2
             ↓
cycle N+3  : sum01  ← prod[0] + prod[1]            [S2]
             sum23  ← prod[2] + prod[3]
             p4_d1  ← prod[4]
             a_d3
             ↓
cycle N+4  : sum0123 ← sum01 + sum23                [S3]
             p4_d2   ← p4_d1
             a_d4
             ↓
cycle N+5  : tree_out ← sum0123 + p4_d2             [S4]
             a_d5
             ↓
cycle N+6  : acc ← tree_sext  (if a_d6==0)          [S5 ACC]
             acc ← acc + tree_sext  (else)
             out_valid = ce_d6 && (a_d6 == in_ch_d6-1)
             biased ← acc_final + bias_in  (if out_valid)  [S_bias]
             a_d6
             ↓
cycle N+7  : shifted ← (biased + round_add) >>> nb  [S6]
             bias_valid → rescale_v1
             ↓
cycle N+8  : clamped ← clamp(shifted, ±127)         [S7]
             rescale_v1 → rescale_v2
             ↓
cycle N+9  : relu_out ← (relu_en && neg) ? 0 : clamped  [S8]
             rescale_v2 → relu_v
             ↓
cycle N+10 : Pool update (max_reg, pool_cnt)        [S9]
             If pool_cnt==4 && relu_v:
                pool_write_r ← 1
                pool_out ← max_reg     (broadcast to Pong)
```

**Latency từ `a` đến pool_write** = 10 cycles (best case, khi rơi đúng pool_cnt=4).

### 2.2. Delay chain trong cp_engine

| Register | Mục đích | Source |
|----------|---------|--------|
| `a_d1..a_d6` | Channel counter cho cp_block ACC stage | `a` |
| `inch_d1..inch_d6` | IN_CH cho out_valid comparison | `in_ch` |
| `ce_d1..ce_d6` | compute_en cho S5/S_bias gating | `compute_en` |

Tổng 6 cycles, khớp với 6-stage pipeline trước S5.

### 2.3. Valid chain (sau cp_block S5)

| Signal | Stage | Source | Propagation |
|--------|-------|--------|-------------|
| `out_valid` | S5 end | `ce_d6 && (a_d6 == in_ch_d6-1)` | combinational from delay chain |
| `bias_valid` | S_bias | `out_valid` (registered) | +1 cy |
| `rescale_v1` | S6 | `bias_valid` (registered) | +1 cy |
| `rescale_v2` | S7 | `rescale_v1` (registered) | +1 cy |
| `relu_v` | S8 | `rescale_v2` (registered) | +1 cy |
| `pool_write_r` | S9 | `relu_v && pool_cnt==4` | combinational + gated |

Tổng latency `out_valid → pool_write` = 4 cycles (S_bias→S6→S7→S8→Pool register) + pool_cnt phase (0..4 cycles tuỳ vị trí trong window).

---

## 3. Weight ROM access — Chi tiết

### 3.1. Per-layer ROM layout

| ROM | Entries | Bits | Mapping | Format mỗi entry |
|-----|---------|------|---------|------------------|
| `w_rom_conv1` | 4 (4oc × 1ic) | 160 | MLAB | `{tap4, tap3, tap2, tap1, tap0}` (40-bit packed INT8) |
| `w_rom_conv2` | 16 (4oc × 4ic) | 640 | MLAB | same |
| `w_rom_conv3` | 32 (8oc × 4ic) | 1280 | MLAB | same |
| `w_rom_conv4` | 64 (8oc × 8ic) | 2560 | M10K | same |
| `b_store` | 32 (8oc × 4layer) | 1024 | MLAB | INT32 LE |

### 3.2. Per-layer access pattern

Cycle N: controller phát `a` ∈ [0, IN_CH-1] và `layer_state` ∈ {CONV1..4}. cp_engine read combinational, register vào `w_packed[oc]` ở cycle N+1.

#### Conv1 (IN_CH=1)
```
a always 0
For oc = 0..3:  w_packed[oc] <= w_rom_conv1[oc]
For oc = 4..7:  w_packed[oc] <= 40'd0          (cp_en masks off)
```

#### Conv2 (IN_CH=4)
```
a cycles 0,1,2,3,0,1,2,3,...
For oc = 0..3:  w_packed[oc] <= w_rom_conv2[oc*4 + a[1:0]]
For oc = 4..7:  w_packed[oc] <= 40'd0
```

#### Conv3 (IN_CH=4)
```
For oc = 0..7:  w_packed[oc] <= w_rom_conv3[oc*4 + a[1:0]]
```

#### Conv4 (IN_CH=8)
```
a cycles 0..7
For oc = 0..7:  w_packed[oc] <= w_rom_conv4[oc*8 + a[2:0]]
```

### 3.3. Bias access

`b_cur[oc] <= b_store[oc*4 + layer_idx]` — cập nhật khi layer_idx đổi (mỗi layer transition).

---

## 4. CONV1 — IN_CH=1, OUT_CH=4, IN_LEN=2500, OUT_LEN=500

**Đặc điểm**:
- IN_CH=1 → `a` = 0 vĩnh viễn → `shift_en` = 1 mọi cycle → `a_d6` = 0.
- `sram_addr_en` = 1 mọi cycle (Conv1 special: phát addr cùng cycle với shift, vì in_ch=1).
- `pad_zero` công thức Conv1 dùng `<=` (vs `<` cho layer khác) để bù timing.
- `out_valid` pulse mỗi cycle (RST+OUT đồng thời).
- Source: Input SRAM (input_sram_dout). Destination: Pong bank B (cp_en=0x0F).

### 4.1. Pre-fetch sequence

| Cycle | a | t (current) | `sram_rd_addr` issued | shift_en | srw_din ← | SRW after shift | prefetch_cnt | compute_en (next cy) |
|-------|---|-------------|----------------------|----------|-----------|-----------------|--------------|----------------------|
| 0 | 0 | 0 | rp=0 (clamped) | 1 | 0 (pad) | [0,0,0,0,0] | 0→1 | 0 |
| 1 | 0 | 1 | rp=0 (clamped) | 1 | 0 (pad) | [0,0,0,0,0] | 1→2 | 0 |
| 2 | 0 | 2 | rp=0 | 1 | x[0] | [0,0,0,0,x[0]] | 2→3 | 0 |
| 3 | 0 | 3 | rp=1 | 1 | x[1] | [0,0,0,x[0],x[1]] | 3→4 | 0 |
| 4 | 0 | 4 | rp=2 | 1 | x[2] | [0,0,x[0],x[1],x[2]] | 4→5 | 0 |
| 5 | 0 | 5 | rp=3 | 1 | x[3] | [0,x[0],x[1],x[2],x[3]] | 5 (hold) | **0→1** |

> `prefetch_cnt == 5` ở cycle 5 → compute_en set 1 cho cycle 6.

### 4.2. Compute steady-state

Mỗi cycle: SRW shift, MUX read, conv tính → output sau 6 cycles pipeline.

| Cycle | a | t | SRW center tap | Output ready (after pipeline) | a_d6 | ACC ctrl | out_valid |
|-------|---|---|----------------|------------------------------|------|----------|-----------|
| 6 | 0 | 6 | x[2] | — (pipeline filling) | 0 | RST+OUT | 1 |
| 7 | 0 | 7 | x[3] | — | 0 | RST+OUT | 1 |
| ... | | | | | | | |
| 12 | 0 | 12 | x[8] | conv_out for SRW@cy6 = conv(out_pos=0) | 0 | RST+OUT | 1 |
| ... | | | | | | | |
| 2505 | 0 | 2505 | x[2501] | conv_out for SRW@cy2499 = out_pos=2499 | 0 | RST+OUT | 1 |

> Out_valid pulse 2500 cycles (cy 6..2505). Pool consumes liên tục.

### 4.3. Pool sequence (per channel, oc=0..3 active)

`pool_cnt` đếm 0..4 mỗi `relu_v=1`. Khi `pool_cnt==4`, fire `pool_write_r`.

| pool_cnt | relu_out cycle | max_reg | pool_write | pong_addr | Pong SRAM write |
|----------|---------------|---------|------------|-----------|-----------------|
| 0 | 16 (=cy6+10 latency) | v0 | 0 | − | − |
| 1 | 17 | max(v0,v1) | 0 | − | − |
| 2 | 18 | max(v0..v2) | 0 | − | − |
| 3 | 19 | max(v0..v3) | 0 | − | − |
| 4 | 20 | max(v0..v4) | **1** | 0 | bank B[oc][0] ← max(v0..v4) (oc=0..3) |
| 0 | 21 | v5 | 0 | − | − |
| ... | ... | ... | ... | ... | ... |
| 4 | 2515 | max(v2495..v2499) | **1** | 499 | bank B[oc][499] (last) |

> Total pool_write = 500. Bank B[0..3] được ghi. Bank B[4..7] không ghi (cp_en[4..7]=0).

### 4.4. Layer_done

`layer_done = (pong_addr == 499) && pool_write`. Fire ở cycle 2515.

→ cnn_controller transition CONV1 → CONV2:
- `bank_sel <= ~bank_sel` (0 → 1) — Pong B trở thành Ping cho Conv2.
- `srw_rst <= 1`, `pool_rst <= 1`, `compute_en <= 0`, `a <= 0`, `t <= 0`, `pong_addr <= 0`, `prefetch_cnt <= 0`.
- Load Conv2 config: `in_ch <= 4`, `out_len <= 100`, `nb <= 6`, `cp_en <= 0x0F`.

---

## 5. CONV2 — IN_CH=4, OUT_CH=4, IN_LEN=500, OUT_LEN=100

**Đặc điểm**:
- `a` cycles 0,1,2,3,0,1,2,3,... mỗi 4 cycles.
- `shift_en` = (a==3) → mỗi 4 cy/shift.
- `sram_addr_en` = (a==2) → phát addr 1 cy trước shift (SRAM 1-cy latency).
- Source: Ping bank B (đã được Conv1 ghi). 4 channels song song qua `ping_dout[0..3][7:0]`.
- Destination: Pong bank A, cp_en=0x0F.

### 5.1. Pre-fetch (24 cycles = 6 shifts × 4 cy)

| Shift # | Cycle range (a=0..3) | rp at shift | sram_addr_en cycle | srw_din at shift | SRW after | prefetch_cnt | compute_en |
|---------|--------------------|-------------|---------------------|------------------|-----------|--------------|------------|
| 1 | 0..3 | −2 (clamped 0) | cy 2 | 0 (pad) | [_,_,_,_,0] | 0→1 | 0 |
| 2 | 4..7 | −1 (clamped 0) | cy 6 | 0 (pad) | [_,_,_,0,0] | 1→2 | 0 |
| 3 | 8..11 | 0 | cy 10 | ping[0] | [_,_,0,0,ping[0]] | 2→3 | 0 |
| 4 | 12..15 | 1 | cy 14 | ping[1] | [_,0,0,ping[0],ping[1]] | 3→4 | 0 |
| 5 | 16..19 | 2 | cy 18 | ping[2] | [0,0,ping[0],ping[1],ping[2]] | 4→5 | 0 |
| 6 | 20..23 | 3 | cy 22 | ping[3] | [0,ping[0],ping[1],ping[2],ping[3]] | 5 | **→1** |

> Mỗi channel có SRW riêng. SRW[ch][0..4] shift đồng bộ cùng `shift_en`.

### 5.2. Compute — 1 output mỗi 4 cycles, out_pos=P (P=0..499)

Mỗi vòng `a` xử lý 4 channel partial products, ACC tích luỹ qua 4 cycles, fire OUT ở cycle thứ 4.

| Cycle offset (4P+) | a | a_d6 | sram_addr_en | shift_en | srw_din | ACC ctrl | out_valid | Notes |
|---------------------|---|------|--------------|----------|---------|----------|-----------|-------|
| 0 | 0 | 0 | 0 | 0 | − | **RST** (acc ← tree_sext for ch=0) | 0 | Multiplier reads w[oc][0] for ic=0 |
| 1 | 1 | 1 | 0 | 0 | − | **ACC** (acc += tree_sext for ch=1) | 0 | w[oc][1] for ic=1 |
| 2 | 2 | 2 | **↑** addr=rp+1 | 0 | − | ACC (ch=2) | 0 | Issue next sample addr |
| 3 | 3 | 3 | − | **1** | ping[ch][rp+1] | **OUT** (acc_final + bias) | **1** | All 4 channels accumulated |

> Pong write fires sau pipeline drain: `out_valid` cy 23 → `pool_write` ~cy 23+4+pool_cnt.

### 5.3. Toàn bộ Conv2 timeline (first 3 + last)

| out_pos | First valid cycle (`out_valid`) | Conv center | Pool write cycle | pong_addr |
|---------|--------------------------------|-------------|------------------|-----------|
| 0 | 23 | ping[2] | ~43 | (after 5 outputs collected) |
| 1 | 27 | ping[3] | — | − |
| ... | ... | ... | ... | ... |
| 4 | 39 | ping[6] | 43 | 0 (bank A[0..3][0]) |
| 9 | 59 | ping[11] | 63 | 1 |
| ... | ... | ... | ... | ... |
| 499 | 23 + 499×4 = 2019 | ping[501] (OOB) | ~2039 | 99 |

> Total `out_valid` events = 500 (one per output_pos). After pool /5: pong_addr 0..99 (100 entries). Total pool_write = 100.

### 5.4. Layer_done

`layer_done = (pong_addr == 99) && pool_write` fires ~cy 2039.
Transition CONV2 → CONV3: `bank_sel` 1→0, load Conv3 config.

---

## 6. CONV3 — IN_CH=4, OUT_CH=8, IN_LEN=100, OUT_LEN=20

**Đặc điểm**:
- Timing **giống Conv2** (IN_CH=4): 4 cy/output, pre-fetch 24 cy.
- Khác: `cp_en=0xFF` → tất cả 8 cp_blocks active → Pong ghi đủ 8 banks.
- Source: Ping bank A (Conv2 đã ghi). Destination: Pong bank B.
- `nb = 6` (same as Conv2).

### 6.1. Pre-fetch — giống Conv2 (24 cy)

| Shift # | rp | srw_din from 4 channels (ping_dout[0..3]) | compute_en |
|---------|-----|--------------------------------------------|------------|
| 1 | −2 | 0 (pad) all 4 ch | 0 |
| 2 | −1 | 0 (pad) | 0 |
| 3 | 0 | ping[0..3] @ addr 0 | 0 |
| 4 | 1 | ping[0..3] @ addr 1 | 0 |
| 5 | 2 | ping[0..3] @ addr 2 | 0 |
| 6 | 3 | ping[0..3] @ addr 3 | →1 |

### 6.2. Compute

| Cycle offset (4P+) | a | a_d6 | ACC ctrl | out_valid | Pong ghi |
|---------------------|---|------|----------|-----------|----------|
| 0 | 0 | 0 | RST (oc 0..7) | 0 | − |
| 1 | 1 | 1 | ACC | 0 | − |
| 2 | 2 | 2 | ACC | 0 | − |
| 3 | 3 | 3 | **OUT** | **1** | (later, after pool) |

> 100 output_pos × 4 cy = 400 compute cycles. After pool /5: 20 entries.

### 6.3. Pool & Pong write

| pool_cnt | conv_out range | pool_write | pong_addr | Pong bank B |
|----------|---------------|------------|-----------|-------------|
| 0→4 | 0..4 | 1 | 0 | B[0..7][0] ← max |
| 0→4 | 5..9 | 1 | 1 | B[0..7][1] |
| ... | ... | ... | ... | ... |
| 0→4 | 95..99 | 1 | 19 | B[0..7][19] |

> Total pool_write = 20. **Tất cả 8 channels** được ghi mỗi pool_write.

### 6.4. Layer_done

`pong_addr == 19 && pool_write` → transition CONV3 → CONV4. `bank_sel` 0→1. Load Conv4: `in_ch=8, out_len=4, nb=7, relu_en=1, cp_en=0xFF`.

---

## 7. CONV4 — IN_CH=8, OUT_CH=8, IN_LEN=20, OUT_LEN=4

**Đặc điểm**:
- IN_CH=8 → 8 cy/output. `a` cycles 0..7.
- `shift_en` = (a==7), `sram_addr_en` = (a==6).
- **RELU_EN = 1** — chỉ layer duy nhất có ReLU.
- `nb = 7`. cp_en=0xFF.
- Source: Ping bank B. Destination: Pong bank A.
- Total compute cycles = 20 × 8 = 160. Pool /5 = 4 outputs.

### 7.1. Pre-fetch (48 cycles = 6 shifts × 8 cy)

| Shift # | Cycle (a=0..7) | rp | srw_din (8 ch parallel) | compute_en |
|---------|----------------|-----|------------------------|------------|
| 1 | 0..7 | −2 | 0 (pad) ×8 | 0 |
| 2 | 8..15 | −1 | 0 (pad) ×8 | 0 |
| 3 | 16..23 | 0 | ping[0..7] @ addr 0 | 0 |
| 4 | 24..31 | 1 | ping[0..7] @ addr 1 | 0 |
| 5 | 32..39 | 2 | ping[0..7] @ addr 2 | 0 |
| 6 | 40..47 | 3 | ping[0..7] @ addr 3 | →1 |

### 7.2. Compute — 1 output mỗi 8 cycles, out_pos=P (P=0..19)

| Cycle offset (8P+) | a | a_d6 | sram_addr_en | shift_en | srw_din | ACC ctrl | out_valid |
|---------------------|---|------|--------------|----------|---------|----------|-----------|
| 0 | 0 | 0 | 0 | 0 | − | **RST** (oc 0..7) | 0 |
| 1 | 1 | 1 | 0 | 0 | − | ACC (ic=1) | 0 |
| 2 | 2 | 2 | 0 | 0 | − | ACC (ic=2) | 0 |
| 3 | 3 | 3 | 0 | 0 | − | ACC (ic=3) | 0 |
| 4 | 4 | 4 | 0 | 0 | − | ACC (ic=4) | 0 |
| 5 | 5 | 5 | 0 | 0 | − | ACC (ic=5) | 0 |
| 6 | 6 | 6 | **↑** addr=rp+1 | 0 | − | ACC (ic=6) | 0 |
| 7 | 7 | 7 | − | **1** | ping[0..7] | **OUT** (ic=7 + bias) | **1** |

> Cộng dồn 8 channels → biased = acc_final + bias_in → rescale (nb=7) → clamp → **ReLU** → pool.

### 7.3. Toàn bộ Conv4 timeline

| out_pos P | out_valid cy | bias_valid | relu_v cy | Notes |
|-----------|--------------|------------|-----------|-------|
| 0 | 47 | 48 | 51 | pad-left edge |
| 1 | 55 | 56 | 59 | |
| ... | ... | ... | ... | |
| 17 | 191 | 192 | 195 | |
| 18 | 199 | 200 | 203 | pad-right edge |
| 19 | 207 | 208 | 211 | last conv output |

### 7.4. Pool sequence (Conv4) — chi tiết đầy đủ

`pool_cnt` cycles 0,1,2,3,4 → fire pool_write.

| pool_cnt | relu_v cy | relu_out | max_reg update | pool_write | pong_addr | Pong bank A write |
|----------|-----------|----------|----------------|------------|-----------|---------------------|
| 0 | 51 | v0 | v0 (init) | 0 | − | − |
| 1 | 59 | v1 | max(v0,v1) | 0 | − | − |
| 2 | 67 | v2 | max(v0..v2) | 0 | − | − |
| 3 | 75 | v3 | max(v0..v3) | 0 | − | − |
| 4 | 83 | v4 | max(v0..v4) | **1** | 0 | A[oc][0] ← max(v0..v4) for oc=0..7 |
| 0 | 91 | v5 | v5 | 0 | − | − |
| 1 | 99 | v6 | max(v5,v6) | 0 | − | − |
| 2 | 107 | v7 | max(v5..v7) | 0 | − | − |
| 3 | 115 | v8 | max(v5..v8) | 0 | − | − |
| 4 | 123 | v9 | max(v5..v9) | **1** | 1 | A[oc][1] |
| ... | ... | ... | ... | ... | ... | ... |
| 4 | 163 | v14 | max(v10..v14) | **1** | 2 | A[oc][2] |
| ... | ... | ... | ... | ... | ... | ... |
| 4 | 203 | v19 | max(v15..v19) | **1** | 3 | A[oc][3] |

> Total pool_write = 4 (= OUT_LEN). Tất cả 8 channels ghi mỗi pool_write.

### 7.5. Layer_done

`pong_addr == 3 && pool_write` ~cy 203 → transition CONV4 → GAP_FC_S. `bank_sel` 1→0 → Pong A (vừa ghi) thành Ping cho GAP read.

---

## 8. GAP_FC_S — Sub-FSM (gap_fc_argmax.v)

**Đặc điểm**: 22 cycles total: GAP(6) + FC(10) + FLUSH(1) + ARGMAX(4) + DONE(1).
Đọc từ Ping bank A (Conv4 output, 4 entries × 8 channels). NB = 0 (no rescale), bias = 0.

### 8.1. GAP_SUB — Global Average Pooling (6 cycles)

`gap_acc[ch]` tích luỹ 4 samples (Conv4 OUT_LEN=4), chia 4 (>>2).

| `gap_step` | cycle | `gap_rd_addr` | `ping_dout[ch]` | gap_acc[ch] update | gap_reg[ch] |
|------------|-------|---------------|-----------------|--------------------|--------------|
| 0 | 0 | issue addr=0 | (cũ, ignored) | reset → 0 | − |
| 1 | 1 | issue addr=1 | x[ch][0] | acc += x[ch][0] | − |
| 2 | 2 | issue addr=2 | x[ch][1] | acc += x[ch][1] | − |
| 3 | 3 | issue addr=3 | x[ch][2] | acc += x[ch][2] | − |
| 4 | 4 | hold addr=3 | x[ch][3] | acc += x[ch][3] | − |
| 5 | 5 | − | − | − | gap_reg[ch] ← gap_acc[ch][9:2] (÷4) |

> 8 channels song song. Conv4 RELU_EN=1 → giá trị ∈ [0,127] → gap_acc ∈ [0,508] → /4 ∈ [0,127], no clamp needed.

### 8.2. FC_SUB — Fully Connected (10 cycles + 1 flush)

`fc_w[k*8 + i]` — 4 neurons × 8 inputs INT8.
Pipeline 3-stage: latch gap → multiply → accumulate.

| fc_step | Stage 1: latch | Stage 2: multiply | Stage 3: accumulate | prod_valid |
|---------|---------------|-------------------|--------------------|------------|
| 0 | − | − | clear acc, fc_w_idx=0 | 0 |
| 1 | fc_gap_pipe ← gap_reg[0]; fc_w_idx ← 0 | − | − | 0 |
| 2 | gap_reg[1]; idx ← 1 | prod[k] ← gap[0] × w[k][0] | − | 0 |
| 3 | gap_reg[2]; idx ← 2 | prod[k] ← gap[1] × w[k][1] | acc[k] += prod[k] (gap[0]×w[k][0]) | 1 |
| 4 | gap_reg[3]; idx ← 3 | prod[k] ← gap[2] × w[k][2] | acc[k] += prod[k] (gap[1]×w[k][1]) | 1 |
| 5 | gap_reg[4]; idx ← 4 | prod[k] ← gap[3] × w[k][3] | acc[k] += prod[k] (gap[2]×w[k][2]) | 1 |
| 6 | gap_reg[5]; idx ← 5 | prod[k] ← gap[4] × w[k][4] | acc[k] += prod[k] (gap[3]×w[k][3]) | 1 |
| 7 | gap_reg[6]; idx ← 6 | prod[k] ← gap[5] × w[k][5] | acc[k] += prod[k] (gap[4]×w[k][4]) | 1 |
| 8 | gap_reg[7]; idx ← 7 | prod[k] ← gap[6] × w[k][6] | acc[k] += prod[k] (gap[5]×w[k][5]) | 1 |
| 9 | hold; idx=7 | prod[k] ← gap[7] × w[k][7] | acc[k] += prod[k] (gap[6]×w[k][6]) | 1 |
| FC_FLUSH | − | − | acc[k] += prod[k] (gap[7]×w[k][7]) — last | 1 |

> After FC_FLUSH: fc_acc[0..3] hold INT32 dot-products. Move to ARGMAX_SUB.

### 8.3. ARGMAX_SUB (4 cycles)

| argmax_step | argmax_max | argmax_idx | Action |
|-------------|------------|------------|--------|
| 0 | fc_acc[0] | 0 | init |
| 1 | max(fc_acc[0..1]) | 0 or 1 | compare fc_acc[1] |
| 2 | max(fc_acc[0..2]) | 0..2 | compare fc_acc[2] |
| 3 | max(fc_acc[0..3]) | 0..3 | compare fc_acc[3] |

### 8.4. DONE_SUB (1 cycle)

| Signal | Action |
|--------|--------|
| `done` | 1 (pulse) |
| `result` | argmax_idx (latched in controller as `result[1:0]`) |
| FSM | GAP_FC_S → DONE_S |

---

## 9. DONE_S — Final state

| Signal | Value |
|--------|-------|
| `busy` | 0 |
| `done` (controller) | 1 then 0 next cycle (1-cy pulse) |
| `result[1:0]` | latched argmax class (0..3) |
| HPS reads | STATUS[1]=done_latched, STATUS[0]=busy, RESULT[1:0]=result |
| Next | Wait for HPS to write START register again, or rst |

---

## 10. Bảng tổng kết Timing toàn bộ inference

| Phase | Layer | Pre-fetch | Compute | Pool/Drain | Subtotal |
|-------|-------|-----------|---------|------------|----------|
| 1 | LOAD_INPUT | − | − | − | 1 cy (FSM transition) |
| 2 | CONV1 | 6 | 2500 | ~10 | ~2516 cy |
| 3 | CONV2 | 24 | 2000 | ~10 | ~2034 cy |
| 4 | CONV3 | 24 | 400 | ~10 | ~434 cy |
| 5 | CONV4 | 48 | 160 | ~10 | ~218 cy |
| 6 | GAP_FC_S | − | 22 | − | 22 cy |
| 7 | DONE_S | − | 1 | − | 1 cy |
| | **TOTAL** | | | | **~5226 cy** |

@ 100 MHz → **~52.3 µs per inference**.

---

## 11. Critical paths & timing concerns

| Stage | Path | Estimated delay | Status |
|-------|------|----------------|--------|
| S1 MULT | 8×8 → 16b (DSP18 hard block) | ~2 ns (DSP) | OK |
| S4 TREE | 18b + 16b adder (carry chain) | ~3 ns | OK |
| S5 ACC | 32b + 32b (registered) | ~3 ns | OK |
| **S5→S_bias** | **(acc + tree_sext) + bias_in — 2 cascaded 32b adders + MUX** | **~5-7 ns** | ⚠ Expected critical — verify TimeQuest |
| S6 RESCALE | (biased + round_add) >>> nb — 32b add + barrel shift | ~5 ns | OK |
| S7 CLAMP | 32b compare ×2 + MUX | ~3 ns | OK |
| w_packed MUX | 4-way case on layer_state, 40b output | ~2 ns | OK |
| Pong SRAM write | per-channel `we` gating + 64b din | ~1 ns | OK |

**Critical path mitigation** (nếu TimeQuest fail trên S5→S_bias):
- Thêm register `acc_final_r` giữa adder #1 và adder #2 → +1 cy latency.
- Cần đẩy `bias_valid` chain +1 stage tương ứng.

---

## 12. Ghi chú đặc biệt

### 12.1. Conv1 padding edge case

Conv1 có `sram_addr_en=1` mọi cycle (vì IN_CH=1, không có wait cycle cho a). Điều này khác Conv2/3/4 (chỉ phát addr tại `a==in_ch-2`). Vì SRAM 1-cy latency, dout đến cycle kế tiếp khi `t` đã tăng.

→ `pad_zero` cho Conv1 dùng `sram_rd_addr_in <= 2` (vs `< 2` cho layer khác).

### 12.2. acc_final combinational forward

```verilog
wire acc_final = (a_in == 0) ? tree_sext : (acc + tree_sext);
biased <= acc_final + bias_in;  // when out_valid
```

Khi `out_valid=1`, `a_d6 == in_ch-1` (channel cuối). Tại cùng cycle, `acc` register **chưa** update với tree_sext của channel cuối (non-blocking). → Forward combinational: `acc + tree_sext` để có giá trị đúng cuối.

### 12.3. Pool write all 8 channels

Mỗi `pool_write_r` fire đồng thời ở 8 cp_blocks. `pong_we[oc] = cp_pool_write[oc] && cp_en[oc]` gate per-channel. Conv1/2 chỉ ghi 4 channel (cp_en=0x0F), Conv3/4 ghi 8 (cp_en=0xFF).

### 12.4. Ping-Pong bank switching

| Layer | bank_sel (during) | Read from | Write to |
|-------|-------------------|-----------|----------|
| Conv1 | 0 (initial) | Input SRAM (special) | Pong = bank B (set-B) |
| Conv2 | 1 (toggled after Conv1) | Ping = bank B | Pong = bank A |
| Conv3 | 0 (toggled after Conv2) | Ping = bank A | Pong = bank B |
| Conv4 | 1 (toggled after Conv3) | Ping = bank B | Pong = bank A |
| GAP | 0 (toggled after Conv4) | Ping = bank A (Conv4 output) | − |

### 12.5. Weight ROM refresh

`$readmemh` chạy 1 lần lúc elaboration → weights cố định trong bitstream.
Để update weights mà không re-synth: dùng **Quartus In-System Memory Content Editor** (yêu cầu `ramstyle="M10K"/"MLAB"` đã có) hoặc thêm Avalon-MM write port (out of scope).

---

## 13. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---------|---------|
| `a` | Channel counter (0..IN_CH-1) |
| `a_d6` | `a` delayed 6 cycles, dùng trong S5 ACC |
| `t` | Output position counter |
| `rp` | Read pointer = t - 2 (padding offset) |
| `IN_CH` | Input channels (1/4/4/8) |
| `OUT_CH` | Output channels (4/4/8/8) |
| `IN_LEN` | Input sequence length per channel |
| `OUT_LEN` | Output length after pool (= IN_LEN/5) |
| `nb` | Rescale right-shift amount per layer |
| `cp_en` | Bitmask of active output channels |
| `bank_sel` | Ping/Pong bank selector (0/1) |
| `shift_en` | SRW shift trigger = (a == IN_CH-1) |
| `sram_addr_en` | SRAM read address issue trigger |
| `compute_en` | Pipeline enable (0 during pre-fetch) |
| `out_valid` | ACC output ready = ce_d6 && (a_d6==in_ch-1) |
| `pool_write` | Pool stage output strobe (cp_pool_write[oc]) |
| `pong_we[oc]` | Per-channel Pong write enable = pool_write && cp_en[oc] |
| `layer_done` | (pong_addr == out_len-1) && pool_write |
| `gap_step/fc_step/argmax_step` | Sub-FSM step counters |
| RST/ACC/OUT | ACC stage control mode |
| NOP | Pipeline idle (compute_en=0) |
| `w_packed[oc]` | 40-bit weight word for output channel oc |
| `b_cur[oc]` | INT32 bias for output channel oc, current layer |
