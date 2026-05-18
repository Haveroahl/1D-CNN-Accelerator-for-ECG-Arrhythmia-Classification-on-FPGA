# GAP / FC / Argmax Engine — Chi Tiết

## Tổng Quan

Engine độc lập, chạy sau Conv4 hoàn thành. Không dùng CP block pipeline.

**Input:** 8 Ping banks [0..7], mỗi bank 4 entries × INT8
(= Conv4 Pong sau bank_sel swap tại CONV4→GAP_FC transition)

**Quantization:**
```
GAP: gap[c] = sum(SRAM[c][0..3]) >> 2  (Conv4 ReLU → values ≥ 0)
FC:  NB=0 → không rescale; bias=0; logit = raw INT32 acc
```

## Kiến Trúc

```
Pong SRAM[0..7] (8 banks × 4 entries × INT8)
        ↓ đọc song song, 1 addr chung
┌─────────────────────────────┐
│  GAP: gap_acc[0..7] 10-bit  │  ← 4 samples × 8 channels song song
│       gap_reg[0..7] INT8    │  ← >>> 2 sau khi tích lũy xong
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│  FC: fc_w[4][8] INT8 regs   │  ← register array, 0-cycle latency
│     4 multipliers parallel  │  ← gap[i] × w[k][i] cho k=0..3
│     fc_acc[0..3] INT32      │  ← accumulate 8 inputs per output
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│  Argmax: rolling max        │  ← 4 sequential INT32 comparisons
│  result[1:0]                │
└─────────────────────────────┘
```

**FC weights:** Register array (không phải ROM) — 32 weights × 8-bit = 256 bits, quá nhỏ cho M10K (9216 bits). 0-cycle latency.

```verilog
reg signed [7:0] fc_w [0:31];   // 1D flat, addr = k*8 + i
initial $readmemh("fc_weights.hex", fc_w);
```

## GAP Phase (6 cycles)

```
gap_step  gap_rd_addr  pong_dout valid   Action
──────────────────────────────────────────────────────────────
   0          0              —           gap_acc[ch] ← 0; issue addr=0
   1          1          SRAM[ch][0]     gap_acc[ch] += sign_ext(dout[ch]); issue addr=1
   2          2          SRAM[ch][1]     gap_acc[ch] += sign_ext(dout[ch]); issue addr=2
   3          3          SRAM[ch][2]     gap_acc[ch] += sign_ext(dout[ch]); issue addr=3
   4          —          SRAM[ch][3]     gap_acc[ch] += sign_ext(dout[ch])
   5          —              —           gap_reg[ch] ← gap_acc[ch][9:2]   (>>> 2)
```

- 8 channels tích lũy song song mỗi bước
- gap_acc: 10-bit (max 4×127=508, không overflow)
- Không cần clamp vì Conv4 ReLU đảm bảo ≥ 0 → `[9:2]` = floor division đúng

```verilog
reg signed [9:0]  gap_acc [0:7];
reg signed [7:0]  gap_reg [0:7];
reg [1:0]         gap_rd_addr;    // 2-bit, broadcast to ping_pong_sram
// gap_rd_addr exposed as [8:0] wire (sign-extended) to match pp_sram port
```

**Note:** `gap_rd_addr` trong gap_fc_argmax.v là [8:0] để match port `ping_pong_sram.rd_addr[8:0]`. Chỉ [1:0] thực sự dùng.

## FC Phase (10 cycles + 1 flush)

```
fc_step  gap input latch   fc_prod compute        fc_acc accumulate
──────────────────────────────────────────────────────────────────────
   0     —                 —                      fc_acc[k] ← 0
   1     gap[0]            —                      —
   2     gap[1]            gap[0]×w[k][0]         —
   3     gap[2]            gap[1]×w[k][1]         acc += gap[0]×w[k][0]
   4     gap[3]            gap[2]×w[k][2]         acc += gap[1]×w[k][1]
   5     gap[4]            gap[3]×w[k][3]         acc += gap[2]×w[k][2]
   6     gap[5]            gap[4]×w[k][4]         acc += gap[3]×w[k][3]
   7     gap[6]            gap[5]×w[k][5]         acc += gap[4]×w[k][4]
   8     gap[7]            gap[6]×w[k][6]         acc += gap[5]×w[k][5]
   9     — (hold gap[7])   gap[7]×w[k][7]         acc += gap[6]×w[k][6]
FC_FLUSH —                 —                      acc += gap[7]×w[k][7]
```

Pipeline 2-cycle latency: latch gap → multiply (1cy) → accumulate (1cy)
4 multipliers tính k=0..3 song song mỗi bước.

## Argmax Phase (4 cycles)

```
argmax_step  Action                              max_val       max_idx
────────────────────────────────────────────────────────────────────
    0        max_val=fc_acc[0]; max_idx=2'b00    fc_acc[0]     0
    1        if fc_acc[1]>max_val: update        max(0,1)      winner
    2        if fc_acc[2]>max_val: update        max(0,1,2)    winner
    3        if fc_acc[3]>max_val: update        max(0..3)     winner
```

Tie-breaking: strict `>` → giữ index thấp hơn khi bằng nhau (khớp `torch.argmax()`).

## Cycle Count Summary

```
Phase       Cycles   Ghi chú
────────────────────────────────────────────────────────────
GAP         6        clear + 4 reads (1-cy SRAM latency) + shift
FC          10       init + 8 multiply-acc steps
FC_FLUSH    1        drain last multiply product
ARGMAX      4        4 sequential INT32 comparisons
DONE        1        latch result, assert done
────────────────────────────────────────────────────────────
Total       22 cycles @ 100MHz = 0.22 µs
```

## Transition CONV4 → GAP_FC

```
Cycle N    : Conv4 layer_done=1 (pool_write cuối, pong_addr=3)
Cycle N+1  : layer_state → GAP_FC_S
             bank_sel ← ~bank_sel  ← Conv4 Pong → Ping mới (GAP đọc từ đây)
             cp_en ← 8'h00         ← tắt CP blocks
             fc_sub_state ← GAP_SUB
             gap_step ← 0
Cycle N+2  : GAP step 0: gap_acc ← 0, issue gap_rd_addr=0
Cycle N+3  : pong_dout[ch] = SRAM[ch][0] valid, gap_acc[ch] += dout
...
```

**GAP đọc từ Ping bank mới** (= Conv4 Pong sau swap). Tín hiệu:
```verilog
wire [8:0] pp_rd_addr = (ctrl_layer_state == 3'd6) ? gap_rd_addr
                                                    : cp_sram_rd_addr[8:0];
```

## Output

```
result[1:0]: 0=AFIB, 1=GSVT, 2=SB, 3=SR
done: 1-cycle pulse tại DONE_SUB transition → main FSM asserts done=1
```
