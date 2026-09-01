# CNN Controller FSM — Chi Tiết

## FSM States

```
State        Encoding  Transition condition            Next state
──────────────────────────────────────────────────────────────────────
IDLE         3'd0      start=1                         LOAD_INPUT
LOAD_INPUT   3'd1      (1 cycle only)                  CONV1
CONV1        3'd2      layer_done=1                    CONV2
CONV2        3'd3      layer_done=1                    CONV3
CONV3        3'd4      layer_done=1                    CONV4
CONV4        3'd5      layer_done=1                    GAP_FC_S
GAP_FC_S     3'd6      fc_sub_state==DONE_SUB          DONE_S
DONE_S       3'd7      rst=1                           IDLE
```

```verilog
wire layer_done = (pong_addr == out_len - 12'd1) && pool_write;
```

**pool_write representative:** `cp_pool_write = cp_pong_we[0]` (tất cả active blocks ghi cùng lúc)

## Layer Parameters Per State

```
State    in_ch  out_len  nb   relu_en  cp_en   bank_sel
──────────────────────────────────────────────────────────
CONV1      1      500    8      0      8'h0F     0
CONV2      4      100    7      0      8'h0F     1
CONV3      4       20    6      0      8'hFF     0
CONV4      8        4    7      1      8'hFF     1
GAP_FC_S   —        —    —      —      8'h00     —
```

**NB constants (ningba, re-train 2026-07-28):** `NB1=4'd8, NB2=4'd7, NB3=4'd6, NB4=4'd7`
— via `cfg_nb_of(li)` in `cnn_controller.v` (ROM single-load build; Chapman cũ dùng NB2=6,
xem [PROJECT.md](../../PROJECT.md) mục "nb per layer").

## Derived Signals

```verilog
assign shift_en    = (a == in_ch - 4'd1);
```

Controller **chỉ** xuất `shift_en` và `t`. Không có port `sram_addr_en` — địa chỉ đọc
được tính hoàn toàn trong `cp_engine`.

**sram_rd_addr** (derived trong cp_engine từ `t`):
```verilog
assign sram_rd_addr = (sram_rd_addr_in >= 12'd2) ? (sram_rd_addr_in - 12'd2) : 12'd0;
```
Controller xuất `t`; cp_engine tính `t-2` làm địa chỉ thực tế (padding offset).

## Counter Logic (CONV1..4)

```verilog
// a counter: 0..in_ch-1
if (a == in_ch - 1) a <= 0; else a <= a + 1;

// t counter: tăng khi shift_en
if (shift_en) t <= t + 1;

// pong_addr: tăng mỗi pool_write
if (pool_write) pong_addr <= pong_addr + 1;

// pre-fetch → compute_en (gate by !srw_rst để bỏ qua cycle 0 của Conv1)
if (!compute_en && shift_en && !srw_rst) begin
    if (prefetch_cnt == 3'd4) compute_en <= 1;   // 5 real SRW shifts done
    else                      prefetch_cnt <= prefetch_cnt + 1;
end
```

## Pre-fetch Timing

```
Pre-fetch shifts cần thiết trước compute_en=1:
  Cần 5 SRW shifts để SRW = [data[2], data[1], data[0], 0, 0]
    → MUX a=0 → [0, 0, data[0], data[1], data[2]] = output position p=0 ✓
  Trong đó:
    - 2 pad shifts (slot0 ← 0 khi t < 2)
    - 3 data shifts (slot0 ← data[0..2])

  Conv1 (IN_CH=1): 5 shifts × 1 cy = 5 cycles (+ cy 0 với srw_rst không count)
  Conv2/3 (IN_CH=4): 5 shifts × 4 cy = 20 cycles
  Conv4 (IN_CH=8): 5 shifts × 8 cy = 40 cycles
```

## Layer Transition Sequence

```
Cycle N    : layer_done=1 (pool_write cuối, pong_addr=out_len-1)
Cycle N+1  : FSM chuyển state, reset tất cả counters:
               bank_sel ← ~bank_sel
               a, t, pong_addr ← 0
               prefetch_cnt ← 0
               compute_en ← 0
               srw_rst ← 1'b1  (1 cycle pulse)
               in_ch, out_len, nb, relu_en, cp_en ← giá trị mới
Cycle N+2  : srw_rst=0, pre-fetch bắt đầu
Cycle N+...: compute_en=1 sau khi đếm đủ 5 SRW shifts (prefetch_cnt 0→1→2→3→4)
             - Conv1 (IN_CH=1): 5 cycles
             - Conv2/3 (IN_CH=4): 20 cycles
             - Conv4 (IN_CH=8): 40 cycles
```

**Signals không cần reset:** acc, a_d1..a_d5, shifted, clamped — compute_en=0 bảo vệ downstream (out_valid=0).

**Note**: cp_block gates ACC bằng `a_d5/ce_d5` (5-stage delay). Pipeline thực: mux_comb → mux_s1 → prod → sum01,23 → sum0123 → tree_out (5 registers); acc register update edge ngay sau tree → cần ce_d5 để align.

> `acc_final_r`/`acc_final_v` (S5b) đã bị gộp vào `S_bias` (`biased`/`bias_valid`,
> qua `out_valid_d1`) — xem [cp_pipeline.md](cp_pipeline.md) mục "round_add". Bảng
> dưới dùng tên gọi khái niệm cũ (out_valid) vẫn đúng, chỉ tên thanh ghi nội bộ đổi.

## Per-Layer Counter Timing

Bảng dưới hiển thị `a` tại cycle drive (cp_engine input) cùng cột `a_d5` đến cp_block ở edge sau 5 cycles. ACC update (RST/ACC/OUT) xảy ra tại cp_block khi `a_d5` đạt giá trị tương ứng — không nằm cùng cycle drive `a`.

### Conv1 — IN_CH=1

```
Mỗi cycle: shift_en=1 (a luôn = 0 = in_ch-1)
a_d5 = 0 mỗi cycle → mỗi edge cp_block: RST acc + out_valid=1 mỗi cycle
pool_write: pulse mỗi 5 cycles (5 relu_v)
```

### Conv2/3 — IN_CH=4

```
Cycle (drive)  a  shift_en  | edge cp_block (drive +5) a_d5  ACC
 4k+0          0     -      |                            0    RST
 4k+1          1     -      |                            1    ACC
 4k+2          2     -      |                            2    ACC
 4k+3          3     ↑      |                            3    OUT → out_valid
```

### Conv4 — IN_CH=8 (chuẩn tham chiếu)

```
Cycle (drive)  a  shift_en  | edge cp_block (drive +5) a_d5  ACC
 8k+0          0     -      |                            0    RST
 8k+1          1     -      |                            1    ACC
 8k+2          2     -      |                            2    ACC
 8k+3          3     -      |                            3    ACC
 8k+4          4     -      |                            4    ACC
 8k+5          5     -      |                            5    ACC
 8k+6          6     -      |                            6    ACC
 8k+7          7     ↑      |                            7    OUT → out_valid
```

## GAP_FC Sub-States

```
GAP_SUB    = 3'd1   gap_step 0..5  (6 cycles)
FC_SUB     = 3'd2   fc_step  0..9  (10 cycles)
FC_FLUSH_S = 3'd3   1 cycle        (drain last multiply)
ARGMAX_SUB = 3'd4   argmax_step 0..3  (4 cycles)
DONE_SUB   = 3'd5   → main FSM → DONE_S, done=1
```

**Transition trong controller:**
```verilog
GAP_SUB:    if (gap_step == 4'd5) fc_sub_state <= FC_SUB;
FC_SUB:     if (fc_step  == 4'd9) fc_sub_state <= FC_FLUSH_S;
FC_FLUSH_S: fc_sub_state <= ARGMAX_SUB;
ARGMAX_SUB: if (argmax_step == 2'd3) fc_sub_state <= DONE_SUB;
DONE_SUB:   → main FSM handles
```
