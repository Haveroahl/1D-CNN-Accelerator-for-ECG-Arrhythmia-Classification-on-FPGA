# CP Block Pipeline — Chi Tiết

## Pipeline Stages (S1–S9)

```
Stage    Module              Type    Latency   Ghi chú
──────────────────────────────────────────────────────────────────
S1       5×MULT              reg     1 cy      DSP18, 8×8 signed → 16-bit
S2       ADDER stage-1       reg     1 cy      sum01, sum23, delay prod[4]
S3       ADDER stage-2       reg     1 cy      sum0123, delay prod[4]
S4       ADDER stage-3       reg     1 cy      tree_out (20-bit)
S5       ACC × IN_CH         reg     IN_CH cy  RST khi a_d6=0, ACC khi a_d6=IN_CH-1
S5b      ACC_FINAL           reg     1 cy      acc + tree_sext → acc_final_r (1 adder)
S_bias   +BIAS               reg     1 cy      acc_final_r + bias_in (1 adder)
S6       RESCALE stage-1     reg     1 cy      (biased + round_add) >>> nb
S7       RESCALE stage-2     reg     1 cy      clamp [-127, 127]
S8       ReLU                reg     1 cy      chỉ Conv4 (relu_en=1)
S9       MaxPool comparator  reg     1 cy/hit  rolling max, chốt sau 5 relu_v
──────────────────────────────────────────────────────────────────
Total    IN_CH + 9 cycles/output_position (steady-state)
```

**Delay chain:** MUX_reg(1) + WROM_reg(1) + MULT(1) + TREE(3) = 6 cycles → a_d6, inch_d6, ce_d6

**out_valid:** `compute_en_in && (a_in == in_ch - 1)`  (dùng a_d6, ce_d6 từ cp_engine)

**Valid chain:** `out_valid → acc_final_v → bias_valid → rescale_v1 → rescale_v2 → relu_v`

## round_add — Critical Path Fix

S6 dùng `round_add` precomputed as wire (giảm critical path từ ~4 ops → ~2 ops):

```verilog
wire signed [31:0] round_add;
assign round_add = (nb > 5'd0) ? (32'sd1 << (nb - 5'd1)) : 32'sd0;

// S6: chỉ còn add + shift trên register-to-register path
shifted <= (biased + round_add) >>> nb;
```

Vì `nb` cố định mỗi layer, Quartus constant-fold `round_add` sau synthesis.

## Latency và Throughput Per Layer

```
Layer  IN_CH  Latency/output  Throughput (steady-state)
Conv1    1      10 cycles       1 output / 1 cy
Conv2    4      13 cycles       1 output / 4 cy
Conv3    4      13 cycles       1 output / 4 cy
Conv4    8      17 cycles       1 output / 8 cy
```

**Tổng cycles per layer (xấp xỉ):**
```
Conv1:  2500 × 1 = 2500 cy
Conv2:   500 × 4 = 2000 cy  (+ 20 pre-fetch)
Conv3:   100 × 4 =  400 cy
Conv4:    20 × 8 =  160 cy
Total  ~5060 cy @ 100MHz ≈ 50.6 µs/inference  (8 CP blocks parallel)
```

## SRW — Shift Register Window

```
8 SRW[ch][tap]  (ch=0..7, tap=0..4; tap0=newest, tap4=oldest)

shift_en = (a == in_ch - 1)  → shift tất cả SRW, nhận srw_din[ch]
srw_rst  = 1 cycle pulse      → clear về 0 (layer transition)

Zero-padding:
  Conv1 (in_ch==1): pad khi sram_rd_addr_in <= 12'd2
  Conv2..4         : pad khi sram_rd_addr_in <  12'd2
```

## MaxPool Rolling Comparator

```verilog
always @(posedge clk) begin
    pool_write_r <= 1'b0;
    if (rst || pool_rst) begin
        pool_cnt <= 3'd0; pool_write_r <= 1'b0;
    end else if (relu_v) begin
        if (pool_cnt == 3'd0)         max_reg <= relu_out;
        else if (relu_out > max_reg)  max_reg <= relu_out;

        if (pool_cnt == 3'd4) begin
            pool_cnt     <= 3'd0;
            pool_write_r <= 1'b1;
        end else
            pool_cnt <= pool_cnt + 3'd1;
    end
end
```

**pool_write gating:** `pong_we[oc] = cp_pool_write[oc] && cp_en[oc]` (trong cp_engine)

## Timing Diagram (IN_CH=8, out_pos=0)

```
Cycle   S1(mult) S2(add1) S3(add2) S4(tree) a_d6  ACC       S5b(af) S_bias  S6      S7      S8    POOL
 N+0    ch0×w
 N+1    ch1×w    Σ01,23
 N+2    ch2×w    Σ01,23   Σ0123
 N+3    ch3×w    Σ01,23   Σ0123    tree_ch0  0     RST←t0
 N+4    ch4×w    Σ01,23   Σ0123    tree_ch1  1     +=tree1
 N+5    ch5×w    Σ01,23   Σ0123    tree_ch2  2     +=tree2
 N+6    ch6×w    Σ01,23   Σ0123    tree_ch3  3     +=tree3
 N+7    ch7×w    Σ01,23   Σ0123    tree_ch4  4     +=tree4
 N+8    ch0'×w   Σ01,23   Σ0123    tree_ch5  5     +=tree5
 N+9    ch1'×w   Σ01,23   Σ0123    tree_ch6  6     +=tree6
 N+10   ch2'×w   Σ01,23   Σ0123    tree_ch7  7     (OUT)     af_r←  
 N+11   ch3'×w   ...               tree_ch0' 0     RST(pos1)          bias←   
 N+12                                                                   bias    shift
 N+13                                                                           clamp
 N+14                                                                           relu
 N+18                                                                                   pool_write★
★ pool_write sau 5 relu_v → ghi Pong SRAM
af_r = acc_final_r (S5b); bias = biased (S_bias)
```

## Bảng Chân Trị SRW (Conv4, K=5, pad=2)

```
out_pos=0: [0,    0,    x[0], x[1], x[2]] → padding ✓
out_pos=1: [0,    x[0], x[1], x[2], x[3]] → padding ✓
out_pos=2: [x[0], x[1], x[2], x[3], x[4]] → full ✓
```

## Broadcast Architecture

```
mux_s1 (40-bit, registered) ──broadcast──▶ 8 CP blocks
                                              │ taps_in giống nhau
                                              │ w[39:0] khác nhau mỗi block
                                              ▼
                                        w_packed[oc] từ per-layer ROM
                                        Conv1: w_rom_conv1[oc]
                                        Conv2: w_rom_conv2[oc*4 + a]
                                        Conv3: w_rom_conv3[oc*4 + a]
                                        Conv4: w_rom_conv4[oc*8 + a]
```

## Weight Read Pipeline (FF array, async read + 1 FF stage)

```
Cycle N  : a → w_comb[oc] (combinational: 4:1 layer MUX + 8:1 ic MUX, ~2 LUT levels)
Cycle N+1: w_packed[oc] ← w_comb[oc] (1 FF stage) — aligns với mux_s1
Cycle N+2: S1 MULT dùng mux_s1 và w_packed

FF array storage (no ramstyle — async read, no port replication):
  w_rom_conv1[0:3]   40b/entry  FF  (4oc × 1ic)
  w_rom_conv2[0:15]  40b/entry  FF  (4oc × 4ic)
  w_rom_conv3[0:31]  40b/entry  FF  (8oc × 4ic)
  w_rom_conv4[0:63]  40b/entry  FF  (8oc × 8ic)
  Total: ~185 FF — 0.1% Cyclone V budget
  Entry format: {tap4[7:0], tap3[7:0], tap2[7:0], tap1[7:0], tap0[7:0]}
```
