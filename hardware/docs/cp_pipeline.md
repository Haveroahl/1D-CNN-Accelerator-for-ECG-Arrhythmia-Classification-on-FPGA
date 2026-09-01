# CP Block Pipeline — Chi Tiết

> Bảng port/interface của các module (cp_mac/accres/pool/engine/…): [module_interfaces.md](module_interfaces.md).

## Pipeline Stages (S1–S9)

```
Stage    Module              Type    Latency   Ghi chú
──────────────────────────────────────────────────────────────────
S1       5×MULT              reg     1 cy      DSP18, 8×8 signed → 16-bit
S2       ADDER stage-1       reg     1 cy      sum01, sum23, delay prod[4]
S3       ADDER stage-2       reg     1 cy      sum0123, delay prod[4]
S4       ADDER stage-3       reg     1 cy      tree_out (20-bit)
S5       ACC × IN_CH         reg     IN_CH cy  RST(init) khi a_in(=a_d5)=0 → acc=tree_sext+bias_in+round_add; else acc+=tree_sext
S_bias   delay-then-capture  reg     1 cy      `biased <= acc` khi `out_valid_d1` (KHÔNG có adder — bias+round đã fold ở S5; thay thế cặp S5b+ACC_FINAL cũ, cùng 2-cycle depth từ acc→biased, tiết kiệm 1 adder 32-bit)
S6       RESCALE stage-1     reg     1 cy      biased >>> nb (pure shift; round_add folded into acc-init tại S5)
S7       RESCALE stage-2     reg     1 cy      clamp [-127, 127]
S8       ReLU                reg     1 cy      chỉ Conv4 (relu_en=1)
S9       MaxPool comparator  reg     1 cy/hit  rolling max, chốt sau 5 relu_v
──────────────────────────────────────────────────────────────────
Total    IN_CH + 8 cycles/output_position (steady-state; S5b gộp vào S_bias, giảm 1 stage so với bản cũ)
```

> **Lịch sử đổi tên (commit 369f200 "fold S5b into S_bias"):** bản cũ có 2 stage
> riêng `S5b (ACC_FINAL, acc_final_r/acc_final_v)` rồi mới tới `S_bias`. RTL hiện
> tại (`cp_accumulate_rescale.v`) gộp 2 stage này làm 1: `S_bias` chỉ còn
> delay-then-capture (`out_valid_d1 → bias_valid`, `biased <= acc`), không còn
> adder — vì `acc` tại cycle `out_valid_d1` đã chứa đúng tổng hoàn chỉnh (S5 cộng
> dồn `tree_sext` mỗi cycle kể cả cycle cuối). Bit-exact và cùng tổng pipeline
> depth (2 cycle từ acc→biased), tiết kiệm 1 adder 32-bit trên critical path.

**Delay chain:** mux_s1(1) + MULT(1) + TREE(3) = **5 cycles** → `a_d5, inch_d5, ce_d5` feed cp_block (`a_in, in_ch, compute_en_in`).


**out_valid:** `compute_en_in && (a_in == in_ch - 1)` — `a_in == a_d5`, `compute_en_in == ce_d5`. RST(init) acc khi `a_in == 0` (cộng thêm bias+round_add); `S_bias` capture `acc` 1 cycle sau `out_valid` (qua `out_valid_d1`).

**Valid chain:** `out_valid → out_valid_d1 → bias_valid → rescale_v1 → rescale_v2 → relu_v`

## round_add — Critical Path Fix (folded into acc-init)

`round_add` (round-half-up) **và** `bias` được fold vào acc-init term (`a_in==0`) ở
S5, nên S6 chỉ còn 1 barrel shift thuần — cả +bias lẫn +round rời khỏi critical path:

```verilog
wire signed [31:0] round_add;
assign round_add = (nb > 4'd0) ? (32'sd1 << (nb - 4'd1)) : 32'sd0;

// S5: fold bias + round_add vào init khi a_in==0
if (a_in == 0) acc <= tree_sext + bias_in + round_add;
else           acc <= acc + tree_sext;

// S_bias: capture, không cộng gì thêm (bias+round đã ở trong acc)
if (out_valid_d1) biased <= acc;

// S6: pure arithmetic shift (round đã cộng ở acc-init)
shifted <= biased >>> nb;
```

Numerically identical to `(acc + bias + round) >>> nb` (round-half-up, signed). Vì `nb`
cố định mỗi layer, Quartus constant-fold `round_add`. Chi tiết cycle từng submodule:
[cp_submodule_timing.md](cp_submodule_timing.md).

## Latency và Throughput Per Layer

```
Layer  IN_CH  nb  Latency/output  Throughput (steady-state)
Conv1    1     8    9 cycles       1 output / 1 cy
Conv2    4     7   12 cycles       1 output / 4 cy
Conv3    4     6   12 cycles       1 output / 4 cy
Conv4    8     7   16 cycles       1 output / 8 cy
```
(Latency giảm 1 cycle so với bản trước fold S5b→S_bias; đo thực tế xem
[cp_submodule_timing.md](cp_submodule_timing.md).)

**Tổng cycles per layer (xấp xỉ):**
```
Conv1:  2500 × 1 = 2500 cy
Conv2:   500 × 4 = 2000 cy
Conv3:   100 × 4 =  400 cy
Conv4:    20 × 8 =  160 cy
GAP/FC/Argmax    =   22 cy
Layer transitions + pipeline flush ≈ 134 cy
Total measured (testbench $time)   = 5216 cy ≈ 52.16 µs @ 100 MHz
Throughput                         ≈ 19,200 inference/s
```

## SRW — Shift Register Window

```
8 SRW[ch][tap]  (ch=0..7, tap=0..4; tap0=newest, tap4=oldest)

shift_en = (a == in_ch - 1)  → shift tất cả SRW, nhận srw_din[ch]
srw_rst  = 1 cycle pulse      → clear về 0 (layer transition)

Zero-padding (đồng nhất mọi layer, K=5 pad=2):
  pad_zero_pre = (sram_rd_addr_in < 12'd2)              // front pad (negative addr)
              || (sram_rd_addr_in >= in_len + 12'd2);   // back pad  (out of valid range)
  pad_zero_r   = registered 1 cy → align với SRAM 1-cy synchronous read latency
                 (reset = 1 khi srw_rst).
```

## MaxPool Rolling Comparator

```verilog
always @(posedge clk) begin
    pool_write_r <= 1'b0;
    if (rst || pool_rst) begin
        pool_cnt <= 3'd0; pool_write_r <= 1'b0;
    end else if (relu_v && compute_en_in) begin
        // gated bởi compute_en_in (=ce_d5) để loại junk từ SRW priming phase
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
Anchor: cycle N = a==0 at mux_comb (= a_d5==0 at edge cy N+5).

Cycle   S1(mult) S2(add1) S3(add2) S4(tree) a_d5  ACC              S_bias  S6      S7      S8    POOL
 N+0    ch0×w
 N+1    ch1×w    Σ01,23
 N+2    ch2×w    Σ01,23   Σ0123
 N+3    ch3×w    Σ01,23   Σ0123    tree_ch0
 N+4    ch4×w    Σ01,23   Σ0123    tree_ch1
 N+5    ch5×w    Σ01,23   Σ0123    tree_ch2  0     init←tree_ch0+bias+round
 N+6    ch6×w    Σ01,23   Σ0123    tree_ch3  1     +=tree_ch1
 N+7    ch7×w    Σ01,23   Σ0123    tree_ch4  2     +=tree_ch2
 N+8    ch0'×w   Σ01,23   Σ0123    tree_ch5  3     +=tree_ch3
 N+9    ch1'×w   Σ01,23   Σ0123    tree_ch6  4     +=tree_ch4
 N+10   ch2'×w   Σ01,23   Σ0123    tree_ch7  5     +=tree_ch5
 N+11   ch3'×w   ...               tree_ch0' 6     +=tree_ch6
 N+12   ch4'×w                     tree_ch1' 7=IC-1 acc←acc+tree_ch7 (out_valid↑)
 N+13                                                                biased←acc (out_valid_d1↑, không adder)
 N+14                                                                        shifted←biased>>>nb (round folded at S5)
 N+15                                                                                clamped
 N+16                                                                                        relu_v↑ ─┐
 ...     (sau đủ 5 relu_v cho window)                                                                 │
                                                                                          pool_write★
★ pool_write: rolling comparator chốt sau 5 relu_v liên tiếp → ghi Pong SRAM.
biased = S_bias output (capture của acc, không cộng thêm gì — bias+round đã fold ở S5 init).
Khoảng cách giữa các relu_v = IN_CH cycles (1/4/4/8). So với bản cũ (S5b riêng), pipeline
ngắn hơn 1 cycle vì S_bias không còn là 1 adder riêng mà chỉ capture.
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
