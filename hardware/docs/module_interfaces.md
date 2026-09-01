# Module Interfaces — ECG CNN Accelerator (Compute Core)

Tài liệu tham chiếu **interface** của 11 module trong compute core (không bao gồm
bus adapter / top wrapper / UART). Mọi width và tên tín hiệu lấy **verbatim** từ
**`hardware/RTL/`** — bản ROM single-load, bản chính luận văn (bỏ qua
`hardware/RTL/txt/` — snapshot cũ).

> ⚠️ **Phạm vi = `hardware/RTL/` (ROM), KHÔNG phải `hardware/RTL_weight/` (production
> weight-load + runtime topology config).** Hai bản khác nhau đáng kể ở
> `cp_weight_store` (117 vs 211 dòng), `cp_engine` (217 vs 255 dòng),
> `gap_fc_argmax` (74 vs 95 dòng), `ecg_core` (166 vs 199 dòng): bản `RTL_weight/`
> có thêm toàn bộ port `cfg_*` (runtime topology), `w_wr_*`/`b_wr_*`/`fcw_wr_*`
> (Avalon bus write cho weight/bias/FC), và `out_ch_mask`. **Bản `RTL/` không có
> bất kỳ port nào trong số này** — topology Chapman/ningba hard-code trong
> `cnn_controller.v` (`cfg_in_ch_of`/`cfg_cp_en_of`/`cfg_nb_of` là **function** nội
> bộ, không phải port bus), weight nạp 1 lần lúc elaboration qua `$readmemh`.
> Xem [System_Design.md](../System_Design.md) mục "Runtime-Reconfigurable Topology"
> cho bản `RTL_weight/`.

**Cách đọc bảng:** mỗi hàng = 1 port. Cột `Dir` (input/output), `Width` (verbatim,
`1` = 1-bit scalar), `Name`, `Group` (nhóm theo cách file gom), `Purpose`.

---

## Phân cấp (ecg_core subtree)

```
ecg_core                              — bus-agnostic compute core
├── cp_engine                         — 8 CP block song song + SRW + tap MUX + addr-gen
│   ├── cp_weight_store               — 4 FF-array ROM per-layer (ROM single-load, không bus write)
│   └── cp_block × 8  (oc = 0..7)     — 1 output channel mỗi block
│       ├── cp_mac                    — S1→S4  MAC + adder tree
│       ├── cp_accumulate_rescale     — S5→S8  acc (fold bias+round) → S_bias → rescale → ReLU
│       └── cp_pool                   — S9     MaxPool
├── ping_pong_sram                    — feature-map giữa các layer (Ping đọc / Pong ghi)
├── gap_fc_argmax                     — GAP → FC → Argmax (wrapper của gap_unit/fc_unit/argmax_unit)
└── cnn_controller                    — FSM thống nhất Conv1..4 + GAP/FC, topology hard-code
```

> **Lưu ý phân cấp:** `input_sram` **KHÔNG** nằm trong `ecg_core` — nó ở **wrapper**
> (`ecg_accelerator_top.v`). Core chỉ *đọc* input SRAM qua cặp `input_rd_addr` →
> `input_dout` (1-cycle latency). Ranh giới này tách "compute latency" (start→done)
> khỏi pha host nạp input. `input_sram.v` vẫn được tài liệu hoá dưới đây vì nó
> thuộc datapath.

---

## 1. `cp_mac` — MAC datapath (S1→S4)

Conv MAC: 5 signed multipliers → adder tree 3 stage → `tree_out`. Feed-forward
thuần, dot-product 5 tap × 5 weight, latency 4 cycle. Width tăng 16→17→18→20
(sign-extend) để không tràn khi cộng 5 tích.

**Parameters:** none.

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input  | `1`      | `clk`      | clock | clock |
| input  | `[39:0]` | `x_in`     | data  | 5 window samples, packed 5×8b |
| input  | `[39:0]` | `w`        | data  | 5 weights, packed 5×8b |
| output signed | `[19:0]` | `tree_out` | data | adder-tree output (sign-extended Σ 5 tích) |

---

## 2. `cp_accumulate_rescale` — accumulate + bias + rescale + ReLU (S5→S8)

Nhận `tree_out`, sinh 1 activation INT8. `bias` và `round_add` (round-half-up) được
**fold vào acc-init** (`a_in==0`) ở S5 → S6 chỉ còn 1 phép `>>> nb` thuần. `S_bias`
là delay-then-capture thuần (`biased <= acc` khi `out_valid_d1`, KHÔNG có adder —
thay thế cặp S5b(`acc_final_r`)+S_bias cũ, xem [cp_pipeline.md](cp_pipeline.md) và
[cp_submodule_timing.md](cp_submodule_timing.md)). S7 clamp [−127,127], S8 ReLU
(chỉ Conv4).

**Parameters:** none.

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input | `1`      | `clk`           | clock/reset | clock |
| input | `1`      | `rst`           | clock/reset | reset |
| input | `1`      | `pool_rst`      | clock/reset | pool/layer-transition reset |
| input signed | `[19:0]` | `tree_out` | data (MAC in) | từ cp_mac |
| input signed | `[31:0]` | `bias_in`  | config | INT32 bias (từ cp_weight_store) |
| input | `[3:0]`  | `a_in`          | control | channel counter delayed 5 cy (a_d5) |
| input | `[3:0]`  | `in_ch`         | control | IN_CH layer hiện tại (1/4/4/8) |
| input | `1`      | `compute_en_in` | control | pipeline enable delayed 5 cy (ce_d5) |
| input | `[3:0]`  | `nb`            | control | rescale shift (0..15; max dùng = 8) |
| input | `1`      | `relu_en`       | control | 1 = Conv4 only |
| output signed | `[7:0]` | `relu_out` | data (to pool) | INT8 activation |
| output | `1`     | `relu_v`        | data (to pool) | valid |

---

## 3. `cp_pool` — MaxPool rolling comparator (S9)

Rolling max qua window 5 sample valid. `pool_cnt` đếm 0..4; ở sample thứ 5 phát
`pool_write` rồi reset. Gate bằng `relu_v && compute_en_in` để loại junk từ pha SRW
priming.

**Parameters:** none.

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input | `1`     | `clk`           | clock/reset | clock |
| input | `1`     | `rst`           | clock/reset | reset |
| input | `1`     | `pool_rst`      | clock/reset | pool/layer reset |
| input signed | `[7:0]` | `relu_out` | data in | INT8 activation từ acc_rescale |
| input | `1`     | `relu_v`        | data in | valid |
| input | `1`     | `compute_en_in` | control | gate gốc (ce_d5) |
| output | `1`    | `pool_write`    | data out | write strobe (AND cp_en ngoài module) |
| output signed | `[7:0]` | `pool_out` | data out | giá trị maxpool → Pong SRAM |

---

## 4. `cp_block` — Conv-Pool block (1 output channel)

Wrapper mỏng ghép `cp_mac` + `cp_accumulate_rescale` + `cp_pool` (tách cấu trúc, không
đổi logic, bit-exact). Pipeline: MULT(1) → TREE(3) → ACC(IN_CH) → S_bias(1, capture
không adder) → RESCALE(2) → RELU(1) → POOL.

**Parameters:** `IN_CH_W = 4` (width của a_d5 counter, cố định 4-bit).

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input | `1`      | `clk`           | clock/reset | clock |
| input | `1`      | `rst`           | clock/reset | reset |
| input | `[39:0]` | `x_in`          | data | 5 tap từ cp_engine mux_s1, packed 5×8b |
| input | `[39:0]` | `w`             | data | weights từ w_packed, packed 5×8b |
| input signed | `[31:0]` | `bias_in`  | config | INT32 bias từ cp_weight_store |
| input | `[3:0]`  | `a_in`          | control | channel counter delayed 5 cy (a_d5) |
| input | `[3:0]`  | `in_ch`         | control | IN_CH layer hiện tại (1/4/4/8) |
| input | `1`      | `compute_en_in` | control | pipeline enable delayed 5 cy (ce_d5) |
| input | `[3:0]`  | `nb`            | control | rescale shift (0..15; max dùng = 8) |
| input | `1`      | `relu_en`       | control | 1 = Conv4 only |
| input | `1`      | `pool_rst`      | control | pool reset (layer transition) |
| output | `1`     | `pool_write`    | data out | write strobe (AND cp_en ngoài module) |
| output signed | `[7:0]` | `pool_out` | data out | giá trị maxpool → Pong SRAM |

---

## 5. `cp_weight_store` — weight + bias storage (ROM single-load)

Lưu conv weight + bias cho 8-PE cp_engine. **Chỉ 1 biến thể trong `RTL/`**: 4 FF-array
ROM per-layer (`w_rom_conv1..4`), MUX kết hợp async (layer 4:1 + ic 8:1) rồi 1 FF stage
(`w_packed` valid ở N+1). Weight nạp 1 lần lúc elaboration qua `$readmemh`
(`conv1_w.hex`..`conv4_w.hex`, `conv_bias.hex`) — **không có bus write, không có
`cfg_base`/runtime reload** (khác bản `RTL_weight/`). Output flatten vì Verilog-2001
cấm array-port.

**Parameters:** none.

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input | `1`      | `clk`         | clock/reset | clock |
| input | `1`      | `rst`         | clock/reset | reset (không dùng trong logic — giữ cho đồng bộ port) |
| input | `[2:0]`  | `layer_state` | selector | CONV1=2 .. CONV4=5 |
| input | `[3:0]`  | `a`           | selector | input-channel counter (weight word index) |
| output | `[319:0]` | `w_packed_flat` | data out | 8 × 40b packed weights (registered N+1) |
| output | `[255:0]` | `b_cur_flat`    | data out | 8 × 32b INT32 bias, layer hiện tại |

---

## 6. `cp_engine` — 8 CP block song song

8 output channel chạy song song. Sở hữu SRW array, tap MUX, delay chain (a_d5/ce_d5),
sinh địa chỉ đọc SRAM (`t−2`), weight store, và gating `pong_we = pool_write & cp_en`.

**Parameters:** none.

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input | `1`      | `clk`             | clock/reset | clock |
| input | `1`      | `rst`             | clock/reset | reset |
| input | `[3:0]`  | `a`               | control | channel counter 0..IN_CH−1 |
| input | `[3:0]`  | `in_ch`           | control | IN_CH layer: 1/4/4/8 |
| input | `[11:0]` | `in_len`          | control | IN_LEN layer (2500/500/100/20) |
| input | `1`      | `shift_en`        | control | = (a == IN_CH−1) |
| input | `1`      | `srw_rst`         | control | SRW clear pulse (layer transition) |
| input | `1`      | `compute_en`      | control | pipeline enable (0 trong pre-fetch) |
| input | `[3:0]`  | `nb`              | control | rescale shift/layer (0..15; max 8) |
| input | `1`      | `relu_en`         | control | 1 = Conv4 only |
| input | `[7:0]`  | `cp_en`           | control | bitmask output channel active |
| input | `[2:0]`  | `layer_state`     | control | CONV1=2..CONV4=5 |
| input | `1`      | `pool_rst`        | control | reset pool_cnt (layer transition) |
| input | `[7:0]`  | `input_sram_dout` | data in | từ input_sram (Conv1 only) |
| input | `[63:0]` | `ping_dout`       | data in | từ ping_pong: `ping_dout[ch*8+:8]` |
| output | `[63:0]` | `pong_din`       | data out | write data/channel: `pong_din[ch*8+:8]` |
| output | `[7:0]`  | `pong_we`        | data out | per-channel write enable |
| output | `[11:0]` | `sram_rd_addr`   | addr | read addr → input_sram + ping_pong_sram |
| input | `[11:0]` | `sram_rd_addr_in` | addr | base từ controller (= t) |

> Không có port `w_wr_*`/`b_wr_*`/`cfg_base` trong bản `RTL/` — đó là phần chỉ có ở
> `RTL_weight/` (weight RAM bus write + runtime layer_base).

---

## 7. `cnn_controller` — FSM thống nhất

FSM điều khiển toàn pipeline: drive cp_engine (Conv1..4) rồi gap_fc_argmax
(GAP/FC/Argmax) tuần tự. States: IDLE/LOAD_INPUT/CONV1..4/GAP_FC_S/DONE_S. Topology
**hard-code** qua 3 function nội bộ (`cfg_in_ch_of`/`cfg_cp_en_of`/`cfg_nb_of`,
input = layer index 0..3, KHÔNG phải port bus) — ningba re-train 2026-07-28:
`in_ch=1,4,4,8` / `cp_en=0F,0F,FF,FF` / `nb=8,7,6,7`.

**Parameters:** none.

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input | `1`      | `clk`           | clock/reset | clock |
| input | `1`      | `rst`           | clock/reset | synchronous reset |
| input | `1`      | `start`         | control in | 1-cycle pulse từ avalon_slave |
| input | `1`      | `pool_write`    | control in | từ cp_engine (ch0 representative) |
| output | `[3:0]`  | `a`            | to cp_engine | channel counter 0..IN_CH−1 |
| output | `[11:0]` | `t`            | to cp_engine | output position counter |
| output | `1`      | `shift_en`     | to cp_engine | = (a == in_ch−1) |
| output | `1`      | `srw_rst`      | to cp_engine | SRW clear pulse (1 cy) |
| output | `1`      | `compute_en`   | to cp_engine | pipeline enable |
| output | `[3:0]`  | `in_ch`        | to cp_engine | IN_CH layer hiện tại |
| output | `[11:0]` | `in_len`       | to cp_engine | IN_LEN layer hiện tại |
| output | `[3:0]`  | `nb`           | to cp_engine | rescale shift (0..15; max 8) |
| output | `1`      | `relu_en`      | to cp_engine | ReLU enable (Conv4 only) |
| output | `[7:0]`  | `cp_en`        | to cp_engine | active output channel bitmask |
| output | `1`      | `bank_sel`     | to SRAM | Ping/Pong bank selector |
| output | `[11:0]` | `pong_addr`    | to SRAM | Pong SRAM write address |
| output | `1`      | `pool_rst`     | to cp_engine | reset pool_cnt (layer transition) |
| output | `[2:0]`  | `fc_sub_state` | to gap_fc | sub-state GAP/FC/flush/argmax/done |
| output | `[3:0]`  | `gap_step`     | to gap_fc | 0..5 |
| output | `[3:0]`  | `fc_step`      | to gap_fc | 0..9 |
| output | `[1:0]`  | `argmax_step`  | to gap_fc | 0..3 |
| input | `[1:0]`  | `argmax_result` | from gap_fc | argmax class |
| output | `[2:0]`  | `layer_state`  | top-level | expose cho cp_engine MUX + top |
| output | `1`      | `busy`         | top-level | 1 khi ≠ IDLE/DONE |
| output | `1`      | `done`         | top-level | 1-cycle pulse |
| output | `[1:0]`  | `result`       | top-level | latched argmax class |

> Không có port `cfg_in_ch`/`cfg_cp_en`/`cfg_nb` trong bản `RTL/` — topology cố định
> qua function nội bộ, không nạp được lúc runtime (khác `RTL_weight/`).

---

## 8. `gap_fc_argmax` — GAP → FC → Argmax

Wrapper mỏng (bit-exact structural split, không đổi logic) ghép 3 submodule:
`gap_unit` (GAP, ping_dout→gap_reg_flat) → `fc_unit` (FC + FC weight/bias store,
gap_reg_flat→fc_acc_flat) → `argmax_unit` (Argmax, fc_acc_flat→result). Tổng
GAP(6cy) → FC(10cy + 1 flush) → Argmax(4cy) → Done(1cy) = 22cy sau khi vào từ Conv4
layer_done. FC weight/bias là FF array trong `fc_unit`, nạp 1 lần qua `$readmemh`
(`fc_weights.hex`/`fc_bias.hex`) — **không có bus write** trong bản `RTL/`.

**Parameters:** none.

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input | `1`      | `clk`          | clock/reset | clock |
| input | `1`      | `rst`          | clock/reset | reset |
| input | `[2:0]`  | `fc_sub_state` | control | GAP_S/FC_S/FC_FLUSH/ARGMAX_S/DONE_S |
| input | `[3:0]`  | `gap_step`     | control | 0..5 |
| input | `[3:0]`  | `fc_step`      | control | 0..9 |
| input | `[1:0]`  | `argmax_step`  | control | 0..3 |
| input | `[63:0]` | `ping_dout`    | data in | Conv4 output; `ping_dout[ch*8+:8]`, 1-cy |
| output | `[8:0]` | `gap_rd_addr`  | addr out | Ping SRAM read addr, broadcast 8 ch (0..3) |
| output | `[1:0]` | `result`       | data out | argmax class index |

> Không có port `out_ch_mask`/`fcw_wr_en`/`fcw_wr_addr`/`fcw_wr_data` trong bản
> `RTL/` — GAP luôn coi 8 kênh Conv4 active (fixed Chapman/ningba topology); đó là
> phần chỉ có ở `RTL_weight/` (Conv4 out_ch<8 qua CONFIG window).

---

## 9. `input_sram` — 2500×8b simple dual-port (M10K)

Buffer input ECG. Write từ bus adapter (host ghi bất kỳ lúc nào); read từ cp_engine
(1-cycle sync). Nằm ở **wrapper**, ngoài `ecg_core`.

**Parameters:** none.

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input | `1`      | `clk`     | clock | clock |
| input | `[11:0]` | `wr_addr` | write port | 0..2499 |
| input | `[7:0]`  | `din`     | write port | write data |
| input | `1`      | `we`      | write port | write enable |
| input | `[11:0]` | `rd_addr` | read port | 0..2499 |
| output | `[7:0]` | `dout`    | read port | read data (1-cycle sync) |

---

## 10. `ping_pong_sram` — feature-map ping/pong

2 set × 8 channel × 500 entry × 8-bit (pad 512 cho M10K). `bank_sel` hoán Ping (đọc)
↔ Pong (ghi). 16 mem array riêng (8 ch × 2 bank), mỗi cái 1 M10K.

**Parameters:** none.

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input | `1`      | `clk`      | clock | clock |
| input | `1`      | `bank_sel` | control | 0: A=Ping B=Pong \| 1: B=Ping A=Pong |
| input | `[8:0]`  | `wr_addr`  | write (Pong) | 0..499 |
| input | `[63:0]` | `din`      | write (Pong) | 8 ch packed: `din[ch*8+:8]` |
| input | `[7:0]`  | `we`       | write (Pong) | per-channel write enable |
| input | `[8:0]`  | `rd_addr`  | read (Ping) | 0..499 |
| output | `[63:0]` | `dout`    | read (Ping) | 8 ch packed: `dout[ch*8+:8]` |

---

## 11. `ecg_core` — bus-agnostic compute core

Core thuần, không biết Avalon: ghép `ping_pong_sram` + `cp_engine` + `gap_fc_argmax` +
`cnn_controller`. Dùng lại được dưới nhiều wrapper (Avalon/Nios V/UART) mà không đụng
datapath đã verify. `input_sram` ở wrapper — core chỉ đọc qua `input_rd_addr`/`input_dout`.

**Parameters:** none.

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input | `1`      | `clk`         | clock/reset | clock |
| input | `1`      | `rst`         | clock/reset | synchronous reset (active high) |
| output | `[11:0]` | `input_rd_addr` | input SRAM | read addr → wrapper input_sram |
| input | `[7:0]`  | `input_dout`  | input SRAM | read data (1-cy latency) |
| input | `1`      | `start`       | control/status | kick off inference |
| output | `1`     | `busy`        | control/status | busy |
| output | `1`     | `done`        | control/status | done pulse |
| output | `[1:0]` | `result`      | control/status | class 0..3 |

> Bản `RTL/` chỉ có 7 port này (bus-agnostic core tối giản). Bản `RTL_weight/` có
> thêm `w_wr_*`/`b_wr_*`/`fcw_wr_*` (weight/bias/FC load) và `cfg_in_ch`/`cfg_cp_en`/
> `cfg_nb`/`cfg_base` (runtime topology) — `out_ch_mask` của gap_fc_argmax ở bản đó
> nối vào `cfg_cp_en[3*8 +: 8]` (slice Conv4) bên trong `ecg_core`.

---

## Build variants

Bản `RTL/` **không có** `ifdef`/`define` build variant nào (khác `RTL_weight/`, nơi
`WEIGHT_ROM`/`NO_WEIGHT_INIT` chọn giữa FF-ROM và M10K weight-RAM reload). Toàn bộ
11 module trong `RTL/` chỉ có 1 biến thể: ROM single-load, weight baked-in qua
`$readmemh`, topology hard-code trong `cnn_controller.v`.

## Load-bearing widths (tham chiếu nhanh)

| Signal / storage | Width | Ghi chú |
|---|---|---|
| ping_pong addr (`wr_addr`/`rd_addr`) | `[8:0]` | dùng 0..499; mem array `[0:511]` |
| input_sram + cp_engine addr (`sram_rd_addr`, `t`) | `[11:0]` | dùng 0..2499 |
| weight ROM entry (`w_rom_conv1..4`) | `[39:0]` × (4/16/32/64 entries) | packed 5 tap × 8b, per-layer FF array |
| bias store (`b_store`) | `[31:0]` × 32 (`[0:31]`) | INT32, addr = oc*4 + layer |
| tap/weight packed (`x_in`, `w`) | `[39:0]` | 5 × 8b |
| tree_out | `[19:0]` | sign-extended Σ 5 tích |
| feature-map bus (`ping_dout`, `pong_din`) | `[63:0]` | 8 ch × 8b |
