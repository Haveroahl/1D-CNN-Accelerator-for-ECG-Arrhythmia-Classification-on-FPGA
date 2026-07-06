# Module Interfaces — ECG CNN Accelerator (Compute Core)

Tài liệu tham chiếu **interface** của 11 module trong compute core (không bao gồm
bus adapter / top wrapper / UART). Đây là bảng port/parameter để đưa thẳng vào luận
văn — mọi width và tên tín hiệu lấy **verbatim** từ RTL trong `hardware/RTL/`
(bỏ qua `hardware/RTL/txt/` — snapshot cũ). Bản LaTeX paste-ready: [module_interfaces_tables.tex](module_interfaces_tables.tex).

**Cách đọc bảng:** mỗi hàng = 1 port. Cột `Dir` (input/output), `Width` (verbatim,
`1` = 1-bit scalar), `Name`, `Group` (nhóm theo cách file gom), `Purpose`.

---

## Phân cấp (ecg_core subtree)

```
ecg_core                              — bus-agnostic compute core
├── cp_engine                         — 8 CP block song song + SRW + tap MUX + addr-gen
│   ├── cp_weight_store               [WEIGHT_ROM: FF-ROM V1 | mặc định: 8× M10K RAM V2]
│   └── cp_block × 8  (oc = 0..7)     — 1 output channel mỗi block
│       ├── cp_mac                    — S1→S4  MAC + adder tree
│       ├── cp_accumulate_rescale     — S5→S8  acc + bias + rescale + ReLU
│       └── cp_pool                   — S9     MaxPool
├── ping_pong_sram                    — feature-map giữa các layer (Ping đọc / Pong ghi)
├── gap_fc_argmax                     — GAP → FC → Argmax
└── cnn_controller                    — FSM thống nhất Conv1..4 + GAP/FC
```

> **Lưu ý phân cấp:** `input_sram` **KHÔNG** nằm trong `ecg_core` — nó ở **wrapper**
> (top-level). Core chỉ *đọc* input SRAM qua cặp `input_rd_addr` → `input_dout`
> (1-cycle latency). Ranh giới này tách "compute latency" (start→done) khỏi pha
> host nạp input. `input_sram.v` vẫn được tài liệu hoá dưới đây vì nó thuộc datapath.

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
**fold vào acc-init** (`a_in==0`) → S6 chỉ còn 1 phép `>>> nb` thuần. S7 clamp
[−127,127], S8 ReLU (chỉ Conv4).

**Parameters:** none.

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input | `1`      | `clk`           | clock/reset | clock |
| input | `1`      | `rst`           | clock/reset | reset |
| input | `1`      | `pool_rst`      | clock/reset | pool/layer-transition reset |
| input signed | `[19:0]` | `tree_out` | data (MAC in) | từ cp_mac |
| input signed | `[31:0]` | `bias_in`  | config | INT32 bias (từ bias_rom) |
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
đổi logic, bit-exact). Pipeline: MULT(1) → TREE(3) → ACC(IN_CH) → ACC_FINAL(1) →
BIAS(1) → RESCALE(2) → RELU(1) → POOL.

**Parameters:** `IN_CH_W = 4` (width của a_d5 counter, cố định 4-bit).

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input | `1`      | `clk`           | clock/reset | clock |
| input | `1`      | `rst`           | clock/reset | reset |
| input | `[39:0]` | `x_in`          | data | 5 tap từ cp_engine mux_s1, packed 5×8b |
| input | `[39:0]` | `w`             | data | weights từ w_packed, packed 5×8b |
| input signed | `[31:0]` | `bias_in`  | config | INT32 bias từ bias_rom |
| input | `[3:0]`  | `a_in`          | control | channel counter delayed 5 cy (a_d5) |
| input | `[3:0]`  | `in_ch`         | control | IN_CH layer hiện tại (1/4/4/8) |
| input | `1`      | `compute_en_in` | control | pipeline enable delayed 5 cy (ce_d5) |
| input | `[3:0]`  | `nb`            | control | rescale shift (0..15; max dùng = 8) |
| input | `1`      | `relu_en`       | control | 1 = Conv4 only |
| input | `1`      | `pool_rst`      | control | pool reset (layer transition) |
| output | `1`     | `pool_write`    | data out | write strobe (AND cp_en ngoài module) |
| output signed | `[7:0]` | `pool_out` | data out | giá trị maxpool → Pong SRAM |

---

## 5. `cp_weight_store` — weight + bias storage (V1 ROM / V2 RAM)

Lưu conv weight + bias cho 8-PE cp_engine. Hai build-variant **chung port + chung
timing** (`w_packed` valid ở N+1) nên cùng bit-exact. Output flatten vì Verilog-2001
cấm array-port.

**Parameters:** none.

**Build variants:** xem [bảng tổng](#build-variants). `WEIGHT_ROM` → V1 (4 FF-array ROM
per-layer, async MUX, topology Chapman cố định, không bus write). Mặc định → V2 (8
M10K RAM per-oc, 40b×32, sync read, có bus write + `cfg_base`).

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input | `1`      | `clk`         | clock/reset | clock |
| input | `1`      | `rst`         | clock/reset | reset |
| input | `[2:0]`  | `layer_state` | selector | CONV1=2 .. CONV4=5 |
| input | `[3:0]`  | `a`           | selector | input-channel counter (weight word index) |
| input | `[19:0]` | `cfg_base`    | config (V2) | 4 × 5-bit weight-RAM word base |
| input | `1`      | `w_wr_en`     | bus write | conv weight write enable |
| input | `[2:0]`  | `w_wr_oc`     | bus write | chọn 1 trong 8 RAM per-oc |
| input | `[4:0]`  | `w_wr_word`   | bus write | RAM word index (0..31) = layer_base + ic |
| input | `[39:0]` | `w_wr_data`   | bus write | full 40-bit packed 5-tap entry |
| input | `1`      | `b_wr_en`     | bus write | bias write enable |
| input | `[4:0]`  | `b_wr_addr`   | bus write | bias addr (0..31) |
| input | `[31:0]` | `b_wr_data`   | bus write | INT32 bias |
| output | `[319:0]` | `w_packed_flat` | data out | 8 × 40b packed weights (registered N+1) |
| output | `[255:0]` | `b_cur_flat`    | data out | 8 × 32b INT32 bias, layer hiện tại |

---

## 6. `cp_engine` — 8 CP block song song

8 output channel chạy song song. Sở hữu SRW array, tap MUX, delay chain (a_d5/ce_d5),
sinh địa chỉ đọc SRAM (`t−2`), weight store, và gating `pong_we = pool_write & cp_en`.

**Parameters:** none. (Kế thừa `WEIGHT_ROM`/`NO_WEIGHT_INIT` qua child `cp_weight_store`.)

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
| output | `[11:0]` | `sram_rd_addr`   | addr | read addr → input_sram + ping_pong |
| input | `[11:0]` | `sram_rd_addr_in` | addr | base từ controller (= t) |
| input | `1`      | `w_wr_en`         | bus write | conv weight WE |
| input | `[2:0]`  | `w_wr_oc`         | bus write | chọn RAM per-oc (0..7) |
| input | `[4:0]`  | `w_wr_word`       | bus write | RAM word index (0..31) |
| input | `[39:0]` | `w_wr_data`       | bus write | 40-bit packed 5-tap entry |
| input | `1`      | `b_wr_en`         | bus write | bias WE |
| input | `[4:0]`  | `b_wr_addr`       | bus write | bias addr |
| input | `[31:0]` | `b_wr_data`       | bus write | INT32 bias |
| input | `[19:0]` | `cfg_base`        | config | weight-RAM word base/layer (4 × 5-bit) |

---

## 7. `cnn_controller` — FSM thống nhất

FSM điều khiển toàn pipeline: drive cp_engine (Conv1..4) rồi gap_fc_argmax
(GAP/FC/Argmax) tuần tự. States: IDLE/LOAD_INPUT/CONV1..4/GAP_FC_S/DONE_S. Config
topology per-layer nạp qua `cfg_*` (reset default = Chapman).

**Parameters:** none.

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input | `1`      | `clk`           | clock/reset | clock |
| input | `1`      | `rst`           | clock/reset | synchronous reset |
| input | `1`      | `start`         | control in | 1-cycle pulse từ avalon_slave |
| input | `1`      | `pool_write`    | control in | từ cp_engine (ch0 representative) |
| input | `[15:0]` | `cfg_in_ch`     | config | 4 × 4-bit |
| input | `[31:0]` | `cfg_cp_en`     | config | 4 × 8-bit |
| input | `[19:0]` | `cfg_nb`        | config | 4 × 5-bit |
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

---

## 8. `gap_fc_argmax` — GAP → FC → Argmax

Engine tuần tự: GAP (6cy) → FC (10cy + 1 flush) → Argmax (4cy) → Done (1cy) = 22cy sau
khi vào từ Conv4 layer_done. FC weight/bias là FF array, nạp qua bus trước inference.

**Parameters:** none.

**Build variants:** `NO_WEIGHT_INIT` → bỏ `$readmemh` init `fc_weights.hex`/`fc_bias.hex`
(bus-only load). Bus write path luôn có.

| Dir | Width | Name | Group | Purpose |
|-----|-------|------|-------|---------|
| input | `1`      | `clk`          | clock/reset | clock |
| input | `1`      | `rst`          | clock/reset | reset |
| input | `[2:0]`  | `fc_sub_state` | control | GAP_S/FC_S/FC_FLUSH/ARGMAX_S/DONE_S |
| input | `[3:0]`  | `gap_step`     | control | 0..5 |
| input | `[3:0]`  | `fc_step`      | control | 0..9 |
| input | `[1:0]`  | `argmax_step`  | control | 0..3 |
| input | `[63:0]` | `ping_dout`    | data in | Conv4 output; `ping_dout[ch*8+:8]`, 1-cy |
| input | `[7:0]`  | `out_ch_mask`  | config | Conv4 active-output mask (= Conv4 cp_en); default FF |
| output | `[8:0]` | `gap_rd_addr`  | addr out | Ping SRAM read addr, broadcast 8 ch (0..3) |
| input | `1`      | `fcw_wr_en`    | bus write | FC weight/bias WE |
| input | `[5:0]`  | `fcw_wr_addr`  | bus write | addr; `[5]=1` → fc_b[addr[1:0]], else fc_w[addr[4:0]] |
| input | `[31:0]` | `fcw_wr_data`  | bus write | INT32 (bias) hoặc INT8 ở [7:0] (weight) |
| output | `[1:0]` | `result`       | data out | argmax class index |

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
| input | `1`      | `w_wr_en`     | weight load | conv weight WE |
| input | `[2:0]`  | `w_wr_oc`     | weight load | per-oc RAM select |
| input | `[4:0]`  | `w_wr_word`   | weight load | RAM word index |
| input | `[39:0]` | `w_wr_data`   | weight load | 40-bit weight entry |
| input | `1`      | `b_wr_en`     | weight load | bias WE |
| input | `[4:0]`  | `b_wr_addr`   | weight load | bias addr |
| input | `[31:0]` | `b_wr_data`   | weight load | INT32 bias |
| input | `1`      | `fcw_wr_en`   | weight load | FC WE |
| input | `[5:0]`  | `fcw_wr_addr` | weight load | FC addr |
| input | `[31:0]` | `fcw_wr_data` | weight load | FC data |
| input | `[15:0]` | `cfg_in_ch`   | topology config | 4 × 4-bit |
| input | `[31:0]` | `cfg_cp_en`   | topology config | 4 × 8-bit |
| input | `[19:0]` | `cfg_nb`      | topology config | 4 × 5-bit |
| input | `[19:0]` | `cfg_base`    | topology config | 4 × 5-bit |
| input | `1`      | `start`       | control/status | kick off inference |
| output | `1`     | `busy`        | control/status | busy |
| output | `1`     | `done`        | control/status | done pulse |
| output | `[1:0]` | `result`      | control/status | class 0..3 |

> `cfg_base` chỉ cp_engine dùng; `out_ch_mask` của gap_fc_argmax được nối vào
> `cfg_cp_en[3*8 +: 8]` (slice Conv4) bên trong `ecg_core`.

---

## Build variants

| Define | File ảnh hưởng | Chuyển đổi gì |
|--------|----------------|---------------|
| `WEIGHT_ROM` | `cp_weight_store.v` | V1 FF-array ROM (async MUX, topology Chapman cố định, không bus write, `$readmemh` init luôn có) ↔ mặc định V2: 8× M10K weight RAM reload runtime. Ở V1, path bias bus-write (`b_wr_en`) bị compile-out. |
| `NO_WEIGHT_INIT` | `cp_weight_store.v` (path V2), `gap_fc_argmax.v` | Bỏ `$readmemh` init: w_ram0..7 + conv_bias (V2 only) và fc_weights + fc_bias → chứng minh đường nạp qua bus đứng độc lập. V1 ROM init KHÔNG bị guard bởi define này. |

Không còn `ifdef`/`define` build variant nào khác trong 11 module.

## Load-bearing widths (tham chiếu nhanh)

| Signal / storage | Width | Ghi chú |
|---|---|---|
| ping_pong addr (`wr_addr`/`rd_addr`) | `[8:0]` | dùng 0..499; mem array `[0:511]` |
| input_sram + cp_engine addr (`sram_rd_addr`, `t`) | `[11:0]` | dùng 0..2499 |
| weight RAM word (`w_ram*`) | `[39:0]` × 32 (`[0:31]`) | packed 5 tap × 8b |
| bias store (`b_store`) | `[31:0]` × 32 (`[0:31]`) | INT32, addr = oc*4 + layer |
| tap/weight packed (`x_in`, `w`) | `[39:0]` | 5 × 8b |
| tree_out | `[19:0]` | sign-extended Σ 5 tích |
| feature-map bus (`ping_dout`, `pong_din`) | `[63:0]` | 8 ch × 8b |
