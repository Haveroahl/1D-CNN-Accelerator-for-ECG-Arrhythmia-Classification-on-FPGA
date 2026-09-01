# Memory Interface — Chi Tiết

## Tổng Quan

```
Avalon-MM master (host)
    │ Avalon-MM
    ▼
avalon_slave ──────────────────────────────▶ Input SRAM (2500×8b, cố định)
                                                  │ (Conv1 only, qua MUX)
                                                  ▼
                                           cp_engine SRW[0]

ping_pong_sram (2 banks × 8 channels × 512 entries × 8b)
    ← Pong write: cp_engine pool_write
    → Ping read: cp_engine SRW[ch] (Conv2..4) / gap_fc_argmax (GAP phase)
```

> `avalon_slave.v` chỉ là bus adapter — bất kỳ Avalon-MM master nào cũng dùng được.
> Thiết kế ban đầu nhắm HPS (Cortex-A9) nhưng Quartus Lite không có IP HPS Cyclone
> V nên Phase D thực tế dùng **JTAG-to-Avalon + System Console** (`jtag_top.v`,
> đã chạy board thật 94.27%) — xem [System_Design.md](../System_Design.md) mục
> Phase D. Địa chỉ/word map dưới đây không đổi dù host là HPS hay JTAG.

---

## Input SRAM

**Module:** `input_sram.v` (2500 × 8-bit, simple dual-port)

```
Write port: avalon_slave → sram_wr_addr[11:0], sram_din[7:0], sram_we
Read port:  cp_engine   ← sram_rd_addr[11:0], sram_dout[7:0]  (1-cycle latency)
```

- HPS ghi bất cứ lúc nào — độc lập với ping-pong bank_sel
- cp_engine địa chỉ: `sram_rd_addr = t - 2` (tính từ cp_engine, clamp ≥ 0)
- Chỉ dùng trong Conv1 (IN_CH=1)

**Lý do Input SRAM riêng (không ghi thẳng vào Ping-Pong):**
- Ping-Pong bank_sel thay đổi sau mỗi layer → HPS không biết bank nào là Ping
- Input SRAM cố định → HPS ghi địa chỉ cố định, không cần biết trạng thái inference
- Chi phí: ~3 M10K (DE10-Standard có 397 M10K)

---

## Ping-Pong SRAM

**Module:** `ping_pong_sram.v`

```
2 banks (Ping/Pong), mỗi bank:
  8 channels × 512 entries × 8-bit = 32 KB/bank
  (dùng max 500 entries cho Conv1 output)

bank_sel: 0 → Ping=BankA, Pong=BankB
          1 → Ping=BankB, Pong=BankA
```

**Write:** cp_engine pool_write (gating: `pong_we[oc] = pool_write && cp_en[oc]`)

**Read:**
- Conv1..4: cp_engine (`cp_sram_rd_addr[8:0]`)
- GAP phase: gap_fc_argmax (`gap_rd_addr[8:0]`, chỉ cần [1:0] thực tế)

**MUX tại top-level:**
```verilog
wire [8:0] pp_rd_addr = (ctrl_layer_state == 3'd6) ? gap_rd_addr
                                                    : cp_sram_rd_addr[8:0];
```

**Data layout per bank:**
```
Layer   Channels  Entries/ch  Total
Conv1      4        500       2000 entries (banks 0..3 only)
Conv2      4        100        400 entries
Conv3      8         20        160 entries
Conv4      8          4         32 entries
```

---

## Top-Level MUX: Conv1 vs Conv2..4

```verilog
// cp_engine srw_din (internal MUX trong cp_engine.v)
assign srw_din[0] = pad_zero ? 8'h00
                  : (layer_state == CONV1) ? input_sram_dout : ping_dout[0];
assign srw_din[1] = pad_zero ? 8'h00 : ping_dout[1];
// ... srw_din[2..7] tương tự

// Read address dùng chung (cp_engine tính t-2)
assign sram_rd_addr = (sram_rd_addr_in >= 12'd2) ? (sram_rd_addr_in - 12'd2) : 12'd0;
```

---

## Avalon-MM Interface

**Module:** `avalon_slave.v` — bản ROM single-load (`hardware/RTL/`). Địa chỉ
`avs_address[13:0]` là **word address** (không phải byte offset). Chỉ có Input SRAM
write port + control/status — **không có** weight/bias/FC bus-write, không có
CONFIG window (khác `RTL_weight/`, xem [System_Design.md](../System_Design.md)
mục "Runtime-Reconfigurable Topology").

**Memory Map** (word address, `avs_address[13:0]`):

```
Word Addr  R/W  Tên       Mô tả
──────────────────────────────────────────────────────────────────────
0x0000     W    DATA_IN   1 sample INT8 (bits [7:0]) — sram_din
0x0001     W    ADDR_IN   Địa chỉ Input SRAM (0..2499), 12-bit — sram_wr_addr
0x0002     W    WR_EN     Ghi bit[0]=1 → write DATA_IN → SRAM[ADDR_IN] — sram_we
0x0003     W    START     Ghi bit[0]=1 → pulse start 1 cycle; clear done_latched
0x0004     R    STATUS    [0]=busy, [1]=done_latched
0x0005     R    RESULT    [1:0]=class (0=AFIB,1=GSVT,2=SB,3=SR)
0x1000..   W    DATA WINDOW (burst) — addr[12]=1: index=addr[11:0] (0..2499),
0x19C3          write word → sram_din<=wd[7:0], sram_wr_addr<=index, sram_we<=1
                (1 word = 1 SRAM byte; dùng bởi JTAG-to-Avalon host, xem
                hardware/fpga/soc/ecg_jtag_console.tcl)
```

Low-register path (0x0000-0x0005) và DATA WINDOW (0x1000+) đều ghi vào cùng
`sram_din`/`sram_wr_addr`/`sram_we` — khác nhau ở việc DATA WINDOW gói cả 3 thành
1 write duy nhất (`avs_address[11:0]` = SRAM address luôn), còn low-register path
cần 3 lần ghi riêng (DATA_IN, ADDR_IN, rồi WR_EN) như `tb_top.v` dùng.

**HPS/host Software Sequence (C, qua low-register path):**

```c
volatile uint32_t *avs = mmap(NULL, 0x20, PROT_RW, MAP_SHARED, fd, 0xFF200000);
// word address → byte offset = word_addr << 2

// 1. Load 2500 ECG samples vào Input SRAM
for (int i = 0; i < 2500; i++) {
    avs[0x0000] = samples[i];   // DATA_IN  (word addr 0x0000)
    avs[0x0001] = i;            // ADDR_IN  (word addr 0x0001)
    avs[0x0002] = 1;            // WR_EN pulse (word addr 0x0002)
}

// 2. Start inference
avs[0x0003] = 1;               // START pulse (word addr 0x0003)

// 3. Poll busy
while (avs[0x0004] & 0x1);     // STATUS bit[0]=busy

// 4. Read result
uint32_t cls = avs[0x0005] & 0x3;   // RESULT
```

**Note:** `done_latched` được clear khi ghi START. `busy` = 1 khi layer_state ≠ IDLE/DONE.
Tất cả 1-cycle strobe (`sram_we`, `start`) tự động về 0 cycle sau nếu không ghi lại.

---

## Weight/Bias Files

```
conv1_w.hex   4  entries × 40b packed  (4oc × 1ic × 5tap)   w_rom_conv1  MLAB
conv2_w.hex   16 entries × 40b packed  (4oc × 4ic × 5tap)   w_rom_conv2  MLAB
conv3_w.hex   32 entries × 40b packed  (8oc × 4ic × 5tap)   w_rom_conv3  MLAB
conv4_w.hex   64 entries × 40b packed  (8oc × 8ic × 5tap)   w_rom_conv4  M10K

Entry format (40-bit, 10 hex chars):
  {tap4[7:0], tap3[7:0], tap2[7:0], tap1[7:0], tap0[7:0]}  MSB→LSB

Address:
  Conv1: oc          (oc=0..3)
  Conv2: oc*4 + ic   (oc=0..3, ic=0..3)
  Conv3: oc*4 + ic   (oc=0..7, ic=0..3)
  Conv4: oc*8 + ic   (oc=0..7, ic=0..7)  ← Conv4 là chuẩn (địa chỉ lớn nhất)

conv_bias.hex   32 entries INT32 little-endian  (8oc × 4layer)
                addr = oc*4 + layer_idx  (layer_idx: Conv1=0..Conv4=3)
                b_store[0:31] MLAB — $readmemh trong cp_weight_store.v

fc_weights.hex  32 entries INT8  (4k × 8i)
                addr = k*8 + i
                fc_w[0:31] — $readmemh trong fc_unit.v (submodule của gap_fc_argmax)

fc_bias.hex     4 entries INT32 little-endian, pre-scaled 2^w_shift[fc]
                fc_b[0:3] — $readmemh trong fc_unit.v; seed vào fc_acc tại fc_step==0
```

**Hex format:** KHÔNG có comment lines ($readmemh Quartus yêu cầu).

---

## M10K / MLAB Usage Estimate (Cyclone V)

```
Module             Size              Storage    Count
──────────────────────────────────────────────────────
input_sram         2500×8b = 20kb   M10K       3
ping_pong_sram     2×8×512×8b=64kb  M10K       8
w_rom_conv1        4×40b   = 160b   MLAB       1
w_rom_conv2        16×40b  = 640b   MLAB       1
w_rom_conv3        32×40b  =1280b   MLAB       1
w_rom_conv4        64×40b  =2560b   M10K       1
b_store            32×32b  =1024b   MLAB       1
fc_w               32×8b   = 256b   FF/MLAB    —
──────────────────────────────────────────────────────
Total M10K         ~12 / 397 = 3.0%
Total MLAB         ~4
```
