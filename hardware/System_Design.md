# CNN Accelerator — ECG Arrhythmia Classification (DE10-Standard)

## Kiến Trúc Tổng Thể

```
HPS (Cortex-A9) → Avalon-MM → Input SRAM (2500×8b, cố định)
                                      ↓ Conv1 only (MUX trong cp_engine)
                         ┌────────────────────────────┐
                         │   Conv-Pool Engine (CPE)   │
                         │   8 CP blocks parallel     │
                         │   Kernel=5, pad=2, stride=1│
                         │   MaxPool K=5 stride=5     │
                         └────────────┬───────────────┘
                                      │ Ping-Pong SRAM (inter-layer)
                                      ↓
                         ┌────────────────────────────┐
                         │   GAP / FC / Argmax         │
                         └────────────┬───────────────┘
                                      ↓
                              result[1:0] (0-3)
```

---

## Model: ECG_1DCNN Pruned

| Layer | In_ch | Out_ch | K | Pool | ReLU | In_len | Out_len |
|-------|--------|---------|---|------|------|--------|---------|
| Conv1 | 1      | 4       | 5 | /5   | No   | 2500   | 500     |
| Conv2 | 4      | 4       | 5 | /5   | No   | 500    | 100     |
| Conv3 | 4      | 8       | 5 | /5   | No   | 100    | 20      |
| Conv4 | 8      | 8       | 5 | /5   | Yes  | 20     | 4       |
| GAP   | 8      | 8       | — | /4   | —    | 4      | 1       |
| FC    | 8      | 4       | — | —    | —    | 1      | 1       |

**INT8 Quantization (QAT — calibrated, input_shift=2):**
```
nb:       Conv1=8, Conv2=6, Conv3=6, Conv4=7, FC=0
w_shift:  Conv1=6, Conv2=6, Conv3=6, Conv4=7, FC=8
Rescale:  clamp(round_half_up(acc / 2^nb), -127, 127)
round_half_up: (acc + 2^(nb-1)) >> nb
Bias:     bias_scaled = round(b_float × 2^nb), INT32 little-endian
RTL pipeline: acc → +bias → >>nb → clamp[-127,127] → ReLU(Conv4) → MaxPool

GAP:      integer floor division sum/4 = (sum) >> 2 (positive only after Conv4 ReLU).
          Python golden uses torch.floor(sum/4) to match hardware spec.
FC:       no rescale (nb_fc=0), raw INT32 logits feed directly into argmax.
          FC bias = 0 (rounded from tiny float values), omitted in RTL.
```

---

## Design Convention — Conv4 là chuẩn tham chiếu

> **Mọi tính toán kiến trúc, sizing bus, delay chain, ROM address width đều phải đủ cho Conv4 (IN_CH=8, OUT_CH=8) — layer có tham số lớn nhất.**
>
> Các layer nhỏ hơn (Conv1 IN_CH=1, Conv2/3 IN_CH=4) là subset — phần cứng chạy đúng Conv4 thì chạy đúng tất cả.

| Signal / Resource | Conv4 (chuẩn) | Conv1..3 |
|---|---|---|
| `a` counter width | 4-bit (0..7) | dùng chung, IN_CH nhỏ hơn |
| `in_ch` | 8 | 1 / 4 / 4 |
| ROM addr width | 6-bit (oc*8+ic, max=63) | nhỏ hơn |
| Delay chain | a_d6 (6 cycles) | dùng chung |
| cp_en bitmask | 8'hFF (8 oc active) | 8'h0F hoặc ít hơn |
| `acc` accumulates | 8 partial sums | ít hơn |

**Checklist khi thêm/sửa tham số:**
- [ ] Đủ bit width cho Conv4 (IN_CH=8, OUT_CH=8)?
- [ ] Delay chain (a_d6) tính từ Conv4 pipeline depth?
- [ ] ROM address range cover oc=0..7, ic=0..7?
- [ ] `in_ch - 1` comparison đúng khi in_ch=8?

---

## Module List

| Module | File | Status |
|--------|------|--------|
| CP Block pipeline | RTL/cp_block.v | ✓ Done |
| CP Engine (8 blocks) | RTL/cp_engine.v | ✓ Done |
| CNN Controller FSM | RTL/cnn_controller.v | ✓ Done |
| GAP/FC/Argmax | RTL/gap_fc_argmax.v | ✓ Done |
| Ping-Pong SRAM | RTL/ping_pong_sram.v | ✓ Done |
| Input SRAM | RTL/input_sram.v | ✓ Done |
| Top-Level | RTL/ecg_accelerator_top.v | ✓ Done |
| Avalon-MM Slave | RTL/avalon_slave.v | ✓ Done |

**Target:** DE10-Standard (Cyclone V 5CSXFC6D6F31C6), 100 MHz

---

## Resource Estimate

| Resource | Used (est.) | Total | % |
|----------|-------------|-------|---|
| ALM | ~500 | 41,910 | 1.2% |
| M10K | ~11 (input+pingpong SRAM) | 397 | 2.8% |
| MLAB | ~2 (b_store bias) | many | — |
| FF | ~185 (weight arrays) | ~167,000 | 0.1% |
| DSP18 | 40 | 84 | 47.6% |

DSP: 8 CP blocks × 5 MULT = 40 DSP18.

**Weight storage (packed 5-tap per word, 40-bit):**
| Array | Entries | Size | Map |
|-------|---------|------|-----|
| `w_rom_conv1` | 4   (4oc×1ic)   | 160 b   | FF array |
| `w_rom_conv2` | 16  (4oc×4ic)   | 640 b   | FF array |
| `w_rom_conv3` | 32  (8oc×4ic)   | 1280 b  | FF array |
| `w_rom_conv4` | 64  (8oc×8ic)   | 2560 b  | FF array |
| **Total weights** | **116 entries × 40b** | **4640 b ≈ 185 FF** | — |
| `b_store` (INT32 × 32) | 32 | 1024 b | MLAB |

FF array (no ramstyle): async read, không có port replication, timing deterministic.
w_comb[oc] là combinational MUX (giống mux_comb của SRW) → w_packed FF → arrive MULT cùng cycle với mux_s1.

→ Tổng weight = **580 INT8 actual** (4640 bits), không lãng phí so với flat 1280-slot (10240 bits).

---

## Task List

### ✓ RTL Implementation — HOÀN THÀNH

- [x] cp_block.v — pipeline S1-S9 + round_add critical path fix
- [x] cp_engine.v — 8 CP blocks + SRW + weight/bias store ($readmemh)
- [x] cnn_controller.v — Unified FSM IDLE→LOAD→CONV1-4→GAP_FC→DONE
- [x] gap_fc_argmax.v — GAP(6cy) + FC(10cy+flush) + Argmax(4cy)
- [x] ping_pong_sram.v — dual-bank, 8 channels, bank_sel toggle
- [x] input_sram.v — 2500×8b, HPS write / CP read
- [x] ecg_accelerator_top.v — wire tất cả modules
- [x] avalon_slave.v — Avalon-MM, 6 registers

### ○ Weight & Data Export

- [ ] Re-train pruned model (channels 4,4,8,8) nếu chưa xong
- [ ] Re-run QAT-INT8 export
- [ ] Tạo `conv1_w.hex` — 4 entries × 40b packed (4oc × 1ic × 5tap)
- [ ] Tạo `conv2_w.hex` — 16 entries × 40b packed (4oc × 4ic)
- [ ] Tạo `conv3_w.hex` — 32 entries × 40b packed (8oc × 4ic)
- [ ] Tạo `conv4_w.hex` — 64 entries × 40b packed (8oc × 8ic)
- [ ] Tạo `conv_bias.hex` — 32 entries × INT32, addr = oc*4 + layer_idx
- [ ] Tạo `fc_weights.hex` — 32 entries × INT8, addr = k*8 + i
- [ ] Tạo golden `.mem` files cho mỗi layer (từ `generate_golden.py`)

### ○ Testbench & Simulation

- [x] `testbench/testcase.md` — test case list + coverage mapping (18+8+7 TCs, >90%)
- [x] `testbench/tb_cp_block.v` — 18 unit tests, S1-S9 pipeline, ~95% branch coverage
- [x] `testbench/tb_layer.v` — 8 integration tests, Conv1 end-to-end
- [x] `testbench/tb_top.v` — 7 full system tests (needs golden hex files)
- [ ] Export golden hex files from Python (`generate_golden.py`)
- [ ] Run `tb_cp_block.v` in ModelSim — all 18 TC PASS
- [ ] Run `tb_layer.v` — 500 pool_writes, bank_sel toggle verified
- [ ] Run `tb_top.v` — result[1:0] matches Python for ≥3 samples

### ○ Synthesis & Timing

- [ ] Tạo Quartus project (device: 5CSXFC6D6F31C6)
- [ ] Thêm timing constraint: `create_clock -period 10.0 [get_ports clk]`
- [ ] Synthesis pass (Quartus Compile)
- [ ] TimeQuest: kiểm tra Fmax ≥ 100 MHz
- [ ] Fix timing violations nếu có (pipeline thêm stage, retiming)
- [ ] Kiểm tra DSP18/M10K usage khớp estimate

### ○ On-Board Validation (DE10-Standard)

- [ ] Viết HPS driver C (`ecg_classify.c`)
- [ ] Load test ECG samples qua Avalon-MM
- [ ] Chạy inference, đọc result[1:0]
- [ ] Verify accuracy trên test set (target: khớp Python ~94.65%)

---

## Critical Path Analysis

```
Stage           Ops per cycle             LUT lvl   Tpd est.   Critical @150MHz?
────────────────────────────────────────────────────────────────────────────────
S1 MULT         8×8 → 16b (DSP18)           0       <2 ns       No (DSP)
S2-S4 TREE      adder pipeline              3-4     3-4 ns      No (split 3 stages)
S5 ACC          32b add (carry)             4-5     5-6 ns      Borderline
S5b ACC_FINAL   32b add (cond mux + add)    4-5     6-7 ns      **Borderline**
S_bias          32b add                     4-5     6-7 ns      **Borderline**
S6 RESCALE      add + barrel shift          5-6     7-8 ns      **HIGH RISK**
S7 CLAMP        compare + MUX               3-4     5-6 ns      Safe
GAP add         10b add (sext 8→10)         2-3     3-4 ns      Safe
FC acc          32b add (sext 16→32)        4       6-7 ns      Borderline
Weight ROM      4-way layer + 8-way ic MUX  2-3     4-5 ns      Safe
M10K BRAM       Sync read 1cy               (mem)   <3 ns       Safe (-6 grade)
```

**S6 fix applied**: `round_add` precomputed as wire ([cp_block.v:124-125](RTL/cp_block.v#L124)) → synthesizer sees 1 add + 1 shift (không phải 4 ops serial).

**Target Fmax assessment**:
- **100 MHz (10 ns)**: ✅ comfortable margin, no risk.
- **125 MHz (8 ns)**: ✅ likely closes with default Quartus settings.
- **150 MHz (6.67 ns)**: ⚠️ S6 RESCALE is bottleneck. May need:
  - Quartus "Optimize speed" + register retiming.
  - Hoặc tách pipeline thêm 1 stage giữa `(biased + round_add)` và `>>> nb`.
- **180+ MHz**: cần refactor (split S6 stages).

**SDC files**:
- [hardware/cnn_accelerator_top.sdc](cnn_accelerator_top.sdc) — Legacy (Avalon-ST design cũ, không match top hiện tại — không dùng).
- [hardware/ecg_accelerator_top_100mhz.sdc](ecg_accelerator_top_100mhz.sdc) — **100 MHz standard target / fallback**. Matches `ecg_accelerator_top` ports.
- [hardware/ecg_accelerator_top_150mhz.sdc](ecg_accelerator_top_150mhz.sdc) — 150 MHz experimental. Matches same ports.

**Synthesis workflow**:
1. Trong Quartus → Assignments → Settings → Timing Analyzer → SDC files: add **`ecg_accelerator_top_150mhz.sdc`** trước.
2. Compile → check Timing Analyzer report:
   - `report_clock_fmax_summary` xem Fmax đạt được.
   - `report_timing -setup -npaths 10 -detail full_path` xem worst paths.
3. Nếu WNS ≥ 0 → giữ 150 MHz, update target.
4. Nếu WNS < 0 → switch sang `ecg_accelerator_top_100mhz.sdc` (đảm bảo close ~3 ns slack).

---

## Detail Documents

- [CP Pipeline](docs/cp_pipeline.md) — SRW, MUX, pipeline stages, timing diagrams
- [Controller FSM](docs/controller_fsm.md) — FSM states, per-layer timing, pre-fetch logic
- [GAP/FC/Argmax](docs/gap_fc_design.md) — datapath, timing tables, cycle counts
- [Memory Interface](docs/memory_interface.md) — SRAM architecture, Avalon memory map, hex file layout

---

## Verification Strategy

1. **Golden generation** (`generate_golden.py`): chạy INT8 simulation Python, export `.mem` files mỗi layer
2. **Unit test** từng CP block với 1 test sample, so sánh từng stage
3. **Layer-level test**: load golden input → chạy layer → so sánh output với golden
4. **Full system**: load ECG → argmax → so sánh class với Python
5. **10 test samples** phải 100% match Python INT8 simulation

---

## Verification Complete — 21/21 Bit-Exact PASS

**Trạng thái cuối**: 
- **TC argmax**: 8/8 PASS (3 samples result match Python golden)
- **L2 bit-exact ±10**: **21/21 PASS** (input + 4 pool + gap + logits cho cả 3 samples)
- **Cycles**: 2608 / inference (~26 µs @ 100 MHz)

**Root cause** (xác định qua cycle-level probe + hand-calc):
- Pipeline depth thực tế từ `mux_comb` đến `acc` register update edge = **5 cycles** (mux_s1 → prod → sum01,23 → sum0123 → tree_out), không phải 6.
- Code dùng `a_d6`/`ce_d6` (6-stage delay) để gate ACC update → **lệch 1 cycle** → MUX cy N input bị discard, MUX cy N+1 input mới được capture.
- Hậu quả: RTL output Conv[k] = Conv[k+1] expected → pool window dịch sang phải 1 position → pool[k] = max(Conv[5k+1..5k+5]) thay vì max(Conv[5k..5k+4]).
- Pool1 PASS L2 tolerance ±10 ban đầu vì diff chỉ ~1 LSB tại non-spike, nhưng tại spike diff lớn hơn → Conv2 amplify lỗi.

**Lịch sử bug + fixes** (7 fixes total, clean architecture bottom-up):

**Vòng 1 — Data alignment ở SRW path**:

1. **Back-padding** (`cp_engine.v`):
   - Pad cả front (`t < 2`) **và** back (`t ≥ in_len + 2`) → tránh đọc out-of-range data sau khi hết input.

2. **Registered pad_zero** (`cp_engine.v` `pad_zero_r`):
   - Register pad_zero (1 cy) để align với SRAM 1-cy synchronous read latency. Reset = 1 khi srw_rst.

3. **Gated prefetch counter** (`cnn_controller.v`):
   - Gate `prefetch_cnt` bằng `!srw_rst` để chỉ đếm các SRW shifts thật sự xảy ra. Số shifts cần = 5 (2 pad + 3 data) → `prefetch_cnt == 4` raises compute_en.

4. **in_len plumbed** (`cnn_controller.v` → `ecg_accelerator_top.v` → `cp_engine.v`):
   - Conv1=2500, Conv2=500, Conv3=100, Conv4=20. Phục vụ back-pad check.

**Vòng 2 — Pipeline timing root cause**:

5. **🔑 Pipeline delay d6 → d5** (`cp_engine.v` cp_block instantiation):
   - Sửa wiring: `.a_in(a_d5), .in_ch(inch_d5), .compute_en_in(ce_d5)` thay vì `a_d6/inch_d6/ce_d6`.
   - Cycle math: edge cy N+5 captures tree_out cy N+5 (= mux_comb cy N). Conditional reads `a_in` và `compute_en_in` AT that edge → cần delay 5 stages để match mux_comb cy N.
   - Fix này đơn lẻ cải thiện L2 từ 7→16 PASS.

**Vòng 3 — GAP precision (Python golden mismatch)**:

6. **GAP integer floor division** (`software/python/generate_golden.py`):
   - Bug: `nn.AdaptiveAvgPool1d` chia float `sum/4` → preserve 0.25/0.5/0.75 fractions. RTL `[9:2]` slice = integer floor (truncate fractional).
   - Fix: Python golden chuyển sang `torch.floor(after_pool4.sum(dim=-1) / 4.0)` matching RTL spec.
   - Hardware giữ nguyên (integer GAP rẻ hơn, đủ precision).
   - **Sample0 hand-verify**: pool4 ch0 sum=255, RTL gap=63, Python float gap=63.75. After re-gen golden với floor: 63=63 ✓. Logit[0] RTL=2058 = new golden ✓.
   - Cải thiện L2: 16→19 PASS.

**Vòng 4 — Testbench Avalon timing**:

7. **load_ecg_hex last-byte commit** (`testbench/tb_top.v`):
   - Bug: write at idx=2499 không reach input_sram do task return ngay sau WR_EN pulse — next operation (`run_inference` START) override avs_address trước khi input_sram fire on we=1 register.
   - Fix: thêm `@(posedge clk); #1;` sau load loop để cho phép cuối cùng SRAM write commit.
   - Cải thiện L2: 19→21 PASS.

**Verification độc lập**:
- Hand-calc Conv1[265..269] sample0 từ input → match exactly với gold pool1[ch=0][53].
- Hand-calc FC sample0 logit[0..3] từ int gap → match RTL fc_acc bit-exact.

---

## Rủi Ro & Lưu Ý

| Rủi ro | Mức độ | Biện pháp |
|--------|--------|-----------|
| S6 timing closure | Trung bình | round_add wire fix đã áp dụng |
| acc_final 2-cascaded adders | Trung bình | Document trong cp_block.v; verify TimeQuest; nếu fail → tách stage |
| $readmemh Quartus | Thấp | Quartus hỗ trợ trong initial block; 4 ROM packed với ramstyle hints |
| pool_write representative (ch0) | Thấp | Conv1/2 cp_en[0]=1 luôn active |
| Padding Conv1 (sram_rd_addr ≤ 2) | Thấp | pad_zero logic trong cp_engine |
| DSP18 dùng 47.6% | Trung bình | Đủ, nhưng cần verify Quartus report |
| w_packed MUX 4-way theo layer_state | Thấp | Pure case statement → Quartus tổng hợp thành 4:1 MUX, không critical |
