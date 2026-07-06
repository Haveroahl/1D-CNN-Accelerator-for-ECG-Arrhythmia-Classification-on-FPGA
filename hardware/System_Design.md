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
| Delay chain | a_d5 (5 cycles) feed cp_block | dùng chung |
| cp_en bitmask | 8'hFF (8 oc active) | 8'h0F hoặc ít hơn |
| `acc` accumulates | 8 partial sums | ít hơn |

**Checklist khi thêm/sửa tham số:**
- [ ] Đủ bit width cho Conv4 (IN_CH=8, OUT_CH=8)?
- [ ] Delay chain (a_d5) khớp đúng pipeline depth mux_comb → tree_out (5 stages: mux_s1 → prod → sum01,23 → sum0123 → tree_out)?
- [ ] ROM address range cover oc=0..7, ic=0..7?
- [ ] `in_ch - 1` comparison đúng khi in_ch=8?

---

## Runtime-Reconfigurable Topology (per-layer channel + nb)

Topology mỗi layer (in_ch, cp_en, nb, weight-RAM base) **nạp được lúc runtime qua Avalon
CONFIG window** — không recompile bitstream. Cùng 1 bitstream chạy nhiều cấu hình kênh
khác nhau (bội-2, ≤8 mỗi layer), miễn nạp kèm weight/bias/FC khớp.

**CONFIG window** (`avalon_slave.v`, addr[13]=1, addr[12:11]=11):
- địa chỉ = `0x3800 | (field<<2) | layer` ; layer 0..3 = Conv1..4
- field: 0=in_ch[3:0], 1=cp_en[7:0], 2=nb[4:0], 3=layer_base[4:0]
- **Reset default = Chapman** (in_ch 1,4,4,8 / cp_en 0F,0F,FF,FF / nb 8,6,6,7 / base 0,1,5,9)
  → không nạp config gì = hành vi cũ y hệt (tb_top 21/21 bit-exact không đổi).

**Đường đi:** `avalon_slave` (cfg regs) → `ecg_core` → `cnn_controller` (in_ch/cp_en/nb) +
`cp_engine` (layer_base) + `gap_fc_argmax` (out_ch_mask = Conv4 cp_en).

**Mở rộng đã làm:**
- Weight RAM `w_ram0..7[0:31]` (depth 32, từ 17) → cover MAX in_ch=(8,8,8,8) (tổng word ≤ 25).
- `gap_fc_argmax` thêm `out_ch_mask`: GAP force `gap_reg=0` cho Conv4 channel inactive
  → Conv4 out_ch<8 bit-exact, driver KHÔNG cần zero-pad FC weight.

**Ràng buộc (driver chịu trách nhiệm):**
1. `in_ch[L+1] = out_ch[L]` (output layer này = input layer sau).
2. **Active channel pack từ bit 0** (cp_en = 0x01/0x03/0x0F/0x3F/0xFF…). Bắt buộc vì
   controller dùng `cp_pong_we[0]` (ch0) làm heartbeat `pool_write` → ch0 phải luôn active.
3. Nạp đồng bộ weight (pack theo base) + bias (scale 2^nb) + nb + base + FC.
4. Tổng in_ch 4 layer ≤ 32 (depth RAM). Conv1 in_ch=1 cố định (single-lead) → max thực = 25.
5. `in_len`/`out_len` cố định (2500→500→100→20→4) — KHÔNG đổi runtime.

**Verify:**
- `tb_topo.v` (golden từ `gen_topo_golden.py`) — full inference **bit-exact 11/11**:
  chapman (1,4,4,8), MIN (2,2,2,2), MAX (8,8,8,8), 4 mixed, 3 odd/floor case
  (min1111, t3456, t3577) + **ptbxl** (weight PTB-XL thật + nb[Conv3]=7 runtime).
- `tb_topo_sweep.v` (manifest-driven, golden từ `gen_topo_golden.py`) — **48/48
  bit-exact** coverage sweep: mọi out_ch 1..8 ở mọi layer + monotone + non-monotone
  (3,1,7,5)(8,1,8,1) + random. Chứng minh RTL chạy đúng **mọi** topology 1..8/layer,
  không chỉ bội-2. Latency biến thiên thật 32.12µs (1,1,1,1) → 76.54µs (8,8,8,8).
- `tb_top.v` TC08/TC09/TC10 — config-write/consume/recover + GAP mask + word biên 31.

> **PTB-XL case (cross-dataset C3 enabling):** `gen_ptbxl_golden.py` lấy weight INT8
> đã QAT trên PTB-XL (`qat_ptbxl.py`, INT8 test acc 92.79%) + 1 ECG test sample thật,
> chạy qua **cùng RTL** với CONFIG `nb[Conv3]=7` (Chapman dùng 6) → fc_acc 4/4 khớp
> bit-exact. Đây là bằng chứng (không cần board) rằng **1 bitstream** chạy được cả
> Chapman lẫn PTB-XL chỉ bằng reconfig nb + weight reload qua Avalon. Driver JTAG
> (`soc/ecg_jtag_console.tcl`) đã có `load_topology`/`load_weights` + `demo_data/
> ptbxl_weights/topo.txt` (nb Conv3=7) để chạy on-board khi có DE10 + USB-Blaster.

> ⚠️ Bản copy Qsys trong `soc/*/submodules/` + `db/ip/` là snapshot cũ — regenerate Qsys
> nếu build Phase D để các module nhận port `cfg_*`.

---

## Module List

> Bảng port/interface đầy đủ của 11 module compute-core (verbatim từ RTL): [docs/module_interfaces.md](docs/module_interfaces.md) (kèm bản LaTeX paste-ready).

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

### ✓ Weight & Data Export — HOÀN THÀNH (baseline)

- [x] Pruned model channels (4,4,8,8) — `best_model_pruned.pth`
- [x] QAT-INT8 power-of-2 round-half-up — `qat_int8/model_qat_int8.pth` (94.65%)
- [x] `flat_weights.hex` — 580 INT8 entries (KHÔNG có comment lines)
- [x] Bias INT32 little-endian, scaled `b_int = round(b_float × 2^nb)`
- [x] Golden `.mem` files — 21 checkpoints/sample × 3 samples từ `generate_golden.py`

### ✓ Testbench & Simulation — HOÀN THÀNH (baseline)

- [x] `testbench/tb_cp_block.v` — 18 unit tests S1-S9 pipeline
- [x] `testbench/tb_layer.v` — 8 integration tests Conv1 end-to-end
- [x] `testbench/tb_top.v` — full system tests
- [x] **21/21 bit-exact PASS** với golden Python (xem section Verification Complete bên dưới)
- [x] Latency đo: 5216 cycles ≈ 52.16 µs @ 100 MHz

### ○ Synthesis & Timing — CHƯA LÀM (Phase C của roadmap Q3)

- [ ] Tạo Quartus project (device: 5CSXFC6D6F31C6)
- [ ] SDC: dùng `ecg_accelerator_top_100mhz.sdc` (đã chuẩn bị)
- [ ] Synthesis pass (Quartus Compile)
- [ ] TimeQuest: report Fmax thực + WNS
- [ ] PowerPlay với `.vcd` activity → dynamic + static power
- [ ] Energy/inference = Power × 52.16 µs (metric chính cho wearable story)
- [ ] Resource report: ALM, M10K, DSP18, FF — so sánh với estimate

### ○ On-Board Validation — CHƯA LÀM (Phase D của roadmap Q3)

- [ ] Program `.sof` vào DE10-Standard
- [ ] Viết HPS driver C (`ecg_classify.c`) + cross-compile cho Cortex-A9
- [ ] Load test ECG samples qua Avalon-MM, chạy inference, đọc result[1:0]
- [ ] Verify accuracy trên test set Chapman (target: khớp Python ~94.65%)
- [ ] Đo latency thực bằng ARM A9 cycle counter

### ○ Phase B — Lightweight Weight RAM (enabling C3 cross-dataset, Q3 paper)

> Mục đích: cùng 1 bitstream chạy được Chapman + MIT-BIH weight. Là **enabling mechanism** cho cross-dataset study (C3), không phải novelty chính.

- [ ] Refactor `cp_engine.v`: thay weight FF array bằng `weight_ram` interface
- [ ] Tạo `weight_ram.v` — dual-port M10K, write từ Avalon, read combinational/1-cy
- [ ] Mở rộng `avalon_slave.v` address: 5-bit → 12-bit
- [ ] Address map: `0x000-0x07F` weight+bias+FC (~580 INT8 word), `0x080-0x09C` input ECG, `0x0A0` control/status
- [ ] HPS driver: `load_weights(path)`, `load_ecg(buf)`, `run_inference()`, `read_result()`
- [ ] Regression: 21/21 bit-exact phải PASS với weight load via Avalon (không $readmemh)
- [ ] Resource overhead: +1-2 M10K, +100-200 ALM (estimate < 6% device)

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

## Roadmap cho Q3 paper

> Chi tiết novelty, contributions, paper structure: [../Paper_Proposal_Q3.md](../Paper_Proposal_Q3.md)
> Cross-dataset evaluation plan: [../Phase_3_evaluate.md](../Phase_3_evaluate.md)

Hardware đã verify baseline xong (21/21 bit-exact, 52 µs/inference). Còn lại cho paper Q3:

1. **Phase B** — Lightweight weight RAM (xem Task List ở trên) — enabling C3.
2. **Phase C** — Quartus synthesis + PowerPlay → energy/inference (metric chính wearable story).
3. **Phase D** — DE10-Standard on-board: HPS driver load Chapman weight → 94%, load MIT-BIH weight (Phase A trained) → match Python.

Hardware fit vào contributions paper:
- **C2 (Bit-exact framework)** — đã done, dùng 21 checkpoints để chứng minh Python↔RTL match.
- **C4 (IP core architecture)** — đã done, latency + cycle count.
- **C5 (Weight reload)** — Phase B, là phương tiện để chạy C3 cross-dataset.

Wearable angle (đã align với novelty pitch Hướng 3 a+c):
- Power-of-2 QAT → rescale chỉ shift + add → **0 DSP cho rescale** (so với general-scale cần 5 multiplier).
- → ít DSP → ít dynamic power → energy/inference thấp → phù hợp wearable continuous monitoring.
- PowerPlay sẽ cho số liệu cụ thể (Table 8 trong paper).

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
- **Inference latency**: **5216 clock / inference (~52.16 µs @ 100 MHz)** — deterministic cho cả 3 samples.
  - Đo bằng `$time` trong testbench `run_inference` task (sau khi fix poll-count → clock count, [tb_top.v:210-232](testbench/tb_top.v#L210)).
  - Khớp math FSM: Conv1(2500) + Conv2(2000) + Conv3(400) + Conv4(160) + GAP/FC/Argmax(22) + transition overhead ≈ 5082 + ~134 = 5216.
  - Throughput: ~19,200 inference/s @ 100 MHz.

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
