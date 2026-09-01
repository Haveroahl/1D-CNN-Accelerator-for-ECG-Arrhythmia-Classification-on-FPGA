# SIMD.md — Specification: SIMD Position-Parallel CNN Accelerator (variant design)

> **Mục đích file**: đặc tả RTL cho một biến thể **SIMD position-parallel** của ECG CNN
> accelerator, làm tham chiếu để code RTL mới. Tách hẳn khỏi production
> (`hardware/` = channel-parallel 8-PE, đã verify 21/21 bit-exact).

---

## 0. Mục tiêu thiết kế & đánh đổi

> **Spec này dùng để build một RTL variant TỐI ƯU LATENCY hơn production (channel-parallel).**
> SIMD-20 đạt **~1284 cycle (~12.8 µs @100MHz)** vs production 5216 cy (52 µs) → **~4.1× nhanh hơn**,
> đổi lại **controller phức tạp hơn** (3 vòng lồng oc×block×a + pha mồi line-buffer + lane-valid)
> và tài nguyên cao hơn. Đây là một điểm khác trên không gian PPA: production ưu tiên **Area+Power**,
> variant này ưu tiên **Performance (latency)**.

### Đánh đổi cần biết (cân nhắc khi build)
| Trục | Production (channel-//) | SIMD-20 variant (spec này) |
|---|---|---|
| **Latency** | 5216 cy / 52 µs | **~1284 cy / ~12.8 µs (~4.1× nhanh)** ✅ |
| Controller | 2 vòng lồng (t×a), cp_en bitmask, pad 1-pos | **3 vòng lồng (oc×block×a) + mồi + lane_valid** (phức tạp hơn) |
| DSP | 28 (25%) | ~70 (63%) |
| Power | thấp (DSP = 68% dynamic) | cao hơn (~2.5× DSP) |
| cycle×DSP (area-latency) | 146,048 | **89,880** (tốt hơn) ✅ |
| Bit-exact | đã verify 21/21 | phải verify lại (golden mở rộng — §12) |

### Khi nào chọn variant này
- Cần **throughput/latency cao** (nhiều inference/s, hoặc latency-critical).
- Chấp nhận DSP/power cao hơn + controller phức tạp + công verify lại.
- Lưu ý: với wearable continuous monitoring (chu kỳ đo ~10 s), latency 52 µs đã dư — speedup ~4×
  là **headroom mở rộng**, không phải ràng buộc bắt buộc. Nhưng là **đóng góp PPA hợp lệ** + material
  Design-Space Exploration cho paper (cho thấy đã khảo sát cả hai góc area↔latency).

---

## 1. Khái niệm & định vị

- **SIMD position-parallel**: L lane song song, mỗi lane = **1 output position liên tiếp**
  của **cùng 1 output channel**. Broadcast **weight** (chung), khác **data** (8+ cửa sổ lệch nhau).
- Đối lập với production (**channel-parallel**): 8 PE = 8 output **channel**, broadcast **input**.
- KHÔNG phải Liu 2023 (= fully-mapped/spatial, mỗi layer 1 khối PE riêng). Cả 3 đều là CNN accelerator,
  khác **dataflow**.
- Trục song song: thesis lấp lane theo **output channel**; SIMD lấp theo **position (độ dài chuỗi 1D)**.

### Cân bằng trục (vì sao nút thắt đối nhau)
| Layer | out_len (trước pool) | IN_CH | OUT_CH | Thesis (channel-//) | SIMD (position-//) |
|---|---|---|---|---|---|
| Conv1 | 2500 | 1 | 4 | lane phí (4/8) | **khít** (dài) |
| Conv2 | 500 | 4 | 4 | lane phí (4/8) | **khít** |
| Conv3 | 100 | 4 | 8 | khít (8/8) | khít |
| Conv4 | 20 | 8 | 8 | **khít** (8/8) | phí (ít position) |

---

## 2. Cấu hình chốt: **L = 20**

Chọn **L = 20 lane × 5-tap** vì:
- **Utilization**: 20 chia hết out_len mọi conv (2500/500/100/20) → 100% mọi layer.
- **Pool**: 20 = 4 × stride(5) → 4 cây-max **combinational**, không carry-over (L=8,16 không bội 5 →
  cửa sổ pool bắc cầu giữa 2 phát → cần carry-over, phức tạp). **L phải là bội số của stride pool (5).**
- L=20 cho **4 pooled value/phát** (20 position ÷ 5).

### Tài nguyên L=20
| | Giá trị |
|---|---|
| Multiplier | 20 × 5 = 100 → **~70 DSP (63% của 112)** |
| Multi line-buffer | 8 buffer × (L+4=24) × 8b ≈ **1536 FF** |
| Pong-sram | wide M10K 32-bit (4 pos/word), 8 ch × 2 bank ≈ 16 M10K |
| Requantize | 20 bộ shifter (power-of-2, **0 DSP**) |
| Pool | 4 cây-max comb |

---

## 2b. Critical path & timing

> Fmax SIMD **chưa synth thật** — mục này là phân tích định tính + giải pháp dự phòng.
> Phải synth (quartus_map+fit+sta) mới chốt số. Mỏ neo: production lõi ~137.6 MHz (Fix B).

### So với production: bỏ 1 path nặng, đẻ 2 path mới
| Path | Production | SIMD-20 | Ghi chú |
|---|---|---|---|
| `a → MUX-channel` (`srw_flat[a*5+slot]`) | **limiter ~10.7ns** | **biến mất** | SIMD không broadcast theo channel → bỏ MUX động ✅ |
| Line-buffer tap → 20 cửa sổ | MUX động (nặng) | **wiring tĩnh** (slot[l..l+4]) | nhẹ hơn — không select động ✅ |
| MAC + adder tree (trong lane) | OK (pipelined, Fix B) | giống | không phải limiter |
| Requantize ×20 | OK | song song độc lập (delay = 1 bản) | không phải limiter; power-of-2 → 0 DSP |
| **Weight broadcast → 20 lane** | fanout 5 (1 PE) | **fanout 100** (20×5) | ⚠️ **nghi phạm mới #1** |
| **Pool 4 cây-max** | rolling (nhẹ) | **comb tree depth-3 sau requant** | ⚠️ **nghi phạm mới #2** |

### Giải pháp #1 — Pool: 2-stage pipeline [CHỐT]
- max(5) = **4 comparator, critical depth = 3 levels**:
  ```
  tầng 1: m1=max(l0,l1), m2=max(l2,l3)   (2 comp song song)
  tầng 2: m3=max(m1,m2)                   (1 comp)
  tầng 3: max(m3,l4)                      (1 comp)
  ```
- Cắt depth-3 thành **2 tầng pipeline** (register giữa) → critical pool ~1.5 comparator/tầng.
- Áp cho cả 4 cây song song. Thêm 1 register stage giữa requantize và pong-write.

### Giải pháp #2 — Weight broadcast fan-out (1 set → 20 lane × 5 = 100 đích)
**Mặc định: W1 (replicate 4 bản theo nhóm lane)** — thử trước (rẻ nhất, +0 latency):
```
w_comb ─► w_reg_grp0 ─► lane0..4    (fan-out 25/bản)
       ├─► w_reg_grp1 ─► lane5..9    (4 nhóm trùng 4 cây pool → nhất quán)
       ├─► w_reg_grp2 ─► lane10..14
       └─► w_reg_grp3 ─► lane15..19
```
- Fan-out 100 → **25/bản** (4 bản), +160 FF, **+0 latency** (thay `w_packed` register, không thêm stage).

**Fallback: W3 (replicate 20 bản, 1/lane)** — CHỈ nếu synth báo weight path vẫn fail:
- Fan-out → **5/bản** (= production), +800 FF (rẻ, ~0.5% FF budget), +0 latency. Chắc ăn timing.

**Tránh W2 (2-stage weight pipeline, +1 latency)** — phải align delay-chain → đúng lớp bug
"pool window dịch 1 vị trí" lịch sử. Chỉ dùng nếu W1+W3 đều không đủ.

> Quy trình: thử **W1** → synth → nếu weight path fail thì **W3**. Đo rồi mới brute-force (giống
> Fix B→đo→Fix A của production). KHÔNG replicate 20 bản ngay khi chưa có số.

### Net Fmax (chưa chắc)
- Nếu xử lý tốt weight fanout (W1/W3) + pool 2-stage → SIMD **có thể ≥ production** (~137 MHz),
  vì bỏ được path `a→MUX-channel`.
- Nếu không xử lý fanout → **có thể tụt** vì fanout 100.
- ⚠️ Điểm hở khác chưa đo: routing congestion (100 DSP + 640 acc FF + 1536 line-buffer FF cụm lại).
  **Phải synth thật.**

---

## 3. Kiến trúc tổng (datapath end-to-end)

```
wrapper input_buffer ──stream──┐  (Conv1 only; §3b)
ping_pong (wide M10K)  ─────────┤  (MUX read-addr: Conv↔GAP, theo layer_state)
                         ▼
   ┌─────────── 8 multi line-buffer (1/in-channel, sâu L+4=24) ───────────┐
   │  mồi: nạp 8 ch song song từ ping-pong channel-major (1 lần/layer)     │
   │  tap: mỗi buffer → L=20 cửa sổ 5-tap chồng lấp                        │
   └──────────────────────────────┬───────────────────────────────────────┘
                                   ▼  (chọn buffer[ic] mỗi vòng in-channel)
   ┌─────────── 20 lane × 5-tap MAC (position-parallel) ──────────────────┐
   │  weight w[oc][ic] broadcast cho 20 lane; acc[lane] += qua IN_CH vòng  │
   └──────────────────────────────┬───────────────────────────────────────┘
                                   ▼  +bias(fold) → requantize(>>nb,clamp,ReLU) — 20 bộ
   ┌─────────── Pool: 4 cây-max combinational (gom 5 lane kề) ────────────┐
   │  pooled[0..3] = max(lane0-4), max(5-9), max(10-14), max(15-19)        │
   └──────────────────────────────┬───────────────────────────────────────┘
                                   ▼  4 pooled/phát → pack word32
   ┌─────────── Ghi pong-sram wide (Phương án B): 4 pos/word, 1 cycle ────┐
   └──────────────────────────────────────────────────────────────────────┘
                                   ▼  (sau Conv4)
   GAP (đọc word32 = 4 pos/ch, sum/4) → FC (8-in MAC, GIỮ NGUYÊN) → Argmax → result[1:0]
```

---

## 3b. Input buffer cho Conv1 — **Mô hình 2: buffer ở wrapper + core streaming** [CHỐT]

> Conv1 đọc input (in_len=2500, IN_CH=1) — khác Conv2-4 (đọc ping-pong). Quyết định: tách buffer
> input 2500 ra **wrapper**, core là **streaming consumer** thuần.

### Kiến trúc
```
WRAPPER (avalon_slave / input stage):          CORE (streaming):
  input_buffer 2500×8b (= isram dời ra ngoài)    KHÔNG có buffer 2500
  nhận data từ nguồn (Avalon write / Nios)        chỉ line-buffer ~24 (1 buffer cho Conv1)
  đủ 2500 → frame_ready → core start    ──stream──► đọc full-speed 1 mẫu/cycle → Conv1
```

### Vì sao Mô hình 2 (không phải Mô hình 1 "core streaming bounded không buffer")
| | Mô hình 1 (core stream, no buffer) | **Mô hình 2 (buffer wrapper)** [chọn] |
|---|---|---|
| Buffer 2500 | không có (tiết kiệm M10K) | wrapper (= isram dời ra) |
| Conv1 tốc độ | **theo nguồn** (ADC chậm → stall) | **full-speed** (đọc từ buffer) ✅ |
| Decouple nguồn/core | ❌ không | ✅ wrapper buffer đệm |
| Giữ speedup ~4×? | ❌ mất nếu nguồn chậm | ✅ giữ |

→ Mô hình 1 **mâu thuẫn mục tiêu latency** (Conv1 stall theo ADC chậm). Mô hình 2 decouple nguồn
chậm vs core full-speed → Conv1 đạt tốc độ thiết kế. Thực chất **≈ isram production**, chỉ đổi
đóng gói (core thuần streaming) + interface (stream + frame_start thay vì đọc bằng địa chỉ).

### Interface core ↔ wrapper buffer
- `stream_data[7:0]`, `stream_valid` — 1 mẫu/cycle vào core.
- `frame_start` — wrapper báo bắt đầu cửa sổ 2500 mới → core reset `stream_cnt`, reset line-buffer (pad trái).
- Overlap: wrapper buffer + free-flag (như `isram_free`) → nạp window N+1 song song compute N.

### Padding khi streaming (Mô hình 2) — counter thay địa chỉ
Streaming không có địa chỉ → dùng counter `stream_cnt` (0..in_len-1, reset mỗi `frame_start`):
- **Pad trái (PP4)**: `frame_start` → reset line-buffer = 0 → 2 zero đầu sẵn. **Y hệt Cách A.**
- **Pad phải (PP1)**: `stream_cnt ≥ in_len (2500)` → ép 0 (tự sinh, không chờ stream — pad là zero
  cố định, không phải data cần chờ). Counter biết khi hết mẫu thật.
- **Mấu chốt**: pad phải KHÔNG cần "nhìn trước" — counter báo hết mẫu → ép 0 ngay.
- ⚠️ **Rủi ro**: `frame_start` sai ranh giới → cửa sổ lệch → pad sai vị trí → output dịch (lớp bug
  alignment). Spec giả định **cửa sổ tách rời** (2500 mẫu/inference, KHÔNG sliding chồng lấp).

> Conv2-4 KHÔNG dùng input buffer này — chúng đọc ping-pong (§4 mồi). Input buffer chỉ phục vụ Conv1.

---

## 4. Multi line-buffer (1/in-channel)

- **Số buffer = max(IN_CH) = 8** (dùng chung 4 layer; Conv1-3 dùng 1/4/4 buffer, dư idle).
- **Độ sâu = L+4 = 24** (chứa L=20 cửa sổ 5-tap chồng lấp: position [t..t+23]).
- **Bề rộng = 8-bit** (INT8). Shift-register FF (tap song song mọi vị trí).

### Quá trình 3 giai đoạn (per layer)
1. **Mồi (fill)**: nạp L+4=24 position của 8 channel **song song** từ ping-pong channel-major
   (8 ch/cycle). 1 lần/layer (~24 cycle, hoặc ~6 nếu đọc wide). Lane chưa tính.
2. **Compute**: mỗi oc, tích lũy qua IN_CH input channel bằng cách **chọn buffer[ic]** (đã đầy)
   → **0 reload** giữa các ic. 20 lane mỗi cái 1 cửa sổ → 20 output position.
3. **Trượt**: nạp tiếp position cho khối kế (stream 1/cycle, không mồi lại).

**Lý do multi-buffer (vs single)**: single-buffer reload mỗi khi đổi ic → `OUT×IN×(L+4)` cycle
reload (Conv4 thảm họa). Multi-buffer mồi **1 lần phủ 8 channel** (nhờ ping-pong channel-major
cấp 8 ch/cycle) → reload ~`IN_CH×(L+4)`/layer. **Đẩy speedup từ ~1.3× lên ~4×.**

---

## 5. Pool (4 cây-max combinational)

- L=20 → 20 lane ra **đồng bộ cùng cycle** → pool là **combinational**, KHÔNG rolling/state:
  ```
  pooled[0] = max(lane0..4)
  pooled[1] = max(lane5..9)
  pooled[2] = max(lane10..14)
  pooled[3] = max(lane15..19)
  ```
- Pool gom giá trị **đã requantize** (INT8) → cây-max 8-bit, 0 DSP.
- ⚠️ Pool **combinational** mới đúng cho SIMD song song. "Pool rolling nhận liên tục 5" là mô hình
  thesis (1 pos/cycle), KHÔNG áp dụng cho SIMD (mâu thuẫn với 20 lane song song).
- **Timing**: max(5) = 4 comparator, depth-3 → **chia 2 tầng pipeline** để cắt critical path (xem §2b).

---

## 6. Ghi pong-sram — **Phương án B (wide M10K 4-pos/word)** [BẮT BUỘC]

### Vấn đề (vì sao cần B)
- Pool ra **4 pooled value/phát**, mỗi phát cách nhau **n = IN_CH cycle**.
- Pong-sram channel-major 8-bit ghi **1 value/cycle** → điều kiện không tràn: **n ≥ 4**.
- Conv2/3/4 (n=4,4,8) OK; **Conv1 (n=1) TRÀN** (vào 4/cycle, ra 1/cycle) → write-FIFO KHÔNG cứu được
  (vào > ra kéo dài, không phải burst nhất thời) → Conv1 phải stall ~2000 cycle (mất lợi ích SIMD).

### Giải pháp: wide M10K
- Mỗi channel 1 M10K **rộng 32-bit** (4 position/word). Ghi 4 pooled = 1 word/cycle → khớp 4/cycle →
  **Conv1 hết nghẽn**. KHÔNG cần write-FIFO.
- **Lợi phụ**: đọc word = 4 position liên tiếp → khớp nhu cầu line-buffer mồi VÀ GAP (sum 4 pos/word).
- Giữ **8 M10K riêng/channel** → mồi vẫn nạp 8 channel song song; GAP đọc 8 channel song song.
- Khối cuối < 4 pooled (out_len không chia hết 4) → ghi word với **lane-valid mask**.

> Tension đã giải: mồi multi-buffer *thích* channel-major; ghi-pool *thích* position-major.
> Wide M10K (8 M10K/channel, mỗi M10K 4-pos/word) **dung hòa cả hai** — channel-major ở mức M10K,
> position-major trong word.

---

## 7. Padding = 2 (K=5) — **PP1 + PP4 + lane-valid gate**

Output position `t` cần input `[t-2..t+2]`. Pad zero ở `pos < 0` (trái) và `pos ≥ in_len` (phải).

| Cơ chế | Lo gì | Cách |
|---|---|---|
| **PP4 padded-init** | pad TRÁI | mồi khối đầu: reset line-buffer = 0, nạp data từ slot 2 (offset 2) → 2 zero trái sẵn |
| **PP1 zero-insert** | pad PHẢI | `pad_right = (pos_load ≥ in_len)` → ép 0 vào slot khi nạp |
| **lane-valid gate** | lane THỪA (out_len ⊥ L) | `lane_valid[i] = (block_base+i) < out_len_conv` → gate write từng lane |

- 1 tín hiệu `pad_zero` chung cho mọi buffer (pad theo position, không theo channel).
- Logic GIỐNG NHAU mọi Conv; khác: nguồn data (Conv1=input streaming §3b, Conv2-4=ping-pong) + in_len plumb.
- **Conv1 streaming (§3b)**: dùng counter `stream_cnt` thay địa chỉ; `frame_start` reset + pad trái;
  `stream_cnt ≥ in_len` → pad phải. Conv2-4 dùng `pos_load` (địa chỉ ping-pong). Logic pad bản chất giống nhau.
- **Pad ~free latency** (~vài cycle/layer, gộp trong mồi/transition).

### ⚠️ Rủi ro bug biên (CHÚ Ý KHI VERIFY)
Padding là bài toán **căn chỉnh (alignment)**, không phải logic. SIMD thêm **~5 trục căn chỉnh mới**
(offset mồi PP4, biên PP1 `≥` vs `>`, lane_valid `<` vs `≤`, 8 buffer đồng nhịp, block_base↔pos_load)
→ mỗi trục là 1 off-by-one tiềm năng. Đúng lớp bug lịch sử thesis ("pool window dịch 1 vị trí").
Lỗi chỉ lệch ở **position biên** → ẩn với test thô, khuếch đại ở layer sau.

---

## 8. Controller (FSM) — vòng lặp 3 lồng

### Giữ (từ production)
FSM 8 state (IDLE→LOAD→CONV1-4→GAP_FC→DONE); per-layer config `in_ch/in_len/nb/relu_en/bank_sel`;
GAP/FC/Argmax control (`fc_sub_state/gap_step/fc_step/argmax_step`); `isram_free` (overlap);
`busy/done/result`.

### Đổi
| Tín hiệu | Production | SIMD |
|---|---|---|
| `a` | in-channel counter 0..IN_CH-1 | **giữ ~nguyên** (SIMD vẫn lặp ic tuần tự) |
| `t` | output position +1 | **block counter +L (=20)/phát** |
| `shift_en` | dịch SRW 1 mẫu | **line-buffer stream control** |
| `pong_addr` | địa chỉ channel-major | **địa chỉ wide-word** (+1 mỗi 4 pos) |

### Thêm mới
- **`oc` counter** (0..OUT_CH-1): SIMD lặp output channel tuần tự (thay `cp_en` bitmask song song).
- **`block_base`**: position đầu khối hiện tại (cho lane-valid + địa chỉ ghi).
- **`lane_valid[L-1:0]`** generator: gate khối cuối.
- **`pos_load` + `pad_zero`**: counter nạp + cờ pad (PP1/PP4).
- **line-buffer fill-counter**: pha mồi (đếm L+4) trong mỗi CONV.

### Bỏ
- `cp_en` 8-bit bitmask (→ `oc` counter).
- `prefetch_cnt` cũ (SRW priming 5-shift) (→ fill-counter mới).

### Vòng lặp cốt lõi
```
PRODUCTION (2 lồng):  for t: for a:  [8 oc song song]
SIMD (3 lồng):        for oc: for block(t+=20): for a:  [20 lane position song song]
                      + pha MỒI line-buffer đầu mỗi (oc hoặc layer)
```

---

## 9. GAP / FC / Argmax — **gần như GIỮ NGUYÊN**

- **GAP**: với wide M10K, đọc 8 M10K song song → 8 word32 = 8 ch × 4 position **1 lần** →
  unpack + sum/4 → gap[0..7]. **Nhanh hơn production** (1 đọc thay 4 đọc tuần tự).
- **FC**: ăn gap[0..7] (8 giá trị), 4 neuron × 8-input MAC → **Y HỆT production** (không phụ thuộc
  kiểu conv). Bias pre-scaled 2^w_shift[fc], nb_fc=0.
- **Argmax**: so 4 logit → result[1:0]. **Y HỆT.**
- **MUX read-addr ping-pong** (production `ecg_core.v:96`, chọn Conv↔GAP theo layer_state):
  **VẪN CẦN** (1 read port chia 2 khách, 2 pha). Chỉ đổi nguồn (line-buffer-addr thay cp-engine-addr).
- Phân biệt: MUX **read-addr** (giữ) ≠ MUX **channel-select trong cp_engine** (`srw_flat[a*5]`, BỎ).

---

## 10. Cycle estimate (bậc-một, KHÔNG phải số đo)

Compute/layer ≈ `⌈out_len/20⌉ × OUT_CH × IN_CH`; mồi ≈ L+4/layer.

| Layer | Mồi | Compute | Total |
|---|---|---|---|
| Conv1 | ~6 | ~520 | ~526 |
| Conv2 | ~24 | ~400 | ~424 |
| Conv3 | ~24 | ~160 | ~184 |
| Conv4 | ~24 | ~64 | ~88 |
| GAP/FC/Argmax | — | ~22 | ~22 |
| Transition (4 layer) | — | ~40 | ~40 |
| **TỔNG** | | | **≈ 1284 cycle (~12.8 µs @100MHz)** |

- Speedup vs production (5216 cy): **~4.1×**.
- cycle×DSP: SIMD 1284×70 = **89,880** < production 5216×28 = 146,048 (SIMD tốt hơn area-latency).
- ⚠️ **Sai số ±20-30%** (pipeline fill, transition, mồi, n/phát). Phải sim RTL mới có số thật.

### Bảng DSE đầy đủ (tham khảo)
| Config | Cycle | DSP | Speedup | cycle×DSP | Ghi chú |
|---|---|---|---|---|---|
| **Production 8-lane** | 5216 (đo thật) | 28 | 1× | 146,048 | channel-//, verified |
| SIMD-8 multi | ~3094 | 28 | 1.7× | 86,632 | cùng DSP, pool carry-over (L≠bội5) |
| SIMD-16 multi | ~1854 | 56 | 2.8× | 103,824 | pool carry-over |
| SIMD-20 multi + B | ~1284 | ~70 | 4.1× | 89,880 | **spec này**, pool comb, Conv1 hết nghẽn |
| SIMD-20 single-buffer | ~3930 | 70 | 1.3× | — | reload nặng (KHÔNG dùng) |

---

## 11. Danh sách module cần viết lại / mới

| Module | Việc | Mức |
|---|---|---|
| `cp_engine` → line-buffer engine | thay SRW+MUX-channel bằng 8 multi line-buffer + tap 20 cửa sổ | **viết lại lớn** |
| `cp_block` | 20 lane (thay 8), pool 4-cây comb thay rolling, ghi burst | viết lại |
| `cnn_controller` | 3 vòng lồng (oc×block×a) + pha mồi + lane_valid + pad dải | **viết lại lớn** |
| `ping_pong_sram` | wide M10K 32-bit (4 pos/word) + lane-valid mask write | đổi layout |
| `gap_fc_argmax` | GAP đọc word32 (unpack 4 pos); FC/Argmax giữ | sửa nhỏ GAP |
| **input_buffer (wrapper)** | 2500×8b (= isram dời ra wrapper §3b); xuất stream + frame_start | dời/đổi interface |
| `ecg_core` | nối lại; MUX read-addr giữ; thêm interface stream (stream_data/valid/frame_start) | sửa nối dây |
| `avalon_slave` | giữ nguyên (bus adapter); + nạp input_buffer như isram | ~không đổi |

---

## 12. Verification (BẮT BUỘC mở rộng so với production golden)

Golden production (7 checkpoint **sau pool**, 3 mẫu) **KHÔNG đủ** cho SIMD vì:
- Lỗi bị **MaxPool che** (sai 1 position không phải max) → golden sau-pool không thấy.
- Lỗi **ghi rác ngoài vùng** (lane-valid `<` vs `≤`) nếu vùng không bị đọc lại.
- Lỗi chỉ lộ ở **mẫu/spike cụ thể** → 3 mẫu có thể không kích hoạt.

### Golden cần bổ sung cho SIMD
- [ ] Checkpoint **TRƯỚC pool** (conv raw: 4×2500, 4×500, 8×100, 8×20) — bắt lỗi bị pool che.
- [ ] Kiểm **vùng ghi pong-sram** không có index ngoài out_len (bắt lỗi lane-valid).
- [ ] **Mọi channel** (không chỉ ch0 đại diện) — bắt lỗi buffer lệch nhịp.
- [ ] **Nhiều mẫu** (>3, gồm mẫu có spike ở biên position).
- [ ] **Tolerance 0 LSB** (như production hiện tại).
- [ ] Regression bit-exact với golden Python (cùng QAT-INT8 power-of-2 round-half-up, model 4,4,8,8).

### Bit-exact contract (giữ từ production)
- Quant: `out = clamp(round_half_up(acc/2^nb), -127, 127)`, `round_half_up = (acc + 2^(nb-1)) >> nb`.
- nb per layer: conv1=8, conv2=6, conv3=6, conv4=7, fc=0. w_shift: 6,6,6,7,8. input_shift=2.
- Bias fold vào acc init (như production Fix B): `acc = tree + bias + round_add` tại ic=0.
- GAP: `floor(sum/4) = sum >> 2`. FC: nb=0, raw INT32 logits → argmax. ReLU chỉ Conv4.

---

## 13. Tóm tắt điểm quyết định (cheat-sheet khi code)

1. **L = 20** (bội stride pool 5; chia hết mọi out_len).
2. **8 multi line-buffer**, sâu 24, mồi 1 lần/layer (channel-major nạp 8 ch song song).
2b. **Input buffer ở WRAPPER + core streaming (Mô hình 2, §3b)** — Conv1 đọc stream full-speed từ
    wrapper buffer 2500 (= isram dời ra), decouple nguồn chậm. KHÔNG dùng "core stream không buffer"
    (Mô hình 1 — Conv1 stall theo nguồn). Conv1 pad: `stream_cnt` + `frame_start`.
3. **Pool combinational** 4 cây-max (KHÔNG rolling).
4. **Ghi wide M10K 4-pos/word (Phương án B)** — bắt buộc, để Conv1 (n=1) không tràn. KHÔNG dùng FIFO.
5. **Pad: PP4 (init pad trái) + PP1 (zero-insert pad phải) + lane-valid gate** (khối cuối).
6. **Controller 3 vòng lồng** (oc × block × a) + pha mồi; thêm oc counter, block_base, lane_valid.
7. **GAP/FC/Argmax + MUX read-addr + avalon_slave: GIỮ** (chỉ GAP đọc word32 khác).
8. **Verify: thêm checkpoint trước-pool + mọi channel + nhiều mẫu** (golden sau-pool không đủ).
9. **Conv4 là layer khổ nhất** (ít position, nhiều in-channel, pad nặng) — verify kỹ nhất.
10. **Timing (§2b)**: bỏ path `a→MUX-channel` của production; xử lý 2 path mới — **Pool 2-stage
    pipeline** (depth-3, 4 comparator) + **Weight fan-out W1 (4 bản nhóm) → W3 (20 bản) nếu fail**.
    Fmax phải synth mới chốt.
11. **Mục tiêu: latency ~4.1× nhanh hơn production** — đổi lại DSP ~70 (63%), power cao hơn,
    controller phức tạp hơn, phải verify lại (§0, §12).
