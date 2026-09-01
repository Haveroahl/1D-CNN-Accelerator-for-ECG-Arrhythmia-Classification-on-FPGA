# spec_fail.md — Các điểm RTL SIMD-20 LỆCH so với SIMD.md

> ## ✅ TRẠNG THÁI (cập nhật): pipelined + wide-4 + settle=2, **2755 cy** bit-exact 93384/93384
> - **SF-3 (wide-read) RESOLVED**: `input_buffer.v` 625×32-bit (4 pos/word) +
>   `line_buffer_engine.v` `wide_load` (4 pos/cy). Áp cho cả Conv1 input lẫn Conv2-4
>   pong. → 13827 → 11935 cy, vẫn bit-exact.
> - **Drain stall (mục dưới) RESOLVED**: controller viết lại dạng **pipelined
>   issue↔writeback** — `PH_SWEEP` bơm (oc,a) liên tục, writeback chạy độc lập theo
>   `pooled_valid` với counter `wb_oc`/`wb_block` (không FIFO). `in_flight` gate
>   `PH_TRANS`. → 11935 → 3223 cy, vẫn bit-exact.
> - **Settle 5→2cy**: cắt biên an toàn dư của `PH_*_SETTLE` (settle=1 vỡ — data cần
>   1cy shift_d + 1cy ghi line-buffer). → 3223 → **2755 cy** (1.89× nhanh hơn prod), bit-exact.
> - **Bài học tb (KHÔNG phải lỗi RTL)**: `lane_valid` (S8) dẫn `pooled_valid` (sau pool
>   2-stage) đúng **2 cy**. Bản tb cũ đọc `wb_oc/wb_block` (bám `pooled_valid`) tại thời
>   điểm `lane_valid` → lệch 2 kết quả → BÁO 44526 "lỗi" GIẢ. Sửa: tb dùng counter replica
>   riêng (`cap_oc`/`cap_block`) tăng theo `lane_valid`. RTL không sai chỗ này.
> - **Synth**: Fmax 104.85 MHz (meets 100MHz), ALM 16,976 (41%), DSP 64 (57%).
> - SF-1/SF-2/SF-4: model này mọi out_len ⋮20 → gate numeric vô hại; cơ chế giữ verifiable.

> Phạm vi: chỉ liệt kê chỗ RTL trong `hardware/RTL_simd/` **làm khác / thiếu so với
> điều SIMD.md đã chốt**. KHÔNG liệt kê bug logic chung (timing off-by-one, v.v.) —
> những cái đó spec không quy định nên không tính là "lệch spec".
>
> Đối chiếu: SIMD.md (branch `feature/simd-spec`) ↔ RTL_simd/*.v
> Mức độ: 🔴 lệch nặng (cơ chế spec bắt buộc bị bỏ) · 🟠 lệch trung bình (làm khác
> cách spec mô tả) · 🟡 lệch nhẹ (không đạt mục tiêu spec, chức năng vẫn chạy).

---

## SF-1 🔴 Lane-valid gate KHÔNG được hiện thực (chỉ stub all-ones)

### Spec yêu cầu
- §7 bảng cơ chế padding:
  > **lane-valid gate** | lane THỪA (out_len ⊥ L) | `lane_valid[i] = (block_base+i) < out_len_conv` → gate write từng lane
- §8 "Thêm mới":
  > **`lane_valid[L-1:0]`** generator: gate khối cuối.
- §13 điểm 5:
  > Pad: PP4 (init pad trái) + PP1 (zero-insert pad phải) + **lane-valid gate** (khối cuối).
- §12 (verification) yêu cầu golden phải:
  > Kiểm **vùng ghi pong-sram** không có index ngoài out_len (bắt lỗi lane-valid).

### RTL thực tế
- `ecg_core_simd.v:142` — `simd_pool` nhận `.lane_valid_mask({L{1'b1}})` (hardwire toàn 1).
- `ecg_core_simd.v:164` — `pong_sram_wide` nhận `.wmask(4'hF)` (hardwire toàn 1).
- `simd_controller.v` — tính `block_base` (output reg) nhưng **KHÔNG sinh tín hiệu
  `lane_valid` nào**. Không có port `lane_valid` ra khỏi controller.
- `simd_pool.v:28-42` — module CÓ hỗ trợ `lane_valid_mask` (force -128), nhưng không ai
  drive nó ≠ all-ones.

### Hệ quả
- Với model hiện tại mọi `out_len` (2500/500/100/20) chia hết L=20 → numeric vô hại.
- NHƯNG: cơ chế spec bắt buộc **không tồn tại** → §12 golden "kiểm vùng ghi pong-sram
  không index ngoài out_len" **không có gì để verify** (gate không tồn tại thì không
  bắt được lỗi ghi rác).
- Đây là điểm lệch spec rõ nhất: spec liệt kê lane-valid generator là "Thêm mới" bắt buộc
  trong §8, RTL bỏ trống.

### Cần sửa
- Controller sinh `lane_valid[L-1:0]` = `(block_base + i) < out_len` cho mỗi lane.
- Dẫn vào `simd_pool.lane_valid_mask` và dẫn `wmask` của lane cuối-trong-word vào
  `pong_sram_wide.wmask` (cho khối cuối lẻ word — dù model này không kích hoạt).
- File động tới: `simd_controller.v` (thêm generator + port), `ecg_core_simd.v` (nối dây).

---

## SF-2 🟠 Padding Conv1 dùng `pos_load` thay vì `stream_cnt`/`frame_start` (sai Mô hình 2)

### Spec yêu cầu
- §3b CHỐT "Mô hình 2: buffer ở wrapper + core streaming", interface:
  > `stream_data[7:0]`, `stream_valid` — 1 mẫu/cycle vào core.
  > `frame_start` — wrapper báo bắt đầu cửa sổ 2500 mới → core reset `stream_cnt`, reset line-buffer.
- §3b "Padding khi streaming (Mô hình 2) — counter thay địa chỉ":
  > Streaming không có địa chỉ → dùng counter `stream_cnt` (0..in_len-1, reset mỗi `frame_start`):
  > - Pad trái (PP4): `frame_start` → reset line-buffer.
  > - Pad phải (PP1): `stream_cnt ≥ in_len` → ép 0.
- §13 điểm 2b:
  > Conv1 pad: `stream_cnt` + `frame_start`.

### RTL thực tế
- `simd_controller.v` **không có** `stream_cnt`, **không có** `frame_start`.
- Conv1 dùng CHUNG đường địa chỉ `pos_load` với Conv2-4:
  - PRIME: `pos_load <= block_base + prime_cnt` (`:122`), pad bằng so sánh địa chỉ (`:123-124`).
  - SLIDE: `pos_load <= block_base + 24 + slide_cnt` (`:189`), pad `(block_base+22+slide_cnt) >= in_len` (`:190`).
- `ecg_core_simd.v:65` — `log_pos = c_pos_load - 12'd2` rồi `ibuf_rd_addr = log_pos`
  → Conv1 đọc input_buffer **bằng địa chỉ**, đúng kiểu §3b gọi là cách "đọc bằng địa chỉ"
  (Mô hình 1), KHÔNG phải streaming + counter (Mô hình 2).
- `input_buffer.v` — chỉ là RAM đọc/ghi địa chỉ thường, không xuất `stream_valid`/`frame_start`.

### Hệ quả
- Có thể tương đương về kết quả số (vì địa chỉ ↔ counter ánh xạ 1-1), NHƯNG:
  - Mất ý nghĩa "decouple nguồn chậm" của Mô hình 2 (§3b lý do chọn) — core vẫn phụ thuộc
    địa chỉ buffer, không có handshake `frame_start`.
  - Không có `frame_start` reset cửa sổ → không hiện thực được "overlap window N+1" (xem SF-4).
- Lệch spec về **cơ chế padding + interface streaming Conv1**.

### Cần sửa
- Thêm `stream_cnt` counter trong controller (reset mỗi frame), pad theo `stream_cnt`.
- Thêm `frame_start` từ wrapper; `input_buffer` xuất `stream_valid`.
- File động tới: `simd_controller.v`, `input_buffer.v`, `ecg_core_simd.v`.

---

## SF-3 🟠 Mồi (priming) đọc 1 byte/cycle thay vì đọc wide 4-pos/word

### Spec yêu cầu
- §6 (Phương án B), lợi ích wide M10K:
  > **Lợi phụ**: đọc word = 4 position liên tiếp → khớp nhu cầu line-buffer **mồi** VÀ GAP.
- §4 quá trình mồi:
  > nạp L+4=24 position của 8 channel **song song** ... (~24 cycle, **hoặc ~6 nếu đọc wide**).
- §10 cycle estimate dựa trên mồi nhanh:
  > Conv1 Mồi ~6; Conv2-4 Mồi ~24. TỔNG ≈ 1284 cycle.

### RTL thực tế
- `ecg_core_simd.v:68-70`:
  ```
  wire [6:0] prime_word = log_pos[8:2];
  wire [1:0] prime_byte = log_pos[1:0];
  ```
- `ecg_core_simd.v:99-103` — chọn **1 byte** từ word theo `byte_d`, nạp 1 position/cycle
  vào line-buffer.
- `simd_controller.v` PRIME (`:120-130`) và SLIDE (`:188-198`) — stream 1 position/cycle,
  PRIME đếm tới `L+4-1` (24 cycle), SLIDE đếm `L` (20 cycle).

### Hệ quả
- Đọc lại cùng 1 word M10K tới 4 lần → **không khai thác wide-read** mà §6 quảng cáo.
- Mồi Conv2-4 = ~24 cycle (không phải ~6) → **latency thực > §10 ước lượng ~1284 cy**.
- Đây là lệch "không đạt mục tiêu §0 (tối ưu latency)" — chức năng vẫn đúng.

### Cần sửa (nếu muốn đạt §10)
- Mồi đọc nguyên word 4-pos, nạp 4 position/cycle vào line-buffer (cần line-buffer
  nhận shift 4 hoặc nạp song song 4 slot).
- File động tới: `line_buffer_engine.v` (nhận nạp 4-wide), `ecg_core_simd.v`, `simd_controller.v`.
- ⚠️ Tùy chọn — nếu chấp nhận latency cao hơn §10 thì để nguyên, nhưng phải cập nhật
  con số §10/§0 cho trung thực (không còn ~4.1×).

---

## SF-4 🟠 `frame_start` + overlap reload (window N+1) thiếu

### Spec yêu cầu
- §3b:
  > Overlap: wrapper buffer + free-flag (như `isram_free`) → nạp window N+1 **song song**
  > compute N.
- §8 "Giữ (từ production)": `isram_free` (overlap).
- §3b interface: `frame_start`.

### RTL thực tế
- `simd_controller.v:212` — có xuất `isram_free<=1` khi vào CONV2 (giống production).
- NHƯNG **không có `frame_start`**, `input_buffer.v` không có free-flag / double-buffer /
  cơ chế "đủ 2500 → frame_ready → core start".
- Interface streaming §3b (`stream_data` ✅ / `stream_valid` ❌ / `frame_start` ❌)
  chỉ hiện thực 1/3.

### Hệ quả
- `isram_free` xuất ra nhưng không có phía wrapper tiêu thụ để overlap → cờ "treo".
- Không đạt overlap window N+1 mà §3b mô tả.
- Lệch spec về **interface + cơ chế overlap**.

### Cần sửa
- `input_buffer` thành double-buffer (hoặc free-flag) + `frame_start` handshake.
- File động tới: `input_buffer.v`, `ecg_core_simd.v`, `simd_controller.v`.

---

## Bảng tổng hợp lệch spec

| ID | Mức | Điều khoản spec | File RTL | Tóm tắt |
|----|-----|-----------------|----------|---------|
| SF-1 | 🔴 | §7, §8, §12, §13-5 | `ecg_core_simd.v:142,164` · `simd_controller.v` | lane-valid gate stub all-ones, generator không tồn tại |
| SF-2 | 🟠 | §3b, §13-2b | `simd_controller.v` · `input_buffer.v` · `ecg_core_simd.v:65` | Conv1 pad dùng `pos_load` địa chỉ thay `stream_cnt`/`frame_start` |
| SF-3 | 🟠 | §4, §6, §10 | `ecg_core_simd.v:68-70,99-103` · `simd_controller.v` | mồi 1 byte/cycle, không đọc wide → latency > §10 |
| SF-4 | 🟠 | §3b, §8 | `input_buffer.v` · `simd_controller.v` | `frame_start`/overlap window N+1 thiếu; `isram_free` treo |

## Các phần KHÔNG lệch spec (đã đối chiếu, ĐÚNG)

| Điều khoản | File | Trạng thái |
|-----------|------|------------|
| L=20, 4 cây pool comb 2-stage pipeline (§5, §2b) | `simd_pool.v` | ✅ |
| Weight fan-out W1 (4 bản nhóm) + cờ W3 fallback (§2b) | `simd_lane_array.v:56-87` | ✅ |
| Wide M10K 4-pos/word, 8 M10K/ch × 2 bank (§6) | `pong_sram_wide.v` | ✅ |
| Bit-exact: nb={8,6,6,7,0}, w_shift, round-half-up, bias fold acc init (§12) | `simd_lane_array.v` · `simd_weight_rom.v` | ✅ |
| GAP floor(sum/4)=sum>>2, FC nb=0, ReLU chỉ Conv4 (§12) | `gap_fc_argmax_simd.v` | ✅ |
| GAP/FC/Argmax + MUX read-addr giữ nguyên (§9) | `gap_fc_argmax_simd.v` · `ecg_core_simd.v:73` | ✅ |
| 8 multi line-buffer sâu 24, tap tĩnh (§4, §2b) | `line_buffer_engine.v` | ✅ |
| Weight ROM layout = production (§12) | `simd_weight_rom.v` | ✅ (khớp `cp_engine.v`) |

---

## Lưu ý: bug KHÔNG thuộc phạm vi "lệch spec" (spec không quy định)

Ghi lại để không lẫn — đây là bug logic/timing, cần fix nhưng KHÔNG phải lệch spec:
- **B-1** `pong_waddr` (registered `<= block_idx` ở `simd_controller.v:169`) trễ `pong_we`
  1 cycle → ghi đúng data/we nhưng địa chỉ lệch 1 cycle.
- **B-2** Conv1 (IN_CH=1): sweep `a=0` kéo 2 cycle (`sweep_armed`, `simd_controller.v:152-163`)
  → tích lũy `a=0` mơ hồ, chưa chứng minh bit-exact.
- **B-3** GAP sum ở `gap_step==2` (`gap_fc_argmax_simd.v:67`) có thể lệch 1 cycle so với
  bank_sel_d + read latency.

(Chi tiết 3 bug này nằm ngoài phạm vi spec_fail — xử lý ở pass debug logic riêng.)

---

## SF-5 🔴 Controller KHÔNG giữ pipeline đầy → latency thực >> §10 (~1284 cy)

> Đây là điểm khiến kết luận "~4.1× nhanh hơn production" của §0/§10 **đảo ngược**.
> Phân loại: lệch **mục tiêu §0 (tối ưu latency)** do RTL, KHÔNG phải lỗi mô hình spec.

### Spec ước lượng (§10) — đúng cho SIMD pipelined-đầy LÝ TƯỞNG
- §10 công thức: `Compute/layer ≈ ⌈out_len/20⌉ × OUT_CH × IN_CH`; mồi ≈ L+4/layer.
- Ngầm định: pipeline **đầy** → độ trễ khởi động (MAC 5 + requant 5 + pool 2 ≈ **12 cy**)
  chỉ trả **MỘT lần**, sau đó mỗi cycle bơm 1 (block,oc) mới + lấy 1 kết quả ra.
- Với giả định đó: Conv1 ~526, tổng ~1284 cy, speedup ~4.1×. **Số học §10 không vô lý
  dưới giả định pipeline đầy.**

### RTL thực tế — controller TUẦN TỰ, stall mỗi (block,oc)
- `simd_controller.v` chạy nối tiếp 3 pha: `PH_OC_RST → PH_SWEEP → PH_DRAIN`, và
  **PH_DRAIN ĐỨNG ĐỢI `pooled_valid`** mới sang oc/block kế:
  ```
  PH_DRAIN: if (pooled_valid) begin ... oc<=oc+1; phase<=PH_OC_RST; end  // chờ rồi mới đi
  ```
- Nó **không bơm (block,oc) kế trong lúc pipeline đang chảy** → pipeline RỖNG ~12 cy
  giữa mỗi oc. Drain bị trả **lặp lại mỗi (block,oc)**, không phải 1 lần.
- Ngoài ra SLIDE (`PH_SLIDE`/`PH_WSLIDE`) chạy **nối tiếp** sau DRAIN (không overlap
  compute) → cộng thêm ~8-22 cy/block.

### Hệ quả định lượng (ước lượng bậc-một)
Per (block,oc): `OC_RST(1) + arm(1) + sweep(IN_CH) + drain(~12)` ≈ `IN_CH + 14` cy.
Per block thêm slide ~8-22 cy.

| Layer | #block | cost/block (RTL tuần tự) | ≈ cycle |
|---|---|---|---|
| Conv1 (IN_CH=1, OC=4) | 125 | 4×(1+1+1+12) + slide~22 ≈ 82 | **~10,250** |
| Conv2 (IN_CH=4, OC=4) | 25  | 4×(1+1+4+12) + ~10 ≈ 82 | ~2,050 |
| Conv3 (IN_CH=4, OC=8) | 5   | 8×(1+1+4+12) + ~10 ≈ 154 | ~770 |
| Conv4 (IN_CH=8, OC=8) | 1   | 8×(1+1+8+12) ≈ 176 | ~176 |
| GAP/FC/Argmax | — | — | ~22 |
| **TỔNG RTL** | | | **~13,000–14,000 cy** |

→ So §10 ghi **~1284 cy**: sai số **~10×** (vượt xa disclaimer "±20-30%" của §10).
→ So production 5216 cy: RTL SIMD hiện tại **CHẬM HƠN** production (đảo ngược claim §0).

### Vì sao Conv1 sập nặng nhất (drain không khấu hao được)
- Drain ~12 cy là **cố định/(block,oc)**. Khấu hao tốt khi mỗi oc có nhiều cy compute.
- Conv1 IN_CH=1 → mỗi oc chỉ **1 cy compute** vs **12 cy drain** → phí 12/13. Lại có
  **125 block × 4 oc = 500 lần** trả drain → thống trị toàn bộ latency.
- Production xử Conv1 bằng stream 1 mẫu/cycle **liên tục, pipeline luôn đầy** → 2500 cy
  thẳng (drain trả 1 lần). Đó là lý do production thắng ở Conv1.
- ⇒ SIMD position-parallel hợp **IN_CH lớn** (drain bị che); Conv1 (IN_CH=1) là case
  XẤU NHẤT — ngược trực giác "chuỗi dài thì SIMD thắng".

### Cần sửa (để RTL đạt ~§10)
- **Decouple issue ↔ writeback**: bơm (block,oc) kế NGAY, KHÔNG đợi `pooled_valid`;
  đường ghi pong-sram chạy độc lập theo strobe `pooled_valid` (FIFO/skid nhỏ giữ
  block_idx+oc kèm kết quả). → pipeline đầy, drain trả 1 lần/layer.
- **Overlap SLIDE với compute**: slide line-buffer cho block kế trong lúc compute
  block hiện tại (cần line-buffer cho phép đọc tap khối hiện tại trong khi shift khối kế,
  hoặc double line-buffer).
- File động tới: `simd_controller.v` (viết lại FSM dạng pipelined/issue-writeback) —
  **đây là "viết lại lớn" §11 chưa thực sự đạt**; `ecg_core_simd.v` (đường ghi tách);
  có thể `line_buffer_engine.v` (double-buffer cho overlap slide).
- ⚠️ Đây là phần KHÓ NHẤT của variant (đúng như §0 cảnh báo "controller phức tạp hơn").
  RTL hiện tại mới đạt bản tuần tự (chạy đúng số học nếu fix bug, nhưng KHÔNG đạt latency).

### Đính chính phân loại
- Câu phân tích trước từng quy "§10 sai mô hình". Chính xác hơn: **§10 đúng cho SIMD
  pipelined-đầy lý tưởng; RTL hiện tại lệch vì controller tuần tự (stall PH_DRAIN)**
  không giữ pipeline đầy → đây là **lệch spec (mục tiêu latency §0/§10)**, không phải
  spec tự sai. Spec chỉ thiếu sót: §10 KHÔNG nói rõ "yêu cầu controller pipelined
  issue-writeback" — nên bổ sung ghi chú đó vào §8/§10 để tránh hiểu nhầm.
