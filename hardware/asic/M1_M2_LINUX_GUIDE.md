# M1 + M2 — Hướng dẫn chi tiết cho session Linux (OpenLane / Sky130)

> Tài liệu TỰ-CHỨA. Session Linux đọc file này là đủ để chạy, không cần context phiên trước.
> Bối cảnh tổng: [PLAN_OPENLANE.md](PLAN_OPENLANE.md) · trạng thái M0: [README.md](README.md)

## 0. Tình trạng kế thừa từ M0 (đã xong, bit-exact)

M0 đã refactor memory sang macro-friendly và verify 21/21 bit-exact trên Questa
(max|diff|=0, 5216 cy). Các file ASIC nằm trong `hardware/asic/rtl/`:

| File | Vai trò | Memory bên trong |
|---|---|---|
| `rtl/ecg_core_asic.v` | **Chip top thật** (8-wire parallel interface) | instantiate 2 memory dưới |
| `rtl/input_sram_asic.v` | input SRAM | `reg [7:0] mem [0:4095]` (4096×8) |
| `rtl/ping_pong_sram_asic.v` | ping-pong | `reg [63:0] mem_a/mem_b [0:511]` (2×512×64, per-byte wmask) |
| `rtl/ecg_accelerator_top_asic.v` | wrapper Avalon — **CHỈ để sim, KHÔNG synth** | — |

Module logic dùng chung từ `hardware/RTL/` (technology-agnostic, KHÔNG có bản asic):
`cp_block.v`, `cp_engine.v`, `cnn_controller.v`, `gap_fc_argmax.v`.
(KHÔNG đưa `avalon_slave.v` vào synth ASIC — nó chỉ thuộc wrapper sim.)

## 1. File hex cần (đặt CÙNG cwd khi synth + khi sim)

7 file, nạp qua `$readmemh` trong `initial`:

| File | Module nạp | #dòng | Định dạng |
|---|---|---|---|
| `conv1_w.hex` | cp_engine | 4  | 40-bit/dòng (5 tap × 8b packed) |
| `conv2_w.hex` | cp_engine | 16 | 40-bit/dòng |
| `conv3_w.hex` | cp_engine | 32 | 40-bit/dòng |
| `conv4_w.hex` | cp_engine | 64 | 40-bit/dòng |
| `conv_bias.hex` | cp_engine | 32 | 32-bit/dòng (INT32 LE) |
| `fc_weights.hex` | gap_fc_argmax | 32 | 8-bit/dòng (INT8, addr=k*8+i) |
| `fc_bias.hex` | gap_fc_argmax | (nhỏ) | 32-bit/dòng |

Nguồn gốc các file: `hardware/RTL/*.hex` (bản chuẩn) và bản copy ở
`hardware/fpga/simulation/questa/*.hex`. Mang nguyên 7 file sang Linux.

---

## 2. Milestone 1 — Generate 3 SRAM macro (OpenRAM, Sky130)

### 2.1. BLOCKER phải verify TRƯỚC khi generate (đừng bỏ qua)

1. **OpenRAM sky130 có dual-port + per-byte wmask không?**
   RTL cần mỗi bank: 1 write-port + 1 read-port ĐỒNG THỜI (địa chỉ khác nhau),
   1-cycle sync read. `ping_pong_sram_asic` còn cần **byte write-mask** (8 byte/word).
   - Cấu hình OpenRAM cần: `num_rw_ports`/`num_r_ports`/`num_w_ports` sao cho có
     1 write + 1 read song song (vd `num_rw_ports=1, num_r_ports=1`), và bật
     `write_size` = 8 (byte-writable) cho macro 512×64.
   - NẾU OpenRAM sky130 KHÔNG hỗ trợ wmask trên config dual-port → 2 phương án dự phòng:
     (a) chia mỗi bank 512×64 thành 8 macro 512×8 (per-channel we riêng — quay lại
         16 macro, đúng nhưng tốn công), hoặc
     (b) đổi `ping_pong_sram_asic` sang **read-modify-write** trong RTL (đọc word,
         ghép byte mới, ghi lại) — NHƯNG cần thêm 1 read-port hoặc 1 chu kỳ; rủi ro
         phá bit-exact + tăng area (đã thấy nổ ALM trên FPGA). Tránh nếu được.
   - → Quyết định (a) vs (b) sau khi biết OpenRAM hỗ trợ tới đâu. Ghi lại lựa chọn.

2. **`$readmemh` trong yosys** — synth `cp_engine` + `gap_fc_argmax` ra netlist, kiểm tra
   weight được suy thành ROM đúng giá trị. Test nhanh:
   ```bash
   # đứng ở thư mục có 7 file hex
   yosys -p "read_verilog -sv \
       <asic>/rtl/ecg_core_asic.v <asic>/rtl/input_sram_asic.v \
       <asic>/rtl/ping_pong_sram_asic.v \
       <RTL>/cp_block.v <RTL>/cp_engine.v <RTL>/cnn_controller.v <RTL>/gap_fc_argmax.v; \
       hierarchy -top ecg_core_asic; synth -top ecg_core_asic; stat"
   ```
   - Nếu yosys báo lỗi không đọc được hex (sai cwd / path tương đối) → CHẠY yosys TỪ
     thư mục chứa hex, hoặc tạm sửa path `$readmemh` thành tuyệt đối.
   - Nếu yosys đọc được nhưng tối ưu hết weight thành const (mong muốn — weight là ROM
     cố định) → OK.
   - DỰ PHÒNG nếu `$readmemh` không hoạt động trong flow: sinh ROM logic (case
     statement) từ hex bằng Python, thay 2 khối `initial $readmemh`. (Script chưa viết —
     viết khi cần.)

3. **ramstyle MLAB / M10K** — chỉ là attribute Altera, yosys bỏ qua. b_store thành FF
   (32×32b — nhỏ, chấp nhận). KHÔNG cần xử lý.

### 2.2. 3 macro cần generate

| Macro | Kích thước | Dùng cho | Ghi chú |
|---|---|---|---|
| `sram_4096x8`  | 4096 word × 8b  | input_sram_asic | single byte/word, no wmask |
| `sram_512x64`  | 512 word × 64b  | ping_pong bank A | **byte wmask (8)** |
| `sram_512x64`  | (cùng config)   | ping_pong bank B | instantiate macro 2 lần |

OpenRAM config mẫu (điều chỉnh theo version OpenRAM thực tế):
```python
# myconfig_512x64.py
word_size = 64
num_words = 512
write_size = 8           # byte write-mask -> 8 mask bits
num_rw_ports = 1
num_r_ports  = 1         # 1 write/read-write + 1 read = đọc & ghi song song
num_w_ports  = 0
tech_name = "sky130"
process_corners = ["TT"]
supply_voltages = [1.8]
temperatures = [25]
output_path = "macro/sram_512x64"
output_name = "sram_512x64"
```
Chạy: `python3 $OPENRAM_HOME/sram_compiler.py myconfig_512x64.py`
→ sinh `sram_512x64.gds`, `.lef`, `.lib`, `.v` (behavioral). Tương tự cho 4096x8.

### 2.3. Thay 2 module memory bằng macro

Có 2 cách:
- **Cách A (sạch)**: tạo `rtl/*_sram_macro.v` wrapper map port RTL ↔ port macro
  (tên port macro do OpenRAM đặt: `clk0/csb0/web0/wmask0/addr0/din0/dout0` + port1...).
  Giữ `ecg_core_asic.v` instantiate wrapper → interface không đổi.
- **Cách B**: sửa thẳng `ecg_core_asic.v` instantiate macro. Ít file hơn nhưng bẩn.
→ Khuyến nghị A. Map cẩn thận: 1-cycle sync read của RTL phải khớp latency macro.

### 2.4. Re-verify bit-exact với behavioral model của macro

Sửa `sim/run_tb_top_asic.do` (hoặc bản iverilog tương đương trên Linux) để compile
behavioral `.v` của macro thay cho `*_sram_asic.v`, chạy lại `tb_top_asic`:
```
# kỳ vọng: 10/10 + 21/21 bit-exact, max|diff|=0, 5216 cy
```
Nếu KHÔNG khớp → vấn đề ở map port/latency macro hoặc wmask. Sửa tới khi bit-exact.
(Trên Linux không có Questa → dùng `iverilog -g2012` + `vvp`, hoặc Verilator.
 tb_top_asic.v là Verilog-2001/SV nhẹ, iverilog chạy được.)

→ **M1 DONE khi**: 3 macro generate xong + tb_top_asic bit-exact với behavioral model.

---

## 3. Milestone 2 — OpenLane flow → GDSII

### 3.1. File config

Tạo `hardware/asic/openlane/config.json` (OpenLane 2 / Nix flow):
```json
{
  "DESIGN_NAME": "ecg_core_asic",
  "VERILOG_FILES": [
    "dir::../rtl/ecg_core_asic.v",
    "dir::../rtl/input_sram_macro.v",
    "dir::../rtl/ping_pong_sram_macro.v",
    "dir::../../RTL/cp_block.v",
    "dir::../../RTL/cp_engine.v",
    "dir::../../RTL/cnn_controller.v",
    "dir::../../RTL/gap_fc_argmax.v"
  ],
  "CLOCK_PORT": "clk",
  "CLOCK_PERIOD": 10.0,
  "PDK": "sky130A",

  "VERILOG_FILES_BLACKBOX": [],
  "EXTRA_LEFS": ["dir::macro/sram_4096x8/sram_4096x8.lef",
                 "dir::macro/sram_512x64/sram_512x64.lef"],
  "EXTRA_LIBS": ["dir::macro/sram_4096x8/sram_4096x8.lib",
                 "dir::macro/sram_512x64/sram_512x64.lib"],
  "EXTRA_GDS_FILES": ["dir::macro/sram_4096x8/sram_4096x8.gds",
                      "dir::macro/sram_512x64/sram_512x64.gds"],
  "MACRO_PLACEMENT_CFG": "dir::macro_placement.cfg",

  "FP_PDN_MULTILAYER": true,
  "RUN_HEURISTIC_DIODE_INSERTION": true
}
```
Lưu ý:
- KHÔNG đưa `ecg_accelerator_top_asic.v` / `avalon_slave.v` vào (chỉ thuộc sim).
- 7 file hex phải nằm ở cwd lúc synth (yosys đọc `$readmemh`). Đặt symlink/copy vào
  thư mục run của OpenLane, hoặc set path tuyệt đối nếu flow đổi cwd.
- `macro_placement.cfg`: liệt kê vị trí 3 macro, vd:
  ```
  u_isram        300 300 N
  u_pp_a         300 900 N
  u_pp_b         900 900 N
  ```
  (tên instance khớp trong ecg_core_asic.v: `u_isram`, và 2 instance ping_pong).

### 3.2. Chạy flow + tiêu chí signoff

```bash
openlane hardware/asic/openlane/config.json
# OpenLane 1 cổ điển: ./flow.tcl -design ecg_core_asic
```
Lặp tới khi TẤT CẢ:
- DRC clean (magic / klayout)
- LVS clean (netgen)
- STA: **không hold violation**, setup slack ≥ 0
  - Nếu setup fail @100MHz (CLOCK_PERIOD=10) → nới CLOCK_PERIOD, đo Fmax thật
    (giống Phase C FPGA: report Fmax + WNS).

### 3.3. Thu số liệu cho paper

Từ `runs/<tag>/reports/` + `metrics.json`:
- **Die area** (µm²) + diện tích 3 macro vs logic.
- **Gate count** logic (số cell standard).
- **Power**: internal + switching + leakage (OpenSTA / report power). Energy/inf =
  Power × (5216 cy / Fmax).
- **Fmax** đạt được.
→ Tạo bảng "ASIC Sky130" cạnh cột FPGA (Phase C: DSP 28, ALM 2261, Fmax~137MHz).

→ **M2 DONE khi**: GDSII signoff clean (DRC/LVS) + no hold + setup≥0 + đã ghi số liệu.

---

## 4. Checklist tóm tắt cho session Linux

- [ ] Mang sang: `hardware/asic/` + `hardware/RTL/*.v` + 7 file hex.
- [ ] M1.1 verify blocker: OpenRAM dual-port+wmask? `$readmemh` trong yosys?
- [ ] M1.2 generate 3 macro (sram_4096x8 + 2× sram_512x64 wmask).
- [ ] M1.3 wrapper map port + thay vào ecg_core_asic.
- [ ] M1.4 re-verify tb_top_asic bit-exact (iverilog/verilator) với behavioral model.
- [ ] M2.1 viết config.json + macro_placement.cfg.
- [ ] M2.2 chạy OpenLane tới DRC/LVS clean + no hold + setup≥0.
- [ ] M2.3 thu area/power/Fmax → bảng ASIC.
- [ ] Cập nhật memory `asic-openlane-m0.md` (hoặc tạo `asic-openlane-m1m2.md`).

## 5. Cạm bẫy đã biết (đừng lặp lại)

- Đừng "ghi cả word 64-bit bỏ channel inactive" trong ping_pong — M0 đã loại vì rủi ro
  bit-exact. Giữ per-byte wmask.
- Đừng đưa avalon_slave/wrapper sim vào synth ASIC.
- 7 file hex PHẢI cùng cwd khi synth, nếu không yosys fail `$readmemh` im lặng hoặc lỗi.
- 1-cycle sync read latency của RTL phải khớp đúng latency macro — sai latency = lệch
  bit-exact ở pool window (xem lịch sử bug d6→d5 trong System_Design.md).
