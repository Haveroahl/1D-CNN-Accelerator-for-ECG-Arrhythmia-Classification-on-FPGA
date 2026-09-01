# ASIC flow (OpenLane / Sky130) — ecg_core

Đưa `ecg_core` (đã verify bit-exact trên FPGA) qua OpenLane ra GDSII, lấy số liệu
area/power/timing làm cột "ASIC" cạnh cột FPGA cho paper. Kế hoạch đầy đủ:
[PLAN_OPENLANE.md](PLAN_OPENLANE.md).

## Trạng thái

- **M0 — Chuẩn bị RTL (DONE, verify trên Windows/Questa)**
  - `rtl/ping_pong_sram_asic.v` — 16 mảng 512×8 → **2 mảng 512×64** (per-byte wmask).
  - `rtl/input_sram_asic.v` — 2500×8 → **4096×8** (depth lũy thừa 2 cho macro).
  - `rtl/ecg_core_asic.v` — core dùng 2 memory trên, expose 8-wire parallel interface.
  - `rtl/ecg_accelerator_top_asic.v` — wrapper Avalon CHỈ để regression (không phải chip top).
  - **Verify**: `sim/tb_top_asic.v` → **10/10 + 21/21 bit-exact, max|diff|=0 LSB,
    15312/15312 exact, 5216 cycles** — pack không đổi logic.
- **M1 — Generate OpenRAM macro (TODO, Linux/WSL)**
- **M2 — OpenLane tích hợp macro → GDSII (TODO, Linux/WSL)**

## Chạy regression M0 (Windows, Questa)

```powershell
cd d:\Thesis101\hardware\fpga\simulation\questa   # cwd có conv*_w.hex, golden/, ecg_sample*.hex
& "D:\altera_lite\25.1std\questa_fse\win64\vsim.exe" -c `
    -do "do D:/Thesis101/hardware/asic/sim/run_tb_top_asic.do; quit -f"
```
Kỳ vọng: `ALL TESTS PASSED (incl. per-layer bit-exact)`, `max|diff|=0`.

## Blocker cần verify đầu M1 (trên Linux, trước khi đi xa)

1. **`$readmemh` trong yosys** — `cp_engine.v` nạp weight bằng 5 file `$readmemh`
   (`conv1_w.hex`..`conv4_w.hex`, `conv_bias.hex`) trong `initial`. Yosys CÓ hỗ trợ
   `$readmemh` trong initial → suy ra ROM. **Phải verify**: yosys đọc đúng + giá trị
   khớp. Nếu lỗi → sinh ROM logic (case) từ hex bằng script Python.
   - Lưu ý: hex phải nằm cùng cwd khi chạy synth (như Questa cần cwd = thư mục questa).
2. **`(* ramstyle = "MLAB" *)`** trên `b_store` (cp_engine.v) — pragma Altera, yosys
   bỏ qua (chỉ là attribute, không lỗi). b_store sẽ thành FF — chấp nhận được (nhỏ, 32×32b).
3. **Macro dual-port** — RTL cần 1 write + 1 read ĐỒNG THỜI (địa chỉ khác nhau) mỗi bank.
   Xác nhận OpenRAM sky130 hỗ trợ cấu hình dual-port (1RW1R / 2 port) TRƯỚC khi generate.
   `ping_pong_sram_asic` còn cần **per-byte write mask** (wmask) → check OpenRAM hỗ trợ wmask.

## M1 — Generate macro (Linux/WSL, phác thảo)

3 macro Sky130 cần generate:
- `sram_4096x8`  — input_sram.
- `sram_512x64` ×2 — ping_pong bank A/B (cần wmask byte-level).

Mỗi macro sinh `.gds/.lef/.lib` + behavioral Verilog. Thay 2 module `*_sram_asic.v`
bằng instantiate macro (hoặc wrapper map port). Re-run `tb_top_asic` với behavioral
model → phải vẫn 21/21 bit-exact.

## M2 — OpenLane (Linux/WSL, phác thảo)

`config.json`: `DESIGN_NAME=ecg_core_asic`, `CLOCK_PORT=clk`, `CLOCK_PERIOD=10` (100MHz),
`PDK=sky130A`, `EXTRA_LEFS/LIBS/GDS` + `MACRO_PLACEMENT_CFG` cho 3 macro.
Flow: synth→floorplan→place→cts→route→signoff. Lặp tới DRC/LVS clean + no hold +
setup≥0. Thu die area, gate count, power, Fmax.

## Lưu ý

- KHÔNG đụng RTL FPGA gốc `hardware/RTL/` — bản ASIC nằm riêng trong `hardware/asic/rtl/`.
- `ecg_accelerator_top_asic.v` chỉ dùng cho regression sim; chip top thật là `ecg_core_asic`.
- cp_block / cnn_controller / gap_fc_argmax / cp_engine dùng chung từ `hardware/RTL`
  (logic technology-agnostic, không cần bản ASIC riêng).
