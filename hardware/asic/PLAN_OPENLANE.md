# Kế hoạch — ASIC flow cho ecg_core qua OpenLane (Sky130)

## Mục tiêu
Mang `ecg_core` (đã verify bit-exact trên FPGA) qua OpenLane/Sky130 ra GDSII,
lấy số liệu area / power / timing cho paper (thêm cột "ASIC" cạnh cột FPGA C4/C5).

## Quyết định đã chốt
- Môi trường: chạy trên **máy Linux/WSL riêng** → repo chỉ cần RTL + config sẵn sàng để mang sang.
- Scope: **ecg_core + interface đơn giản** (bỏ avalon_slave/JTAG/PLL — không synth ASIC được).
- Memory: **dùng RAM macro cho TOÀN BỘ memory, gộp thành ít macro** (không FF, không macro rời 17 cái).

## Cấu hình macro RAM (gộp)
RTL hiện tại có 17 mảng rời: `input_sram` (2500×8) + `ping_pong_sram` (16×512×8).
Gộp lại còn **3 macro**:
- **input_sram → 1 macro** 4096×8 (làm tròn lên lũy thừa 2; OpenRAM thường cần kích
  thước "đẹp"). Dùng 2500 word đầu.
- **ping_pong → 2 macro 512×64** (mỗi macro = 8 channel × 8 bit của 1 bank A/B).
  → ĐÒI HỎI SỬA RTL `ping_pong_sram`: 16 mảng 8-bit → 2 mảng 64-bit, write cả 8-ch
  một word (we gộp), read cả word. Đây CHÍNH LÀ pack mà nhánh `pingpong-pack-512x16`
  đã thử trên FPGA — fail timing FPGA do ràng buộc I/O Avalon standalone, NHƯNG trên
  ASIC không có ràng buộc đó → khả thi. Phải verify lại bit-exact sau pack.

→ verify pack: tb_top.v vẫn 21/21 bit-exact sau khi đổi ping_pong sang 512×64.

---

## Milestone 0 — Chuẩn bị RTL cho ASIC ✅ DONE (verify Questa 2026-06-15: 10/10 + 21/21 bit-exact, max|diff|=0, 5216 cy)

Tạo `hardware/asic/rtl/` chứa RTL đã điều chỉnh (KHÔNG sửa RTL FPGA gốc trong `hardware/RTL/`):

1. **`ecg_core_asic.v`** — wrapper top cho ASIC:
   - Instantiate `ecg_core` + interface I/O đơn giản (parallel load: `wr_addr/din/we/start`
     + `busy/done/result/isram_free`) thay avalon_slave. Đây chính là 8-wire interface
     core đã có sẵn → chỉ cần expose ra chân top, KHÔNG cần bus adapter.
   - Bỏ PLL: clock vào thẳng từ chân `clk`.
2. **`ping_pong_sram_asic.v`** — pack 16 mảng 512×8 → **2 mảng 512×64** (1 word = 8 ch).
   Port đổi din/dout sang 64-bit, we gộp (8-ch ghi đồng thời — đúng vì cp_en power-of-2,
   cặp channel luôn ghi cùng lúc). KHÔNG đọc mem trong write (tránh read-during-write
   làm logic nổ — bài học từ nhánh pingpong-pack). Đây là bước chuẩn bị để map 2 macro.
3. **`input_sram_asic.v`** — giữ 2500×8, chuẩn bị map 1 macro 4096×8 (dùng 2500 word đầu).
4. **Weight `$readmemh`** trong `cp_engine.v`: kiểm tra OpenLane/yosys đọc được `$readmemh`
   trong `initial`. Nếu không → chuyển weight thành ROM logic (case statement) sinh từ
   `flat_weights.hex` bằng script Python. → verify đây là blocker hay không TRƯỚC.

→ verify: `iverilog` compile sạch `ecg_core_asic.v` + chạy lại `tb_top.v` (sửa instantiate
   sang core ASIC, ping_pong 512×64) phải **21/21 bit-exact PASS** — pack không đổi logic.

## Milestone 1 — Generate OpenRAM macro (trên Linux/WSL)

5. Generate 3 macro SRAM bằng OpenRAM cho sky130:
   - `sram_4096x8`  — input_sram (1 macro).
   - `sram_512x64`  — ping_pong bank A (1 macro).
   - `sram_512x64`  — ping_pong bank B (cùng config, instantiate 2 lần).
   - Mỗi macro: single-port hay simple-dual-port? RTL cần **1 write + 1 read đồng thời**
     (sync write port + sync read port khác địa chỉ) → cần **dual-port (1RW1R hoặc 2RW)**.
     Xác nhận OpenRAM sky130 hỗ trợ cấu hình này TRƯỚC khi generate.
6. Mỗi macro sinh ra: `.gds`, `.lef`, `.lib`, Verilog behavioral model.
   Thay 2 module `*_sram_asic.v` bằng instantiate macro (hoặc wrapper map port).

→ verify: re-run iverilog tb_top.v dùng behavioral model của macro → vẫn 21/21 bit-exact.

## Milestone 2 — OpenLane flow tích hợp macro → GDSII (trên Linux/WSL)

7. Tạo `hardware/asic/config.json` (OpenLane 2):
   - `DESIGN_NAME = ecg_core_asic`, `VERILOG_FILES` = RTL ASIC (logic, không gồm macro).
   - `CLOCK_PORT = clk`, `CLOCK_PERIOD` bắt đầu **10 ns (100 MHz)** khớp FPGA target.
   - `PDK = sky130A`.
   - Tích hợp macro: `EXTRA_LEFS` / `EXTRA_LIBS` / `EXTRA_GDS_FILES` trỏ tới 3 macro.
   - `MACRO_PLACEMENT_CFG` đặt vị trí 3 macro trong floorplan (input + 2 ping-pong).
8. Chạy flow: `synth → floorplan (đặt macro) → place → cts → route → signoff (DRC/LVS/STA)`.
9. Lặp đến khi: DRC clean, LVS clean, STA không hold violation, setup slack ≥ 0.
   - Nếu setup fail @100MHz → nới CLOCK_PERIOD đo Fmax thật (giống Phase C FPGA).
10. Thu thập số liệu: die area (µm²) + diện tích macro, gate count logic, total power
    (internal+switching+leakage), Fmax. → bảng "ASIC Sky130" cho paper, so cạnh FPGA.

→ verify: `runs/.../reports/` — DRC/LVS clean, `metrics.json` có area/power/timing.

---

## Việc làm ngay (trên Windows, không chờ Linux)
- M0: tạo `hardware/asic/rtl/` + pack ping_pong 512×64 + input_sram + giải quyết
  blocker `$readmemh`.
- M0 verify: iverilog + tb regression 21/21 bit-exact (pack không hỏng logic).
- Viết `config.json` skeleton + `hardware/asic/README.md` (lệnh OpenRAM + OpenLane trên Linux).
→ Khi sang máy Linux: generate 3 macro → chạy `openlane config.json`.

## Out of scope (không làm)
- Không đụng RTL FPGA gốc `hardware/RTL/` (giữ nguyên cho nhánh FPGA).
- Không làm tapeout thật / sky130 MPW submission — chỉ tới GDSII + signoff reports.
- Không port avalon/JTAG/UART sang ASIC.
