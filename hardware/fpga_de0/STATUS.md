# DE0-Nano port — STATUS (2026-06-21, GLS đã giải quyết)

> Port `ecg_accelerator_top` từ DE10 (Cyclone V) sang DE0-Nano (Cyclone IV E).
> Compile/fit/timing/power + RTL sim + **gate-level SDF sim** đều xong và verify.

## ✅ KẾT QUẢ THẬT — đã verify, dùng được cho thesis

Thư mục `hardware/fpga_de0/` = port của `ecg_accelerator_top` (8-PE production core)
từ DE10 (Cyclone V) sang DE0-Nano (Cyclone IV E `EP4CE22F17C6`).

| Hạng mục | Số | Nguồn (đã chạy thật) |
|---|---|---|
| Compile | 0 errors | quartus_map + quartus_fit |
| Logic | 8,035 LE (36%) | `output_files/ecg_de0_top.fit.summary` (rev 50MHz) |
| Block RAM | 95,776 bit / 456 M9K seg (16%) | fit.summary — **RAM vào M9K đúng**, không rớt LE |
| Multipliers | 44 / 132 (33%) | fit.summary |
| Timing @50MHz | slack **+7.14 ns** (demo) | rev `ecg_de0_top`, sta.rpt |
| Timing @100MHz | slack **+0.44 ns PASS**, Fmax **104.6 MHz** | rev `ecg_de0_100`, sta.rpt (85C slow) |
| Power (VCD gate-level SDF) | Total **247mW** / Dyn **135mW** / Static **80mW** / IO 32mW | `ecg_de0_100.pow.summary`, confidence **Medium** |

**Điểm chốt:** static **412mW (DE10) → 80mW (DE0) = −81%**. Đây là bằng chứng low-power
thật (die nhỏ Cyclone IV, không ARM hard-core). Power confidence = **"Medium"** (gate-level
SDF VCD, Unknown 0.0%, Toggle 80.6%, 86.9% signals có toggle rate thật) — mức trần thực tế
của Quartus Lite + Questa FSE. Chi tiết quy trình ở nhánh GLS bên dưới (đã xong).

## Cấu trúc đã tạo (GIỮ)
- `ecg_de0_top.qpf/.qsf` — rev demo 50MHz
- `ecg_de0_100.qpf/.qsf` — rev timing-check 100MHz
- `ecg_de0_common.qsf` — device/RTL/HEX dùng chung (có `SEARCH_PATH ../RTL` cho $readmemh)
- `ecg_de0_50mhz.sdc` / `ecg_de0_100mhz.sdc`
- `rtl_de0/ping_pong_sram.v` + `rtl_de0/cp_engine.v` — fork DUY NHẤT đổi ramstyle
  M10K→M9K (+ b_store bỏ hint MLAB). 10 module còn lại dùng chung `../RTL/`.
- `README_DE0.md`

## ✅ NHÁNH GLS (gate-level power) — ĐÃ GIẢI QUYẾT (2026-06-21)

Mục tiêu: VCD gate-level → PowerPlay confidence cao hơn "Low" của VCD RTL. **ĐẠT.**

### Root cause của `cycles=2` (đã fix):
`avs_readdata` là **registered** (`avalon_slave.v:179-184`): nó capture register được chọn
TẠI posedge mà `avs_read` được sample =1 → data chỉ valid ở posedge KẾ. Task `avs_rd` cũ
sample ngay posedge đầu → ở RTL tình cờ đúng, nhưng ở **gate (SDF delay)** `avs_read` tới
register trễ vài ns sau cạnh → MISS cạnh đó → đọc giá trị cũ → busy=0 giả → poll thoát
ngay → `cycles=2`. **Fix:** `avs_rd` chờ THÊM 1 posedge trước khi sample (đúng semantic
Avalon readdatavalid). Sau fix: RTL 6/6 PASS, gate SDF PASS `result=3 cycles=5216`.

> Latency đo qua bus = đếm clock bằng counter free-running `cyc_cnt` (không dùng
> `(t_end-t_start)/poll-rate` mong manh nữa). Kiểm tra latency nới thành cửa sổ
> [5216, 5220] vì readback thêm vài cycle — số cycle chính xác đã chứng minh bởi
> tb_top trên DE10, đây chỉ quan sát chứ không assert lại.

### Đã loại trừ (3 hướng SAI trước khi tới root cause đúng):
1. ❌ X power-up register → reset 4→20 cycle. (vô hại nhưng không phải nguyên nhân)
2. ❌ `$hold` timing-check → `+notimingchecks`. (không phải nguyên nhân)
3. ❌ Zero-delay race → SDF back-annotation. (đúng flow nhưng không phải nguyên nhân)
→ Cả 3 đều không sửa được vì bug nằm ở **testbench readback timing**, không ở netlist.

### KẾT QUẢ POWER GATE-LEVEL (dùng cho thesis):
| | Số |
|---|---|
| Total Thermal Power | **247.3 mW** |
| Core Dynamic | 135.4 mW |
| Core Static | 79.7 mW |
| I/O | 32.2 mW |
| Confidence | **Medium** (gate VCD, 86.9% signals toggle thật, **Unknown 0.0%**, Toggle 80.6%) |

- Số gate-level (247mW) ≈ số RTL-VCD cũ (243mW) → xác nhận số gốc đúng, gate chỉ nâng
  **confidence Low→Medium** + bằng chứng đúng đắn netlist với delay thật slow-corner 85C.
- **Static 79.7mW (DE0) vs 412mW (DE10) = −81%** — bằng chứng low-power thật.
- "High" KHÔNG đạt được với Quartus Lite + Questa FSE (cần 100% toggle gồm glitch nội bộ);
  Medium + 0% unknown là mức trần thực tế của free-tier — defendable.

### Cấu hình PowerPlay (đã set trong `ecg_de0_100.qsf`):
- VCD = `simulation/questa/tb_gate_de0.vcd` (gate-level, SDF).
- **Cửa sổ POWER_VCD_FILE_START_TIME=25226ns, END=77426ns** = chỉ phần COMPUTE
  sample-0 (52.2µs = 5216cy). Bỏ giai đoạn load Avalon (node còn X) → Unknown 0%,
  confidence Medium. ← QUAN TRỌNG nếu re-run; mốc lấy từ `[VCD] dump ON/OFF @ <time>`
  TB in ra (chạy lại sim sẽ in mốc, set START/END theo đó).
- **TB nạp ECG bằng DATA WINDOW** (`addr=0x1000|i`, 1 bus write/sample) — không còn
  đường legacy 3-lệnh/byte. Load 75µs→25µs, compute bắt đầu sớm hơn → cửa sổ dời.
  Power KHÔNG đổi (247.33mW cả 2 cách) → số chỉ phụ thuộc compute activity, đáng tin.

### Quy trình GLS (đã chạy thành công, để lặp lại):
1. `quartus_eda --simulation --tool=questa_oem --format=verilog ecg_de0_100` → `.vo`+`.sdo`.
2. RTL smoke test: `vsim -c -do run_tb_gate_de0_rtl.do` → phải 6/6 PASS trước.
3. Gate SDF: `vsim -c -do run_tb_gate_de0_sdf.do` → sinh `tb_gate_de0.vcd` (windowed sample-0).
4. `quartus_pow ecg_de0_100` → đọc `output_files/ecg_de0_100.pow.summary`.
- **1 vsim tại 1 thời điểm** — Questa FSE nodelocked chỉ 1 session (nhiều session → "License checkout disallowed").
- Weight hex PHẢI ở CWD sim — đã có: w_ram0..7, conv_bias, fc_weights, fc_bias, ecg_sample0..2, expected_results.

## Testbench liên quan
- `testbench/tb_gate_de0.v` — black-box GLS tb (fork tb_gate.v, addr 5→14 bit,
  +VCD windowing, +DBG_POLL guard, `avs_rd` +1-cycle readback fix, free-running
  cycle counter). RTL 6/6 PASS; **gate SDF PASS** (result=3, cycles=5216).
- `testbench/tb_gate.v` — bản gốc Cyclone V (addr 5-bit, lỗi thời cho top 14-bit).
