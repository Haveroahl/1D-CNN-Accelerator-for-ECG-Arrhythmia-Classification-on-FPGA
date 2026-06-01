# IP-Core Packaging — ECG CNN Accelerator

Note các đề xuất chính cho việc đóng gói accelerator hiện tại thành IP độc lập, tái sử dụng được qua Platform Designer (Qsys).

---

## 1. Vấn đề & Mục tiêu

**Hiện tại**: `ecg_accelerator_top.v` vừa là top-level vừa là "IP" — avalon_slave làm cả 2 vai trò (control + ghi input data), không có boundary rõ ràng giữa accelerator core và I/O system.

**Mục tiêu**: Đóng gói thành IP với interface chuẩn Intel/Altera (Avalon-MM + Avalon-ST), độc lập với nguồn data (HPS / DMA / ADC streaming) — không phải sửa RTL core khi đổi I/O path.

---

## 2. Hai phương án I/O Path (đã thảo luận)

### Cách 1 — Buffered qua input_sram (giữ thiết kế hiện tại)
```
DMA / ADC → Avalon-MM/ST write → input_sram (2500×8b) → cp_engine
```
- Load latency: 2500 cy (single-write) hoặc ~300-500 cy (DMA burst)
- Không sửa accelerator core, chỉ thay I/O master.
- Phù hợp: batch processing, test mode, hệ thống có buffer.

### Cách 2 — Streaming trực tiếp vào SRW (bypass input_sram)
```
ADC → sample FIFO (depth 5) → SRW của cp_engine (Conv1)
```
- Load latency: 0 (overlap với thu sample ECG real-time).
- Phải sửa cp_engine: Conv1 source mux = FIFO thay vì input_sram.
- Phù hợp: real-time monitoring, sample-rate-driven.

**Quyết định**: chọn **Cách 1** cho lần đóng gói này — core đã verify 21/21 bit-exact, không cần đụng datapath. Cách 2 ghi nhận làm future work.

---

## 3. Interface IP — 2 cổng tách biệt

```
                    ┌─────────────────────────────────────┐
                    │     ecg_cnn_ip (IP boundary)        │
                    │                                     │
  control/status    │  ┌──────────────────────────────┐  │
  ◄──────────────►  │  │ Avalon-MM Slave (CSR)        │  │
  (Avalon-MM)       │  │ start, done, result, mode    │  │
                    │  └──────────────────────────────┘  │
                    │                                     │
  data input        │  ┌──────────────────────────────┐  │
  ──────────────►   │  │ Avalon-ST Sink (data port)   │  │
  (Avalon-ST)       │  │ valid/ready/data[7:0]        │  │
                    │  └──────────────────────────────┘  │
                    │                                     │
  irq_done  ◄────── │  Interrupt sender                  │
                    └─────────────────────────────────────┘
```

### Tại sao Avalon-ST cho data port

| Tiêu chí | Avalon-MM | **Avalon-ST (chọn)** |
|---|---|---|
| HPS single-write | OK | OK (qua ST adapter) |
| DMA burst | OK | OK (DMA có ST master) |
| ADC streaming | Cần bridge | **Native fit** |
| Backpressure | Không | **Có (ready/valid)** |
| Chuẩn Intel | ✓ | ✓ |

→ Avalon-ST = chuẩn streaming Intel, fit triết lý "IP độc lập I/O".

---

## 4. Cấu trúc RTL sau đóng gói

### Hierarchy mới

| Level | Module | Vai trò | Trạng thái |
|---|---|---|---|
| **0 (IP top)** | `ecg_cnn_ip.v` | Wrapper — IP boundary | **Mới** |
| 1 | `avalon_slave.v` | CSR (control/status) | **Sửa scope** |
| 1 | `st_to_sram_adapter.v` | ST sink → input_sram write | **Mới** |
| 1 | `ecg_accelerator_core.v` | Compute core | **Rename từ top cũ** |
| 2 | `input_sram, ping_pong_sram, cp_engine, cnn_controller, gap_fc_argmax` | Datapath | **Không sửa** |

### Sơ đồ kết nối nội bộ

```
ecg_cnn_ip.v  (WRAPPER — đem đi đóng gói Qsys)
│   Interface ra ngoài:
│   • clk, rst_n
│   • csr_*  (Avalon-MM Slave, 4-bit addr)
│   • asi_*  (Avalon-ST Sink, 8-bit data)
│   • irq_done
│
├── avalon_slave.v          (CSR only — bỏ data port)
├── st_to_sram_adapter.v    (MỚI — ST → input_sram write)
└── ecg_accelerator_core.v  (rename từ top cũ, bỏ avalon_slave bên trong)
    ├── input_sram
    ├── ping_pong_sram
    ├── cp_engine
    ├── cnn_controller
    └── gap_fc_argmax
```

---

## 5. Vai trò avalon_slave.v — KHÔNG bỏ, chỉ sửa scope

### Giữ lại (control/status registers)
- `start` register
- `done`, `busy` status
- `result[1:0]` readback
- `mode`, `irq_enable` (nếu có)

### Bỏ ra (chuyển sang Avalon-ST)
- Cổng `isram_we`, `isram_addr`, `isram_din` — giờ do `st_to_sram_adapter` drive.
- Đường ghi input qua CSR (hoặc giữ optional cho test mode).

→ avalon_slave vẫn là module nội bộ của IP, chỉ thu hẹp scope còn CSR.

---

## 6. Wrapper `ecg_cnn_ip.v` — Cách A (mỏng, khuyến nghị)

Wrapper chỉ làm **netlist** (instantiate + wire-up), không chứa logic.

```verilog
module ecg_cnn_ip (
    input  wire clk, rst_n,
    // CSR
    input  wire [3:0]  csr_address,
    input  wire        csr_write, csr_read,
    input  wire [31:0] csr_writedata,
    output wire [31:0] csr_readdata,
    // ST data sink
    input  wire [7:0]  asi_data,
    input  wire        asi_valid,
    output wire        asi_ready,
    // IRQ
    output wire        irq_done
);

    wire        start, done;
    wire [1:0]  result;
    wire [11:0] isram_wr_addr;
    wire [7:0]  isram_wr_data;
    wire        isram_we;

    avalon_slave u_csr (
        .clk(clk), .rst_n(rst_n),
        .avs_address(csr_address),
        .avs_write(csr_write), .avs_read(csr_read),
        .avs_writedata(csr_writedata),
        .avs_readdata(csr_readdata),
        .start_out(start),
        .done_in(done),
        .result_in(result)
        // KHÔNG còn isram_* ports
    );

    st_to_sram_adapter u_st (
        .clk(clk), .rst_n(rst_n),
        .asi_data(asi_data),
        .asi_valid(asi_valid),
        .asi_ready(asi_ready),
        .wr_addr(isram_wr_addr),
        .wr_data(isram_wr_data),
        .we(isram_we),
        .frame_complete(/* optional auto-start */)
    );

    ecg_accelerator_core u_core (
        .clk(clk), .rst_n(rst_n),
        .start(start),
        .done(done),
        .result(result),
        .isram_wr_addr(isram_wr_addr),
        .isram_wr_data(isram_wr_data),
        .isram_we(isram_we)
    );

    assign irq_done = done;

endmodule
```

**Không chọn Cách B** (nhét logic vào wrapper) — vi phạm separation of concerns.

---

## 7. Lợi ích đóng gói chuẩn này

1. **Core compute không đụng** → giữ nguyên 21/21 bit-exact verification.
2. **Boundary rõ ràng** → đem `ecg_cnn_ip.v` + module con đi đóng gói Qsys độc lập.
3. **Reusable**: project khác / lab khác dùng được không cần sửa RTL — chỉ đổi connection trong Qsys.
4. **System-agnostic**: cùng 1 IP chạy HPS write, DMA burst, ADC streaming — chỉ khác source nối vào `asi_*`.
5. **Professional packaging**: đúng chuẩn Avalon (MM cho CSR + ST cho data) — dễ defend trong thesis.

---

## 8. Đóng gói qua Platform Designer (Qsys)

Tạo `ecg_cnn_ip_hw.tcl` khai báo:
- **Clock/reset interface**: clk, rst_n
- **csr** = Avalon-MM Slave (4-bit address)
- **asi** = Avalon-ST Sink (8-bit data, ready/valid)
- **irq** = Interrupt Sender

User kéo IP vào Qsys, nối:
- HPS-FPGA bridge → `csr` (software control)
- DMA controller / ADC IP → `asi` (data path)

Một IP — 3 use-case (HPS / DMA / ADC) — chỉ khác connection.

---

## 9. Effort & Risk

| Việc | File | Effort |
|---|---|---|
| Rename `ecg_accelerator_top.v` → `ecg_accelerator_core.v`, bỏ avalon_slave instance | core | 0.5 ngày |
| Sửa `avalon_slave.v` — bỏ data port, giữ CSR | avalon_slave | 0.5 ngày |
| Viết `st_to_sram_adapter.v` | mới | 0.5 ngày |
| Viết wrapper `ecg_cnn_ip.v` | mới | 0.5 ngày |
| Cập nhật testbench cũ + tạo tb wrapper (ST driver task) | testbench | 1 ngày |
| `_hw.tcl` cho Qsys packaging | scripts | 0.5 ngày |
| **Tổng** | | **~3.5 ngày** |

**Risk**: Thấp — không đụng datapath compute, chỉ thay layer giao tiếp ngoài.

---

## 10. Verification sau đóng gói

1. **Regression cũ**: chạy `tb_top.v` ở level `ecg_accelerator_core.v` → vẫn phải 21/21 PASS.
2. **TB mới `tb_ecg_cnn_ip.v`**:
   - ST driver task: feed 2500 sample qua `asi_valid/data/ready` handshake.
   - CSR task: ghi start, poll done, đọc result qua Avalon-MM CSR.
   - Verify: cùng input → cùng result như tb cũ.
3. **Qsys integration test**: tạo system minimal (HPS + IP) trong Qsys, generate, compile Quartus → check không lỗi packaging.

---

## 11. Out of scope (lần này)

- Streaming trực tiếp vào SRW (Cách 2) — future work.
- Multi-IP instance / multi-channel — không cần cho thesis.
- AXI4 interface — Intel ecosystem dùng Avalon, không cần AXI.
- DMA controller riêng — dùng DMA của Qsys library.
