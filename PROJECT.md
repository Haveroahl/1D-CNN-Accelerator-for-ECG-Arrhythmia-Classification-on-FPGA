# PROJECT.md — CNN Accelerator for ECG Arrhythmia Classification

## Mục tiêu dự án

Thiết kế và triển khai một **CNN Accelerator** trên FPGA để phân loại rối loạn nhịp tim (arrhythmia) từ tín hiệu ECG, gồm hai phần:

- **Software**: Huấn luyện, tối ưu và lượng tử hoá mô hình CNN bằng PyTorch
- **Hardware**: Thiết kế lõi IP CNN Accelerator bằng Verilog, target Intel Cyclone V FPGA

---

## Cấu trúc thư mục

```
Thesis/
├── software/python/          # PyTorch ML pipeline
│   ├── model/model.py        # ECG_1DCNN, ECG_1DCNN_Pruned, ECG_1DCNN_Q88
│   ├── utils/dataset.py      # Chapman ECG dataset loader (trả về ecg, label, hr)
│   ├── utils/evaluate.py     # Metrics
│   ├── train.py              # Training float32 model
│   ├── prune_finetune.py     # Structured channel pruning + finetune
│   ├── generate_golden.py    # Sinh golden reference .mem files cho RTL verification
│   ├── export_weights_int8.py
│   ├── quantization/
│   │   ├── quantize_int8.py  # INT8 PTQ
│   │   ├── qat_int8.py       # QAT: fake-quant train → convert INT8 → eval → export
│   │   └── evaluate_quantized.py
│   └── results/
│       ├── best_model.pth
│       ├── best_model_pruned.pth
│       ├── qat_int8/         # model_qat_int8.pth (current best)
│       ├── golden/           # stage .mem files + golden_meta.json
│       └── weights_qat_int8/ # flat_weights.hex (no comment lines)
├── hardware/                 # ⭐ CNN Accelerator RTL (đang implement)
│   ├── RTL/                  # Verilog modules
│   ├── testbench/            # Testbenches
│   └── System_Design.md      # Design document
└── data/Chapman/             # Dataset ECG
```

---

## Mô hình ECG_1DCNN — Pruned (Hardware target)

```
Input (2500 INT8)
  → Conv1(Cin=1,  Cout=4,  K=5, pad=2)  → MaxPool(K=5,S=5) → 500×4
  → Conv2(Cin=4,  Cout=4,  K=5, pad=2)  → MaxPool(K=5,S=5) → 100×4
  → Conv3(Cin=4,  Cout=8,  K=5, pad=2)  → MaxPool(K=5,S=5) →  20×8
  → Conv4(Cin=8,  Cout=8,  K=5, pad=2, ReLU) → MaxPool(K=5,S=5) → 4×8
  → GAP → FC(8→4) → Argmax → Class (0-3)
```

- **ReLU chỉ sau Conv4** — Conv1-3 không có ReLU
- 4 class: AFIB (0), GSVT (1), SB (2), SR (3)
- Output channels power-of-2 (4,4,8,8) để phù hợp phần cứng

---

## Quantization — QAT-INT8 power-of-2 (phương pháp chính cho hardware)

**Power-of-2 scale:** `shift_bits = floor(log2(127 / abs_max))`

**INT8 hardware pipeline:**
```
x_float → round(x * 2^input_shift) → clamp[-127,127] → x_int8
w_float → round(w * 2^w_shift)     → clamp[-127,127] → w_int8
acc_int32 = conv(x_int8, w_int8) + bias_scaled
out_int8  = clamp(round_half_up(acc_int32 / 2^nb), -127, 127)
```

**nb per layer:** conv1=8, conv2=6, conv3=6, conv4=7, fc=0

**w_shift per layer:** conv1=6, conv2=6, conv3=6, conv4=7, fc=8

**input_shift_bits = 2** (áp dụng cho input ECG → INT8)

**Bias scaling:** `bias_scaled = round(b_float * 2^nb)` — lưu INT32 little-endian

**Rounding:** round-half-up (`acc + 2^(nb-1)) >> nb`), KHÔNG dùng floor truncation

### ROM Layout (flat_weights.hex — KHÔNG có comment lines)

Weight layout per layer: [INT8 weights, PE-major → in_ch → tap] [INT32 bias little-endian]

FC weight layout: [out_ch-major, 8 weights/row] (4 rows × 8 cols) — cập nhật khi re-export

---

## Software Pipeline

### Virtual env (Windows, chạy trong d:\Thesis101)
```powershell
cd d:\Thesis101\software\python
.\.venv\Scripts\Activate.ps1   # venv tại d:\Thesis101\.venv
```

### Training & Export
```powershell
# --data_dir default = ../../data/Chapman (relative), không cần truyền tay
python train.py
python prune_finetune.py --checkpoint .\results\best_model.pth
python quantization\qat_int8.py --checkpoint .\results\best_model_pruned.pth `
    --output_dir .\results\qat_int8
python export_weights_int8.py `
    --checkpoint .\results\qat_int8\model_qat_int8.pth `
    --output_dir .\results\weights_qat_int8
```

### Dataset
- `get_dataloaders()` → `(train_loader, val_loader, test_loader)`
- Mỗi batch item: `(ecg, label, hr)` — unpack: `x, y = batch[0], batch[1]`

---

## Kết quả Software

| Model | Params | Accuracy | F1-macro | AFIB Recall |
|-------|--------|----------|----------|-------------|
| Float32 baseline | 1244 | ~94.8% | — | — |
| Pruned float32 | 654 | ~92% | — | — |
| QAT-INT8 float eval | 654 | ~94.84% | — | — |
| **QAT-INT8 round-half-up** | **654** | **94.65%** | **0.9404** | **0.9404** |

---

## Trạng thái tiến độ (cập nhật 2026-06-04)

### Software — ✅ DONE (baseline)
- [x] Re-prune model → channels (4,4,8,8) — `best_model_pruned.pth`
- [x] QAT-INT8 power-of-2 round-half-up — `qat_int8/model_qat_int8.pth` (94.65% acc, F1 0.9404)
- [x] Export `flat_weights.hex` (580 INT8 entries, không comment lines)
- [x] Golden `.mem` files (21 checkpoints / sample × 3 samples) — `results/golden/`

### Hardware — ✅ DONE (baseline verify)
> Chi tiết kiến trúc, datapath, timing: @hardware/System_Design.md
- [x] 8 RTL modules (cp_block, cp_engine, controller, gap_fc_argmax, ping_pong_sram, input_sram, top, avalon_slave)
- [x] Testbench tb_top.v — **21/21 bit-exact PASS** với golden Python (3 samples)
- [x] Latency đo thật: **5216 cycles ≈ 52.16 µs @ 100 MHz** (deterministic)
- [x] Throughput: ~19,200 inference/s @ 100 MHz
- [x] SDC 100/150 MHz chuẩn bị sẵn (chưa synthesis)

---

## Phase tiếp theo — Roadmap Q3 paper (3 tuần)

> Chi tiết novelty, methodology, paper structure: @Paper_Proposal_Q3.md
> Cross-dataset evaluation plan: @Phase_3_evaluate.md

**Story paper Q3** (Hướng 3 a+c, target Electronics MDPI hoặc Sensors MDPI):
- **C1 — Power-of-2 QAT methodology** với ablation định lượng vs general-scale INT8.
- **C2 — Bit-exact verification framework** (21 checkpoints Python↔RTL).
- **C3 — Cross-dataset transfer study** Chapman ↔ MIT-BIH trên FPGA INT8.
- **C4 — IP core architecture** (52 µs/inference, 5K cycles).
- **C5 — Lightweight Avalon weight reload** (enabling mechanism cho C3).
- **Wearable angle**: power-of-2 → 0 DSP rescale → low energy → fit wearable monitoring.

### Block 1 — Software (~1 tuần) — ✅ DONE
- [x] Phase A' — ✅ DONE: ablation A0/A0'/A2/A3/A4 + 5-fold + Chapman CM/ROC → `results/ablation_quant/TABLE4_FINAL.md`. Kết luận: power-of-2 ≈ general (Δ<std) nhưng −4 DSP18 → Pareto-ưu.
- [x] Phase A — ✅ DONE: dùng **PTB-XL** (không MIT-BIH) — 19,952 records, 500→250Hz, lead II → `cross_eval/ptbxl_cross_eval.json`
- [x] Phase A — ✅ DONE: 6 modes C1–C6 + U0. C1 94.46% / zero-shot 77.1% / linear-probe 92.6% / full-FT 93.3%. **C2==C6 → quant drop = 0%** (toàn bộ drop = distribution shift)

### Block 2 — Windows (Hardware, ~1 tuần)

**Phase B — tách core/bus + wrapper (LÀM NGAY, để setup HPS + đẩy xuống board)**
- [ ] Phase B — tách `ecg_core.v` (core thuần: isram+pp+cpe+gfa+ctrl, interface = 8 dây sram_wr_addr/din/we + start/busy/done/result) khỏi bus
- [ ] Phase B — `ecg_accelerator_top.v` co thành wrapper mỏng: instantiate `avalon_slave` (bus adapter) + `ecg_core`, nối 8 dây. `avalon_slave.v` GIỮ NGUYÊN — dùng chung Phase C (chân ảo) và Phase D (Qsys/HPS bridge)
- [ ] Phase B — refactor thuần cấu trúc (copy/paste, KHÔNG đổi logic). Reset giữ nguyên: core dùng `rst` sync, bus dùng `rst_n` async
- [ ] Phase B — regression sim: `tb_top.v` cũ phải 21/21 bit-exact PASS y như trước (thước đo refactor không hỏng gì); thêm `tb_core.v` drive thẳng start/sram_we (không qua bus)

**Phase B01 — weight ROM → RAM reload (NOTE, XỬ LÝ SAU)**
> Tách ra khỏi Phase B. Là enabling mechanism cho C5 (Avalon weight reload) + C3 cross-dataset on-hardware. Làm sau khi core đã tách + HPS chạy được trên board với weight $readmemh.
- [ ] Phase B01 — refactor weight FF/ROM ($readmemh) → `weight_ram.v` dual-port M10K, write từ Avalon
- [ ] Phase B01 — mở rộng avalon_slave 5-bit → 12-bit address (cover ~580 INT8 weight word)
- [ ] Phase B01 — regression: 21/21 bit-exact PASS với weight load via Avalon (không $readmemh)

**Phase C — Synthesis + Power (✅ DONE)**
- [x] Phase C — ✅ DONE: Quartus Compile thật (5CSXFC6D6F31C6) — DSP 28/112 (25%), ALM 2261 (5%), Reg 3196, **Timing PASS @100MHz** (slack +0.508ns, Fmax ≈105MHz)
- [x] Phase C — ✅ DONE: PowerPlay+VCD thật (95.6% toggle) — Total 623mW / Dyn 198mW / Static 413mW → **Energy/inf 10.3µJ dyn / 32.5µJ total**. DSP = 68% dynamic → nối thẳng C1

**Phase D — On-board DE10-Standard (HPS)**
- [ ] Phase D — dựng Qsys `soc_system` từ `ecg_core_hw.tcl` (component bọc `ecg_accelerator_top`) + HPS hard IP, đổi top synth → `soc_top.v`, thêm PLL 50→100MHz
- [ ] Phase D — HPS driver C (`ecg_classify.c`): mmap h2f_lw bridge, nạp ECG → start → poll done → đọc result
- [ ] Phase D — verify on-board accuracy ~94.65% (weight $readmemh, Chapman). Load PTB-XL weight → match Phase A là việc của Phase B01

### Block 3 — Writing (any env, ~1 tuần)
- [ ] Phase E — bảng SoTA 6-8 papers ECG-FPGA + Pareto chart
- [ ] Phase F — draft 8-12 trang Electronics MDPI template
- [ ] Phase F — GitHub public + Zenodo DOI reproducibility artifact

---

## Lưu ý quan trọng

- **flat_weights.hex**: KHÔNG có comment lines — $readmemh đọc từ byte 0
- **Không có ReLU Conv1-3**: preserve ECG features âm
- **QAT dùng cho checkpoint chính** — nhưng power-of-2 INT8 robust: PTQ (calibrate, no fine-tune) cũng đạt 94.08%, QAT chỉ hơn ~0.3% (94.37%). PTQ là baseline A0 trong Table 4 (KHÔNG sập — claim "~22% broken" cũ đã bị bác bỏ bằng số đo bit-exact)
- **Rounding**: round-half-up, KHÔNG phải floor
- **Output channels mới**: 4,4,8,8 (power-of-2) — cần re-train trước khi update hardware
- **Dataset**: `d:\Thesis101\data\Chapman` (cross-dataset: `d:\Thesis101\data\ptbxl`)
- **Simulation tool**: ModelSim/Questa (Windows) — RTL trong `hardware/`, sim trong `hardware/fpga/simulation/questa/`
- **Cross-dataset = PTB-XL** (không phải MIT-BIH — đã chuyển vì PTB-XL 500Hz/10s sẵn, không cần resample 360Hz)
- **`hardware/` = production** (8 modules, 21/21 bit-exact, synth thật). **`hardware_v3/` = skeleton reference** fully-mapped mirror Liu 2023 — KHÔNG thay production, chỉ để so sánh kiến trúc trong paper.
- **`hardware/fpga/soc/`** = template Qsys/HPS (soc_top.v) cho Phase D — chưa dựng/chạy board.
- **`avalon_slave.v` vẫn cần khi dùng HPS** — Qsys/HPS chỉ tự sinh interconnect + h2f bridge + decode địa chỉ, KHÔNG diễn dịch thanh ghi riêng của core (địa chỉ nào nạp SRAM, bit nào là start, đọc đâu ra done/result). Phần đó là `avalon_slave.v` thủ công, đóng vai **bus adapter** dùng chung Phase C/D. Chỉ bỏ được nếu đổi kiến trúc sang PIO core hoặc On-Chip RAM+DMA (đều là refactor lớn → không làm).
- **Quartus install**: `D:\altera_lite\25.1std` — project `hardware/fpga/ecg_accelerator_top.qsf` (top = `ecg_accelerator_top` cho Phase C; đổi sang `soc_top` khi Phase D).
- **Toàn bộ dự án chạy trong `d:\Thesis101`** — không tách Windows/WSL.
