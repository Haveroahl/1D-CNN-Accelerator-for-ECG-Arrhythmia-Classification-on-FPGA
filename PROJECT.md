# PROJECT.md — CNN Accelerator for ECG Arrhythmia Classification

> ## ⚠️ PHẠM VI LUẬN VĂN (chốt 2026-07-30)
> **Luận văn chỉ trình bày bản RTL ROM (`hardware/RTL/`) chạy trên DE10-Standard.**
> Khi viết bài / cập nhật doc luận văn, KHÔNG đề cập những phần sau (chúng vẫn tồn tại trong
> repo và mô tả kỹ thuật bên dưới vẫn giữ để tham chiếu, nhưng **ngoài phạm vi bài**):
>
> | Ngoài phạm vi | Ghi chú |
> |---|---|
> | Biến thể **SIMD-20** position-parallel + bảng DSE 2 dataflow | worktree `feature/simd-spec` |
> | Cơ chế **weight-load qua Avalon** (`hardware/RTL_weight/`, Phase B01) + runtime topology config | bản nộp dùng ROM `$readmemh` |
> | **Elastic-Pareto** (nhiều topology / 1 bitstream) | phụ thuộc weight-load |
> | Port **DE0-Nano / Cyclone IV E** + power gate-level SDF | ngoài device mục tiêu |
>
> **Số liệu chốt cho bài** (nguồn duy nhất = `PAPER_DATA.md`):
> ALM **2.120** (5%) · Reg **3.158** · DSP 28 (25%) · M10K 20 (4%) · Fmax **108.46 MHz** ·
> latency 52.16 µs · power 805.53 mW → **42.02 µJ/inf** (confidence **Low**, phải kèm caveat).
> Accuracy: **94.27%** (Chapman-Ningbo INT8 khớp-bit, số triển khai) và **94.65%** (Chapman,
> dùng cho quant-ablation + board) — **KHÔNG trộn hai số**.
> KHÔNG dùng số cũ: 2.201 ALM / 3.177 Reg / 104.85 MHz / 623 mW / 10.3 µJ / 137.6 MHz.

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

**nb per layer:** conv1=8, conv2=7, conv3=6, conv4=7, fc=0 *(ningba re-train 2026-07-28; Chapman cũ conv2=6)*

**w_shift per layer:** conv1=6, conv2=7, conv3=6, conv4=7, fc=7 *(ningba; Chapman cũ conv2=6, fc=8)*

**input_shift_bits = 2** (áp dụng cho input ECG → INT8; ningba cần clip input ±16 để giữ shift=2)

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

### Chapman gốc (~10646, baseline lịch sử — dùng cho Phase D board / SoTA table)

| Model | Params | Accuracy | F1-macro | AFIB Recall |
|-------|--------|----------|----------|-------------|
| Float32 baseline | 1244 | ~94.8% | — | — |
| Pruned float32 | 640 | ~92% | — | — |
| QAT-INT8 float eval | 640 | ~94.84% | — | — |
| **QAT-INT8 round-half-up** (bit-exact) | **640** | **94.65%** | **0.9396** | **0.9266** |

### Ningba (Chapman mở rộng, 33143 records — build chính hiện tại, re-train 2026-07-28)

Train `ningbo_dataset_clip16.npz` (input clip ±16 → input_shift=2). Test 4973.
INT8 = bit-exact GAP floor (= số RTL ROM cho ra). Chi tiết: `results/ningba/EVAL_TABLES.md`.

| Model | Params | Accuracy | F1-macro | AFIB Recall |
|-------|--------|----------|----------|-------------|
| Float32 dense | 1244 | 95.35% | 0.9478 | — |
| Pruned+FT INT8 | 640 | 94.51% | 0.9390 | — |
| Float32 (QAT ckpt) | 640 | 95.03% | 0.9446 | 0.9442 |
| **QAT-INT8 bit-exact** | **640** | **94.27%** | **0.9356** | **0.9062** |

macro-AUC: float32 0.9938 / INT8 0.9712. INT8↔float agreement 0.976.

**Cross-dataset Georgia (5459, zero-shot far-transfer, cùng model ningba):**
float32 acc 92.91% / F1 0.9142 · **INT8 acc 93.00% / F1 0.9151** · AUC 0.958.
INT8 bám float trong 0.1pp (quant không phải nguồn drop). Chi tiết: `results/georgia/EVAL_TABLES.md`.

**Golden test đã sinh** (ningba + Georgia): `results/{ningba,georgia}/test_golden/` gồm
`label_golden/` (4 mẫu/nhãn cho sim nhãn), `bitexact/` (1 mẫu 21-checkpoint),
`fullset/expected_argmax.hex` (toàn test so accuracy RTL≈SW). Consistency PASS 100%.

---

## Trạng thái tiến độ (cập nhật 2026-06-18)

### Software — ✅ DONE (baseline)
- [x] Re-prune model → channels (4,4,8,8) — `best_model_pruned.pth`
- [x] QAT-INT8 power-of-2 round-half-up — `qat_int8/model_qat_int8.pth` (94.65% acc bit-exact, F1 0.9396)
- [x] Export `flat_weights.hex` (580 INT8 entries, không comment lines)
- [x] Golden `.mem` files (21 checkpoints / sample × 3 samples) — `results/golden/`

### Hardware — ✅ DONE (baseline verify)
> Chi tiết kiến trúc, datapath, timing: @hardware/System_Design.md
- [x] 8 RTL modules (cp_block, cp_engine, controller, gap_fc_argmax, ping_pong_sram, input_sram, top, avalon_slave)
- [x] Testbench tb_top.v — **21/21 bit-exact PASS** với golden Python (3 samples)
- [x] Latency đo thật: **5216 cycles ≈ 52.16 µs @ 100 MHz** (deterministic)
- [x] Throughput: ~19,200 inference/s @ 100 MHz
- [x] SDC 100/150 MHz chuẩn bị sẵn (chưa synthesis)

### Hardware — ✅ DONE (runtime-reconfigurable topology, 2026-06-17)
> Chi tiết: @hardware/System_Design.md mục "Runtime-Reconfigurable Topology"
- [x] CONFIG window (avalon_slave) nạp per-layer in_ch/cp_en/nb/base lúc runtime, reset=Chapman
- [x] Weight RAM depth 17→32 (cover MAX in_ch=8,8,8,8); GAP `out_ch_mask` cho Conv4 out_ch<8
- [x] `tb_topo.v` — full inference **bit-exact 7/7**: MIN (2,2,2,2) + MAX (8,8,8,8) + 4 mixed
- [x] `tb_top.v` TC08/09/10 — config write/consume/recover + GAP mask + word biên 31 (15/15 PASS)
- [x] Golden generator topology tùy ý: `software/python/gen_topo_golden.py`

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

### Block 2 — Windows (Hardware, ~1 tuần) — ✅ DONE

**Phase B — tách core/bus + wrapper — ✅ DONE**
- [x] Phase B — tách `ecg_core.v` (core thuần: isram+pp+cpe+gfa+ctrl, interface 8 dây sram_wr_addr/din/we + start/busy/done/result) khỏi bus
- [x] Phase B — `ecg_accelerator_top.v` co thành wrapper mỏng: `avalon_slave` (bus adapter) + `ecg_core`. `avalon_slave.v` dùng chung Phase C (chân ảo) và Phase D
- [x] Phase B — regression sim: `tb_top.v` 21/21 bit-exact PASS sau refactor

**Phase B01 — weight ROM → RAM reload — ✅ DONE (commit 14cac23, 6e1eb9f)**
> Enabling mechanism cho C5 (Avalon weight reload) + C3 cross-dataset on-hardware.
- [x] Phase B01 — weight FF/ROM ($readmemh) → weight RAM dual-port M10K, write từ Avalon (CONFIG window)
- [x] Phase B01 — mở rộng avalon_slave address + DATA WINDOW cho burst weight/ECG load
- [x] Phase B01 — regression: 21/21 bit-exact PASS với weight load via Avalon
- [x] cp_block tách 3 submodule (cp_mac/cp_accumulate_rescale/cp_pool) — bit-exact, commit c2de533 (2026-06-18)

**Phase C — Synthesis + Power — ✅ DONE**
- [x] Phase C — ✅ DONE: Quartus Compile thật (5CSXFC6D6F31C6, re-compile 2026-06-17 sau weight RAM) — DSP 28/112 (25%), **ALM 2851 (7%), Reg 4843**, Timing PASS @100MHz (slack +3.43ns, **Fmax ≈104MHz**). Số cũ ALM 2261/Reg 3196 là trước weight RAM — lỗi thời.
- [x] Phase C — ✅ DONE: Fmax số dùng cho paper (theo PAPER_DATA.md): **DSE = 104.85MHz** (so công bằng với SIMD baseline), **weight-RAM Phase B01 = 108.94MHz**, **board jtag_top ~125MHz** (+2.202ns@100MHz). Timing fix fold bias+round_add (commit 32f7a11) cho internal reg-to-reg ~137.6MHz nhưng **KHÔNG dùng làm Fmax công bố** (đó là internal path, không phải số toàn thiết kế).
- [x] Phase C — ✅ DONE: PowerPlay+VCD thật (95.6% toggle) — Total 623mW / Dyn 198mW / Static 413mW → **Energy/inf 10.3µJ dyn / 32.5µJ total**. DSP = 68% dynamic → nối thẳng C1

**Phase D — On-board DE10-Standard — 🟡 JTAG DONE, UART chờ phần cứng**
> Quartus Lite KHÔNG có IP HPS Cyclone V → Phase D chuyển HPS → JTAG-to-Avalon + System Console. soc_top.v/HPS giữ làm tham khảo.
- [x] Phase D — JTAG-to-Avalon (`jtag_top.v`) + System Console driver: chạy thật trên DE10, **94.27% (1004/1065)** khớp Python 94.65%. JTAG chậm/dễ rớt kênh nhưng đã chứng minh "FPGA-deployed".
- [x] Phase D — variant Nios V/m RISC-V soft-core (Quartus 25.1 bỏ Nios II), on-chip RAM bare-metal — sim 3/3 PASS, compile PASS
- [~] Phase D — variant UART (PC serial → ecg_core, không JTAG/Nios/HPS): RTL+pin+host script READY, merged main — **chờ module USB-TTL 3.3V** để chạy board
- [x] Phase D — PTB-XL on-RTL: 1 bitstream chạy cả Chapman + PTB-XL chỉ bằng reconfig nb + weight reload (bit-exact, không cần board)

### Block 3 — Writing (any env, ~1 tuần) — 🔲 CÒN LẠI (đường găng)
- [~] Phase E — SoTA tables (verified) → `SOTA_TABLE.md`: Bảng A (5 model Chapman, cột Params + Beat/Rhythm) + Bảng B (10 FPGA biomedical, cột Freq + Throughput + Beat/Rhythm). Liu verify bit số (INT8 92.95% < ta 94.65%). Pareto chart 🔲 chưa vẽ.
- [~] Phase E01 — Tài liệu tham khảo (≥15) → `paper/REFERENCES.md`: **17 mục**, định dạng ICDV (tác giả/tiêu đề/nguồn in nghiêng/vol-no-pp/năm + DOI/ISSN/ISBN). 9/17 có DOI ✅, 8 mục 🔲 cần bổ sung citation đầy đủ. Hạn chế link Internet (arXiv → ưu tiên bản published).
- [~] Phase F — draft ICDV ~6 trang (chỉ bản production 8-PE, KHÔNG SIMD/DSE) → `paper/ICDV_draft.md`. Venue = ICDV (xem PAPER_DATA.md), KHÔNG phải MDPI.
- [ ] Phase F — GitHub public + Zenodo DOI reproducibility artifact
- [ ] Phase K — Liêm chính học thuật: (a) tài liệu quy trình `paper/INTEGRITY_CHECK.md` — công cụ (Turnitin/iThenticate qua trường, GPTZero/Originality.ai cho AI-detect), ngưỡng % chấp nhận, cách diễn giải; (b) self-audit thủ công ICDV_draft.md trong cùng file — đánh dấu đoạn rủi ro copy nguyên, kiểm mọi claim có cite, soi câu "giọng AI". KHÔNG upload bản thảo ra dịch vụ ngoài (làm offline trong repo).
- [ ] Phase L — Báo cáo tiến độ Word ~10 trang `paper/PROGRESS_REPORT.md` (Markdown → Word, không cầu kỳ): tóm tắt mục tiêu, kết quả software (94.65% acc, ablation), hardware (52µs, synth, board 94.27%), cross-dataset, trạng thái các Phase A–F/K, việc còn lại.

---

## Lưu ý quan trọng

- **flat_weights.hex**: KHÔNG có comment lines — $readmemh đọc từ byte 0
- **Không có ReLU Conv1-3**: preserve ECG features âm
- **QAT dùng cho checkpoint chính** — nhưng power-of-2 INT8 robust: PTQ (calibrate, no fine-tune) cũng đạt 94.08%, QAT chỉ hơn ~0.3% (94.37%). PTQ là baseline A0 trong Table 4 (KHÔNG sập — claim "~22% broken" cũ đã bị bác bỏ bằng số đo bit-exact)
- **Rounding**: round-half-up, KHÔNG phải floor
- **Output channels mới**: 4,4,8,8 (power-of-2) — cần re-train trước khi update hardware
- **Dataset train chính (2026-07-28)**: `data/ningba_processed/ningbo_dataset_clip16.npz` (Chapman mở rộng 33143, clip input ±16). Chapman gốc `data/Chapman` giữ cho baseline lịch sử. Cross-dataset: Georgia `data/georgia_by_class` (5459, zero-shot), PTB-XL `data/ptbxl`.
- **Loader npz**: `utils/npz_dataset.py get_npz_dataloaders` — flag `--npz` cho train/prune/qat/generate_golden (đọc npz thay Chapman-xlsx). Scripts eval/golden mới: `int8_eval_batch.py` (INT8 bit-exact re-classify + 2 bảng + CM/ROC), `gen_test_golden.py` (label_golden 4 nhãn + bitexact + fullset).
- **RTL ROM đã đồng bộ ningba (2026-07-28)**: `hardware/RTL/*_w.hex` = weight ningba; `cnn_controller.v` nb hard-code đổi Conv2 6→7 (nb=8/7/6/7). RTL/ KHÔNG còn chạy Chapman gốc bằng ROM. **Chưa re-run tb_top.v** với golden ningba mới.
- **Simulation tool**: ModelSim/Questa (Windows) — RTL trong `hardware/`, sim trong `hardware/fpga/simulation/questa/`
- **Cross-dataset = PTB-XL** (không phải MIT-BIH — đã chuyển vì PTB-XL 500Hz/10s sẵn, không cần resample 360Hz)
- **`hardware/` = production** (8 modules, 21/21 bit-exact, synth thật). **`hardware_v3/` = skeleton reference** fully-mapped mirror Liu 2023 — KHÔNG thay production, chỉ để so sánh kiến trúc trong paper.
- **`hardware/fpga/soc/`** = template Qsys/HPS (soc_top.v) — KHÔNG dùng (Quartus Lite không có HPS IP). Phase D thực tế dùng JTAG-to-Avalon (`jtag_top.v`) đã chạy board thật 94.27%, + variant Nios V/m + UART.
- **`avalon_slave.v` vẫn cần khi dùng HPS** — Qsys/HPS chỉ tự sinh interconnect + h2f bridge + decode địa chỉ, KHÔNG diễn dịch thanh ghi riêng của core (địa chỉ nào nạp SRAM, bit nào là start, đọc đâu ra done/result). Phần đó là `avalon_slave.v` thủ công, đóng vai **bus adapter** dùng chung Phase C/D. Chỉ bỏ được nếu đổi kiến trúc sang PIO core hoặc On-Chip RAM+DMA (đều là refactor lớn → không làm).
- **Quartus install**: `D:\altera_lite\25.1std` — project `hardware/fpga/ecg_accelerator_top.qsf` (top = `ecg_accelerator_top` cho Phase C; đổi sang `soc_top` khi Phase D).
- **Toàn bộ dự án chạy trong `d:\Thesis101`** — không tách Windows/WSL.
