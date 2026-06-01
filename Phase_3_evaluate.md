# Phase 3 — Cross-Dataset Evaluation & Hardware-Aware Reconfiguration

## Yêu cầu
- **Cross-data check** cho model: Chapman model có generalize được sang dataset khác?
- **Hardware-aware multi-dataset**: HW IP core chạy được nhiều dataset (target MIT-BIH).

## Trạng thái hiện tại — Chapman hard-coded
- Input length 2500, 4 conv layers (1→4→4→8→8), pool /5, FC 8→4.
- Weight + bias embed bitstream qua `$readmemh` → đổi dataset = re-compile FPGA.

## Phase A — Cross-dataset model eval (Python only, 1-2 ngày)
1. Download MIT-BIH Arrhythmia (PhysioNet, 47 records, 360Hz, 2-lead).
2. Preprocess: resample 360→500Hz, lead II, segment 5s window (2500 samples), normalize giống Chapman.
3. Class mapping AAMI ↔ Chapman 4-class (N→SR, S→GSVT, V→GSVT/SR clinical review, F/Q→drop).
4. Eval modes:
   - **Zero-shot**: Chapman QAT-INT8 → predict MIT-BIH → metrics.
   - **Linear probe**: freeze conv, retrain FC.
   - **Full fine-tune**: unfreeze all.
   - **From-scratch baseline**.
5. Metrics: accuracy, F1-macro, per-class F1, confusion matrix.

**Decision gate**: zero-shot acc > 80% → demo Phase B. Acc thấp → cần fine-tune → Phase B critical.

## Phase B — HW weight reconfig runtime (3-5 ngày)
1. **Weight ROM → RAM**: thay `reg` array + `$readmemh` bằng `weight_ram.v` (dual-port M10K), write từ Avalon, read từ cp_engine.
2. **Extend avalon_slave**: 5-bit → 12-bit address (cover ~4K word weight RAM).
   - Address map: `0x000-0x07F` weight + bias + FC; `0x080` control/status.
3. **HPS C driver**:
   - `load_weights(const char* path)` → đọc hex → write từng word vào FPGA RAM.
   - `run_inference(uint8_t* ecg)` → load input + start + poll done + return result.
4. **Verification**:
   - Sim: load weight via tb write task (thay $readmemh) → 21/21 bit-exact phải PASS.
   - HW: load Chapman weight → 94% acc. Load MIT-BIH weight → expect MIT-BIH acc.
5. **Resource impact**: +1-2 M10K cho weight RAM, +100-200 ALM cho loader logic. Tổng <6% device.

## Phase C — Programmable controller (1-2 tuần, optional)
**Chỉ cần nếu dataset target có input length khác (PTB-XL 10s). MIT-BIH không cần (resample về 2500 OK).**
1. Configurable layer params: `layer_cfg[0..3]={in_len,out_len,in_ch,out_ch,nb,relu_en}` qua register file.
2. Memory upsizing: input_sram 2500→8192, ping_pong 500→1024 (cover worst case PTB-XL).
3. Programmable FC: 8→4 thành 8→N (up to 16 class). Argmax parameterized.
4. Result port 2-bit → 4-bit.
5. Verification: bit-exact 21/21 Chapman phải PASS + thêm test cases MIT-BIH topology.

## Khuyến nghị thực thi
- **Start Phase A ngay** (Python, no HW change) → ra số liệu cho thesis.
- **Phase B** nếu Phase A có ý nghĩa (acc ≥ 70% hoặc fine-tune effective).
- **Phase C** chỉ nếu cần dataset > 5s window (PTB-XL). Với MIT-BIH skip được.

## Dataset comparison
| Dataset | Rate | Window | Class | Phase cần |
|---|---|---|---|---|
| Chapman (current) | 500Hz | 5s/2500 | 4 | A+B đủ |
| MIT-BIH | 360Hz | 5s resample → 2500 | 5 AAMI | A+B đủ |
| PTB-XL | 500Hz | 10s/5000 | 71 sub / 5 super | A+B+C |
| CPSC2018 | 500Hz | 6-60s | 9 | A+B (segment) hoặc C |

## Effort & Risk
| Phase | Effort | Risk |
|---|---|---|
| A | 1-2 ngày | Low (Python only) |
| B | 3-5 ngày | Medium (sim regression manageable) |
| C | 1-2 tuần | High (refactor lớn, dễ break bit-exact) |

## Out of scope
- Không thay đổi quantization scheme (giữ QAT-INT8 power-of-2 shift).
- Không thay đổi topology depth (giữ 4 conv layer).
- Không support multi-lead (luôn lead II).
- Không support real-time streaming (vẫn batch 1 sample/lần).

## Next step cụ thể
1. Confirm dataset priority (MIT-BIH confirmed) + scope (A+B+C confirmed).
2. Start Phase A — viết `software/python/cross_eval/mitbih_eval.py`.
3. Sau Phase A có kết quả → plan chi tiết Phase B trước khi đụng RTL.
