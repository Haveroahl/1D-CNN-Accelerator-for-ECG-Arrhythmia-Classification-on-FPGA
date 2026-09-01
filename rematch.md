# rematch.md — Kế hoạch: convert model Chapman+Ningbo sang INT8 rồi eval

> Mục tiêu: thống nhất TOÀN BỘ Chương 4 về **một** model = Chapman+Ningbo **INT8**
> (power-of-2 round-half-up, khớp RTL). Hiện 4.1 đang là số float32, 4.2 dùng bản
> Chapman-only INT8 → chưa nhất quán. Việc này chuyển cả in-dist lẫn Georgia sang INT8,
> và (theo quyết định) đổi model phần cứng sang combined INT8.

## Quyết định đã chốt (session này)
- **Phạm vi INT8**: convert CẢ in-distribution LẪN Georgia sang INT8 (toàn Ch4 nhất quán INT8).
- **Model phần cứng cuối**: ĐỔI sang **combined (Chapman+Ningbo) INT8**.
  - ⚠️ Hệ quả: golden RTL cũ + số board 94,27% (1004/1065) KHÔNG còn đúng → phải re-gen
    golden + chạy lại tb 21/21 + **chạy lại board DE10**. Số board là kết quả chạy thật,
    KHÔNG tái tạo được bằng phần mềm.
- **Board (Mục 4.3)**: người dùng sẽ **tự chạy lại board sau** với weights combined; tạm
  thời 4.3 đánh dấu "chờ chạy lại", Claude làm phần phần mềm trước.

## Còn phải quyết trước khi chạy
- [ ] **Bước 2 dùng PTQ hay QAT?**
  - PTQ (`ptq_int8.py`): calibrate, không train lại — nhanh, 1 lần chạy; INT8 có thể tụt vài %.
  - QAT (`qat_int8.py`): fine-tune fake-quant ~30 epoch rồi convert — khớp "phương pháp
    chính" luận văn, INT8≈float, nhưng lâu hơn + cần loader combined.
- [ ] Có re-gen golden .mem cho RTL từ weights combined luôn không, hay để riêng khi chuẩn
  bị chạy board?

## Ba điểm vướng kỹ thuật (đã xác minh từ code)
1. **Hai kiến trúc model có thể khác nhau.** case1 dùng `ECG_CNN` (từ `ptbxl_eval.py`);
   pipeline INT8 (`qat_int8.py`/`ptq_int8.py`) dùng `ECG_1DCNN_Pruned`. Phải đối chiếu 2
   định nghĩa — nếu cùng (4,4,8,8), cùng tên conv1-4/fc thì copy weights thẳng; nếu không,
   viết adapter map.
2. **Chọn PTQ vs QAT** (xem trên).
3. **Calibration + eval phải trên data combined, KHÔNG phải Chapman.** Pipeline hiện
   hard-code loader Chapman (`get_dataloaders`). Phải trỏ sang:
   - in-dist: `data/case_study/case1_merged.npz` (đã có train/val/test split)
   - cross: `data/georgia_by_class` (by-class .npy tree, 5.606 record)
   → viết **script mới** (KHÔNG sửa script cũ), tái dùng `int8_forward` (đường bit-exact
   khớp RTL: round-half-up, nb, input_shift) từ `qat_int8.py`.

## Thứ tự thực hiện (chưa chạy — chờ duyệt + 2 quyết định trên)

| Bước | Việc | Verify |
|---|---|---|
| 0 | Đối chiếu `ECG_CNN` (case1) vs `ECG_1DCNN_Pruned` (pipeline INT8) | In 2 định nghĩa; xác nhận copy weights được hay cần adapter |
| 1 | Viết `cross_eval/case1_int8_eval.py`: load `case1_model_float32.pth` → build QAT shell → copy weights | Script chạy không lỗi |
| 2 | Quantize INT8 power-of-2 round-half-up: calibrate w_shift/nb/input_shift từ **train split case1_merged** (PTQ hoặc QAT theo quyết định) | In w_shift/nb/input_shift; cảnh báo nếu nb≠RTL |
| 3 | Eval INT8 in-dist trên **test split case1_merged** | acc/F1 INT8 in-dist |
| 4 | Eval INT8 Georgia zero-shot (`data/georgia_by_class`) | acc/F1 INT8 Georgia |
| 5 | Eval float32 Georgia lại (so C2==C6, chứng minh quant drop) | Δ(float,INT8) |
| 6 | Ghi kết quả ra JSON + cập nhật số thật vào Chương 4 (Bảng 4.2, 4.5) | Số trong bài = số JSON |

## Số hiện tại trong Chương 4 (sẽ bị thay khi có INT8 combined)
- 4.1.1 in-dist (float32 combined): acc **94,63%** / F1 0,9396 / AUC 0,9914 — nguồn
  `results/case_study/case1_result.json` → `indist_test`.
- 4.1.2 Georgia zero-shot (float32 combined): acc **93,07%** / F1 0,9149 / AUC 0,9812 —
  nguồn `results/cross_eval/georgia_th1_eval.json`.
- Baseline Chapman-only zero-shot Georgia (để so vai trò Ningbo): acc **90,24%** / F1 0,8765
  / AFIB F1 0,73 — nguồn `results/cross_eval/georgia_c2_report.json`.
- Bản Chapman-only INT8 (phần cứng hiện tại): **94,65%** / F1 0,9396 (golden RTL, board 94,27%).

## Ràng buộc BẮT BUỘC giữ (session rule)
- KHÔNG sửa/bịa nhãn ground-truth để cải thiện metric (data fabrication).
- Số hand-calc/kết quả trong bài phải LÀ số thật từ file, không bịa.
- KHÔNG dùng PTB-XL (đã bỏ khỏi bài). Cross-check = **Georgia**. Train = **Chapman+Ningbo**.
- KHÔNG nhắc SIMD-20 / DSE (đã bỏ khỏi bài).
- Accuracy nhất quán; Fmax dùng 104,85 MHz (KHÔNG 137,6).
- Board là kết quả chạy thật — không tái tạo bằng phần mềm.
