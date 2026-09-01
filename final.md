# FINAL.md — Kế hoạch chạy lại toàn bộ flow để **thống nhất số liệu**

> **Trạng thái**: ĐỀ XUẤT — chờ giảng viên duyệt. **Chưa thực thi.**
> **Ngày lập**: 2026-06-28
> **Người lập**: Lê Đức
> **Mục tiêu cốt lõi**: Mọi con số trong paper/báo cáo phải đến từ **một model duy nhất, một lần chạy pipeline duy nhất, một bộ golden duy nhất** — chấm dứt tình trạng các bước chạy ở thời điểm khác nhau → số lệch nhau.

---

## 0. Vấn đề hiện tại (lý do phải làm)

Các con số đang dùng được đo **không đồng thời**, trên các phiên bản model/data khác nhau:

| Triệu chứng | Bằng chứng trong repo |
|---|---|
| Accuracy SW (94.65%) đo trên **Chapman thuần** (10,646), nhưng cross-dataset lại dùng Ningbo/Georgia/PTB-XL train/eval rời rạc | `utils/dataset.py` chỉ Chapman; `case1_*`, `ningbo_*`, `georgia_*` mỗi cái một model |
| `case1_*` (Chapman+Ningbo gộp) **chỉ train float32**, KHÔNG qua prune→QAT→export→golden | `cross_eval/case1_train.py` dừng ở `case1_model_float32.pth` |
| Golden RTL chỉ có **10 Chapman + 10 PTB-XL** sample (90%/70% acc trên subset nhỏ) → không phản ánh accuracy thật | `golden/batch_summary.json` (20 sample) |
| TB đo accuracy chỉ lặp **3 sample** | `tb_top.v:743` (`sample_idx < 3`) |
| Hardware có 2 đường nạp trọng số (ROM/RAM) nhưng **chưa được đóng gói + đối chiếu thành 2 phiên bản chính thức** | `cp_engine.v:230` `ifndef NO_WEIGHT_INIT`; `tb_weight_load.v` |

**Hệ quả**: không thể nói "model X đạt acc Y% và FPGA chạy đúng acc đó" vì X, Y đến từ các lần chạy khác nhau.

---

## 1. Quyết định đã chốt (với giảng viên / với người dùng)

| Hạng mục | Quyết định |
|---|---|
| **Dataset training chính** | **Chapman mở rộng = TOÀN BỘ `data/ningba`** (~43k record = Chapman-half JS≤10646 + Ningbo-half JS>10646, gộp trong cùng cây WFDB). Đọc **trực tiếp từ WFDB** bằng `case1_build.py` → **re-build npz mỗi lần chạy** (không dùng npz cũ), output `data/case_study/case1_merged.npz`. |
| **Cross-check (sau pipeline SW)** | **Chỉ Georgia** (12-lead, Emory) — vì Ningbo giờ NẰM TRONG train → không còn là cross-dataset hợp lệ. PTB-XL/Ningbo có thể nhắc trong limitation, không phải cross chính. |
| **Golden cho TB đo accuracy** | **Toàn bộ test split**: toàn bộ test set Chapman-mở-rộng + toàn bộ test set Georgia. TB lặp qua hết → ra accuracy thật, so trực tiếp với SW. |
| **Hardware** | **2 phiên bản chính thức**: (V-ROM) trọng số nạp cứng `$readmemh`; (V-RAM) trọng số nạp runtime qua Avalon. Cả 2 phải cho **cùng** kết quả bit-exact. |
| **Output của task này** | File kế hoạch `final.md` (file này). **Chưa thực thi** — chờ giảng viên duyệt. |

---

## 2. Nguyên tắc "một nguồn số liệu" (single source of truth)

Toàn bộ pipeline phải xuất phát từ **đúng một checkpoint INT8** và một bộ golden:

```
data/ningba/WFDBRecords  (~43k: Chapman-half JS≤10646 + Ningbo-half JS>10646)
   │
   ├─[case1_build.py: đọc WFDB → Lead II, 500→250Hz, z-score → split 70/15/15 seed=42]
   ▼
case1_merged.npz  (RE-BUILD mỗi lần chạy, KHÔNG dùng npz cũ)
   │
   ├─[train]──►  best_model.pth            (float32, merged)
   ├─[prune]──►  best_model_pruned.pth     (4,4,8,8)
   ├─[QAT]────►  model_qat_int8.pth   ◄── ★ SINGLE SOURCE OF TRUTH
   │                 │
   │                 ├─[export]──►  flat_weights.hex + w_ram0..7.hex + conv_bias.hex + fc_*.hex
   │                 │                  └─► dùng cho CẢ V-ROM ($readmemh) lẫn V-RAM (Avalon load)
   │                 │
   │                 ├─[golden ALL test]──►  golden/<set><N>/ cho TOÀN BỘ test split
   │                 │                  ├─ merged_test (Chapman+Ningbo held-out)
   │                 │                  └─ georgia_test (toàn bộ Georgia)
   │                 │
   │                 ├─[SW eval]──►  acc/F1/CM/ROC trên cùng test split  (số SW chính thức)
   │                 └─[cross-check]──►  Georgia zero-shot + C2==C6 decomposition
   │
   └─ MỌI con số (SW acc, RTL acc, FPGA acc) phải truy về đúng model_qat_int8.pth này.
```

**Bất biến phải giữ:** số acc của SW (`evaluate_quantized`), của RTL sim (TB batch), và của board (nếu chạy) phải **khớp nhau** trên **cùng test split**. Lệch = bug, phải truy.

---

## 3. Kế hoạch chi tiết — theo Phase

> Mỗi bước ghi rõ **verify** (success criteria) theo CLAUDE.md mục 4.

### Phase 0 — Chuẩn bị & chốt data (0.5 ngày)

| # | Việc | Verify |
|---|---|---|
| 0.1 | **Re-build** npz từ WFDB: `python cross_eval/case1_build.py` (đọc toàn bộ `data/ningba/WFDBRecords`, ~90k file → ~43k record có nhãn). Ghi lại record count + class dist + nguồn (Chapman-half src=0 / Ningbo-half src=1). | In ra `kept`, `train/val/test` count + `np.bincount` per split + `Chapman-half/Ningbo-half` count; lưu vào `final_run_log.md`. npz được ghi mới (mtime cập nhật). |
| 0.2 | Xác nhận `data/georgia_by_class/` đầy đủ (⚠️ memory ghi đang 5552, thiếu 54 GSVT so với gốc 5606 — **phải làm rõ trước khi đo**). | Đếm file/class = gốc; nếu thiếu, re-run `georgia_preprocess.py` rồi ghi log. |
| 0.3 | Backup branch hiện tại; tạo nhánh `feature/unified-rerun`. | `git branch` cho thấy nhánh mới; working tree sạch các file rác (`a.txt`, `No_use/`, transcript…). |

### Phase 1 — Loader cho dataset gộp (0.5 ngày)

**Vấn đề**: `utils/dataset.py::get_dataloaders()` chỉ đọc Chapman thuần. Các script `train.py/prune_finetune.py/qat_int8.py/generate_golden*.py` đều gọi loader này → cần một loader đọc `case1_merged.npz` mà **không phá** đường Chapman cũ.

| # | Việc | Verify |
|---|---|---|
| 1.1 | Thêm loader mới (đề xuất: `utils/dataset_merged.py` hoặc cờ `--npz` cho `get_dataloaders`) đọc `case1_merged.npz`, trả về `(train, val, test)` cùng interface `(ecg, label[, hr])`. **Surgical**: không sửa logic Chapman cũ. | Unit check: shape `(N,1,2500)`, label ∈ {0..3}, split count khớp `case1_build`. |
| 1.2 | Quyết định cách các script chọn data: thêm `--data_npz` (ưu tiên nếu có), fallback `--data_dir` Chapman. Áp cho `train.py`, `prune_finetune.py`, `qat_int8.py`, `evaluate_quantized.py`, `generate_golden_batch.py`. | Chạy thử `--help` mỗi script thấy arg mới; chạy 1 epoch smoke-test không lỗi. |

> **Lưu ý kiến trúc**: KHÔNG hard-code đường dẫn. Giữ Chapman-only path hoạt động (fallback) để các verify cũ (21/21 tb_top) vẫn tái lập được.

### Phase 2 — Software pipeline trên Chapman mở rộng (1–1.5 ngày máy)

Chạy **tuần tự**, mỗi bước verify trước khi sang bước sau:

| # | Lệnh (trong `software/python`, venv active) | Verify |
|---|---|---|
| 2.1 train | `python train.py --data_npz ..\..\data\case_study\case1_merged.npz` | `best_model.pth` sinh ra; val_acc hội tụ, log vào `final_run_log.md`. |
| 2.2 prune | `python prune_finetune.py --checkpoint .\results\best_model.pth --data_npz <merged>` | `best_model_pruned.pth`; channels = (4,4,8,8); acc drop bounded. |
| 2.3 QAT | `python quantization\qat_int8.py --checkpoint .\results\best_model_pruned.pth --data_npz <merged> --output_dir .\results\qat_int8` | `model_qat_int8.pth`; **INT8 bit-exact acc == float eval** (in cả 2 số). |
| 2.4 SW eval | `python quantization\evaluate_quantized.py` (trên **merged test split**) | Ra **acc / F1-macro / per-class / CM / ROC** → đây là **số SW chính thức mới**. Lưu `results/ablation_quant/` + `results/figures/`. |
| 2.5 export | `python export_weights_int8.py --checkpoint .\results\qat_int8\model_qat_int8.pth --output_dir .\results\weights_int8` | Sinh `flat_weights.hex`, `w_ram0..7.hex`, `conv_bias.hex`, `fc_*.hex`. Copy vào `hardware/RTL/` + `hardware/fpga/simulation/questa/`. |

> **Cảnh báo nb/scale**: model train trên data mới có thể ra **nb per-layer khác** Chapman (giống PTB-XL từng ra nb[Conv3]=7). Phải **đọc nb/w_shift thực tế từ checkpoint** và cập nhật:
> - `golden` (tự lấy từ ckpt — OK),
> - **CONFIG mặc định/đường ROM của RTL** nếu nb đổi (memory `barrel-shifter-nb-narrow`: nb max=8, bus packing không đổi).
> Verify: nb in ra từ ckpt == nb dùng trong golden == nb nạp vào RTL.

### Phase 3 — Golden cho TOÀN BỘ test split (0.5–1 ngày)

**Vấn đề**: `generate_golden_batch.py` hiện hard-code 10 Chapman + 10 PTB-XL class-balanced. Cần: (a) đổi nguồn sang merged-test + georgia-test, (b) **không giới hạn 10** → toàn bộ test split.

| # | Việc | Verify |
|---|---|---|
| 3.1 | Sửa/clone `generate_golden_batch.py` → `generate_golden_full.py`: nguồn = `merged_test` + `georgia_test`; `--per_group` bỏ giới hạn (hoặc = N_test). Sinh `ecg_<set><N>.hex` + `<set><N>/*.mem` + `golden_meta.json` cho **mọi** sample test. | Số thư mục golden == số sample test; mỗi thư mục đủ 7 file `.mem`. |
| 3.2 | Sinh `batch_summary.json` chứa per-sample true/pred + **acc tổng SW** (đây là số SW-golden, phải == Phase 2.4). | `batch_summary` acc == evaluate_quantized acc (cùng test set). Lệch = bug golden. |
| 3.3 | Quyết định **dung lượng/thời gian**: toàn bộ test ~ vài nghìn sample × ~5216 cycle/sim. Ước lượng sim wall-time trước (xem §5 Rủi ro). Nếu quá nặng → báo giảng viên xin subset cân lớp lớn (ghi rõ "subset"). | Bảng ước lượng thời gian trong `final.md` §5; quyết định ghi log. |

### Phase 4 — Hardware: 2 phiên bản trọng số (1 ngày)

Cơ chế đã có (`cp_engine.v:230`); việc còn lại là **đóng gói chính thức + verify song song**.

| Phiên bản | Cách build | TB |
|---|---|---|
| **V-ROM** (nạp cứng) | mặc định `$readmemh` w_ram*.hex/conv_bias.hex (KHÔNG define gì) | `tb_top.v` / TB batch mới |
| **V-RAM** (nạp runtime) | compile `+define+NO_WEIGHT_INIT`, nạp trọng số qua Avalon trước inference | `tb_weight_load.v` / TB batch mới (load weight rồi loop) |

| # | Việc | Verify |
|---|---|---|
| 4.1 | Cập nhật `w_ram*.hex`/`conv_bias.hex`/`fc_*.hex` trong sim dir = export mới (Phase 2.5). | File hex trùng giữa `results/weights_int8` và sim dir (diff sạch). |
| 4.2 | Chạy lại regression cũ trên cả 2 phiên bản với weight mới: `run_tb_top.do` (V-ROM) + `run_tb_weight_load.do` (V-RAM). | Cả 2 PASS; **V-ROM result == V-RAM result** bit-exact (cùng logits). |
| 4.3 | Tài liệu hóa 2 phiên bản trong `System_Design.md` (build command, khác biệt resource nếu synth). | Mục mới trong System_Design.md; bảng so V-ROM vs V-RAM (resource để Phase C sau). |

### Phase 5 — TB đo accuracy TOÀN BỘ (1–1.5 ngày)

**Vấn đề**: chưa có TB lặp toàn bộ test set in accuracy. `tb_top.v` lặp 3; `tb_topo_sweep.v` lặp topology (không phải data).

| # | Việc | Verify |
|---|---|---|
| 5.1 | Viết `tb_batch_acc.v`: đọc danh sách `ecg_<set><N>.hex` + golden argmax, lặp qua **mọi** sample, chạy inference, so argmax với golden, đếm match → in **RTL accuracy + #mismatch**. Hỗ trợ cả V-ROM và V-RAM (define). | TB chạy hết không treo; in `RTL acc = …`, `mismatch list`. |
| 5.2 | Tạo `run_tb_batch_acc.do` (2 biến thể define cho 2 phiên bản). | 2 .do chạy được trong Questa. |
| 5.3 | **Đối chiếu 3 số trên cùng test split**: SW acc (2.4) == golden acc (3.2) == RTL acc (5.1). | 3 số **khớp** (lý tưởng bit-exact mọi logit; nếu chỉ argmax khớp, ghi rõ). Lệch → debug như lịch sử 7-fix trong System_Design.md. |
| 5.4 | Cross-check Georgia: RTL acc trên georgia_test == SW Georgia zero-shot. Ghi C2==C6 decomposition (quant drop vs distribution). | Bảng cross-check Georgia trong log; số RTL == SW. |

### Phase 6 — Tổng hợp & thống nhất số liệu (0.5 ngày)

| # | Việc | Verify |
|---|---|---|
| 6.1 | Lập bảng **UNIFIED_NUMBERS.md**: 1 model, 1 test split, các cột = SW float / SW INT8 / RTL V-ROM / RTL V-RAM / (board nếu có). Acc, F1, per-class, CM. | Mọi cột truy về `model_qat_int8.pth` mới; ngày đo ghi rõ "cùng một lần chạy". |
| 6.2 | Cập nhật `PAPER_DATA.md` (source viết bài) bằng số mới — **đánh dấu rõ số cũ Chapman-thuần đã bị thay**. Không xoá lịch sử, chỉ ghi chú. | PAPER_DATA.md có section "Unified rerun 2026-06-xx". |
| 6.3 | Cập nhật memory: số acc chính thức mới, định nghĩa "Chapman mở rộng", 2 phiên bản HW. | File memory tương ứng được sửa (không tạo trùng). |

---

## 4. Những thứ KHÔNG nằm trong task này (cần phần cứng/Quartus — làm sau khi duyệt)

- **Synthesis Quartus** (Fmax/resource/power) cho V-ROM vs V-RAM — cần Quartus GUI (`D:\altera_lite\25.1std`), không tự chạy headless ổn định. → Phase C riêng.
- **On-board DE10** (JTAG/UART) — cần phần cứng vật lý. → Phase D riêng.
- Hai cái trên **không chặn** việc thống nhất số liệu SW↔RTL-sim (mục tiêu chính của task này).

---

## 5. Rủi ro & ước lượng

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| **Sim toàn bộ test set quá nặng** (vài nghìn sample × 5216 cy) | Cao | Ước lượng wall-time trước (1 sample sim ≈ ? s × N). Nếu > vài giờ → xin giảng viên duyệt **subset cân lớp lớn** (vd 200–500/lớp), ghi rõ "subset" trong báo cáo. |
| **nb/scale đổi** theo data mới phá CONFIG mặc định RTL | Trung bình | Đọc nb từ ckpt, cập nhật golden + RTL CONFIG đồng bộ; verify nb 3 nơi khớp. |
| **Acc tụt** khi train trên data gộp (domain khác Chapman thuần) | Trung bình | Đây là số THẬT của data mới — báo cáo trung thực, không tô. So với baseline Chapman-thuần như một dòng tham chiếu. |
| **Georgia thiếu 54 GSVT** (memory) | Thấp-TB | Phase 0.2 làm rõ trước khi đo; re-preprocess nếu cần. |
| **Loader mới phá verify cũ** (21/21 tb_top Chapman) | Trung bình | Giữ fallback Chapman; chạy lại tb_top Chapman-weight để xác nhận đường cũ còn nguyên. |
| Bit-exact SW↔RTL không khớp trên data mới | Trung bình | Dùng đúng quy trình debug 7-fix đã ghi trong System_Design.md (pipeline depth, GAP floor, round-half-up). |

**Ước lượng công**: ~5–6 ngày máy+người cho Phase 0–6 (chưa gồm Quartus/board).

---

## 6. Thứ tự thực thi (sau khi giảng viên duyệt)

```
0 (data) → 1 (loader) → 2 (SW pipeline) → 3 (golden full) → 4 (HW 2 phiên bản) → 5 (TB acc) → 6 (unify)
```

Mỗi Phase có **verify gate** — không qua gate thì dừng, báo cáo, không chạy bước sau.

---

## 7. Câu hỏi cần giảng viên chốt trước khi chạy

1. **Subset hay full test split cho TB?** Nếu full quá nặng (§5), chấp nhận subset cân lớp lớn không?
2. **Số SW có thể tụt so với 94.65% Chapman-thuần** (vì data gộp khác phân phối) — chấp nhận báo cáo số thật chứ không giữ số cũ?
3. **Cross-check chỉ Georgia** đã đủ cho story C3, hay cần giữ thêm PTB-XL như dòng tham chiếu?
4. **2 phiên bản HW**: chỉ cần verify sim bit-exact (task này), hay yêu cầu cả synthesis so resource (Phase C, cần Quartus)?
```
