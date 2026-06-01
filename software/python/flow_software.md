# Software Export Flow — CNN Accelerator

## Môi trường
- **Windows**, Python 3.14, venv tại `d:\Thesis101\.venv`
- `cd d:\Thesis101\software\python`
- Dataset: `d:\Thesis101\data\Chapman`
- Activate venv: `.\.venv\Scripts\Activate.ps1` (PowerShell) hoặc `.venv\Scripts\activate.bat` (CMD)

## Packages cần thiết
```
torch==2.12.0+cpu  numpy  scipy  scikit-learn  wfdb  tqdm  pandas  matplotlib
```
Cài một lần: `pip install wfdb scikit-learn tqdm pandas`

---

## Bước 1: Re-train (✅ DONE — checkpoint đã có)
```powershell
python train.py --data_dir d:\Thesis101\data\Chapman
python prune_finetune.py --checkpoint .\results\best_model.pth `
    --data_dir d:\Thesis101\data\Chapman
# → results/best_model_pruned.pth  (channels 4,4,8,8 ✅)
```

## Bước 2: QAT-INT8 (✅ DONE — model đã có)
```powershell
python quantization\qat_int8.py `
    --checkpoint .\results\best_model_pruned.pth `
    --output_dir .\results\qat_int8 `
    --data_dir d:\Thesis101\data\Chapman
# → results/qat_int8/model_qat_int8.pth  (94.65% acc, F1=0.9404 ✅)
```

## Bước 3: Export weights → flat_weights.hex (✅ DONE)
```powershell
python export_weights_int8.py `
    --checkpoint .\results\qat_int8\model_qat_int8.pth `
    --output_dir .\results\weights_qat_int8
# → results/weights_qat_int8/flat_weights.hex  (580 INT8, KHÔNG có comment ✅)
# Đã copy sang: hardware/RTL/flat_weights.hex + hardware_v1/RTL/flat_weights.hex
```

## Bước 4: Export golden files (✅ DONE — 3 samples)
```powershell
python generate_golden.py `
    --checkpoint .\results\qat_int8\model_qat_int8.pth `
    --data_dir d:\Thesis101\data\Chapman `
    --output_dir .\results\golden `
    --sample_idx 0   # lặp với 1, 2 cho 3 samples
# → results/golden_{0,1,2}/  (21 checkpoints mỗi sample: input + pool1-4 + gap + logits ✅)
# RTL verification: 21/21 bit-exact PASS ✅
```

---

## Phase A' — QAT Ablation (cho paper contribution C1)

### A3: General-scale INT8 QAT
```powershell
python quantization\qat_int8_general.py `
    --checkpoint .\results\best_model_pruned.pth `
    --output_dir .\results\qat_int8_general `
    --data_dir d:\Thesis101\data\Chapman
# So sánh accuracy vs qat_int8 (power-of-2)
```

### A4: Power-of-2 + floor (không round-half-up)
```powershell
python quantization\qat_int8.py `
    --checkpoint .\results\best_model_pruned.pth `
    --output_dir .\results\qat_int8_floor `
    --data_dir d:\Thesis101\data\Chapman `
    --rescale-mode floor
# Ablation: chứng minh round-half-up cần thiết
```

---

## Phase A — Cross-Dataset MIT-BIH (cho paper contribution C3)

### Download MIT-BIH
```powershell
python cross_eval\download_mitbih.py   # wfdb.dl_database('mitdb', ...)
# → data/mitbih/  (47 records, 360Hz)
```

### Preprocess + Eval 5 modes
```powershell
python cross_eval\mitbih_eval.py `
    --chapman_ckpt .\results\qat_int8\model_qat_int8.pth `
    --data_dir d:\Thesis101\data `
    --output_dir .\results\cross_eval
# → results/cross_eval/{zero_shot,linear_probe,finetune,scratch,float_baseline}_metrics.json
```

---

## Lưu ý quan trọng
- `flat_weights.hex`: KHÔNG có comment lines — `$readmemh` đọc từ byte 0
- `nb` per layer: Conv1=8, Conv2=6, Conv3=6, Conv4=7 (hardcoded trong RTL cnn_controller.v)
- ReLU **chỉ** sau Conv4 — Conv1-3 không có
- GAP: integer floor `sum >> 2` (không phải float average)
- FC: nb=0, raw INT32 logits → argmax
- Dataset path Windows: dùng `d:\Thesis101\data\Chapman` (backslash OK trong argparse)
