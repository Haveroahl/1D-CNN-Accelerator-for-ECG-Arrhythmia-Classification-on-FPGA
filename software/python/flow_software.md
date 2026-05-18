# Software Export Flow — CNN Accelerator

## Môi trường
- Linux (WSL hoặc native), Python 3, PyTorch
- `cd /home/duc/Thesis/software/python`
- Dataset: `/home/duc/Thesis/data/Chapman`

## Bước 1: Re-train (nếu chưa có checkpoint channels 4,4,8,8)
```bash
python3 train.py --data_dir /home/duc/Thesis/data/Chapman
python3 prune_finetune.py --checkpoint ./results/best_model.pth \
    --data_dir /home/duc/Thesis/data/Chapman
# target channels: Conv1=4, Conv2=4, Conv3=8, Conv4=8
```

## Bước 2: QAT-INT8
```bash
python3 quantization/qat_int8.py \
    --checkpoint ./results/best_model_pruned.pth \
    --output_dir ./results/qat_int8 \
    --data_dir /home/duc/Thesis/data/Chapman
# → results/qat_int8/model_qat_int8.pth
```

## Bước 3: Export weights → flat_weights.hex (cho cp_engine $readmemh)
```bash
python3 export_weights_int8.py \
    --checkpoint ./results/qat_int8/model_qat_int8.pth \
    --output_dir ./results/weights_qat_int8
# → results/weights_qat_int8/flat_weights.hex  ($readmemh handle // comment tự động)
# Copy sang: hardware/RTL/flat_weights.hex
```

## Bước 4: Export golden files (cho tb_layer.v và tb_top.v)
```bash
python3 generate_golden.py \
    --checkpoint ./results/qat_int8/model_qat_int8.pth \
    --data_dir /home/duc/Thesis/data/Chapman \
    --output_dir ./results/golden \
    --sample_idx 0
# → results/golden/input_int8.mem  (2500 bytes, ECG input INT8)
# → results/golden/after_pool1.mem (500×4,  Conv1 output)
# → results/golden/after_pool2.mem (100×4,  Conv2 output)
# → results/golden/after_pool3.mem (20×8,   Conv3 output)
# → results/golden/after_pool4.mem (4×8,    Conv4 output)
# Copy sang: hardware/RTL/
```

## Lưu ý
- `generate_golden.py` hiện dùng model cũ (3/6/10ch) — cần update để match (4/4/8/8ch) sau re-train
- `flat_weights.hex` format: INT8 weights [oc][ic][tap], INT32 bias little-endian, KHÔNG có comment
- nb per layer: Conv1=8, Conv2=7, Conv3=6, Conv4=8 (hardcoded trong RTL)
