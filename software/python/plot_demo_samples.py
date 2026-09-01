"""Vẽ 4 tín hiệu ECG dùng trong demo từng mẫu trên board (Chương 4).

Bốn mẫu này chính là các mẫu mà `hardware/fpga/soc/ecg_jtag_one.tcl` nạp vào
accelerator — mỗi lớp một mẫu, chọn từ lần chạy toàn tập trên board (mẫu mà phần
cứng đã dự đoán ĐÚNG). Hình cho thấy "đầu vào trông như thế nào" bên cạnh kết quả
mạch trả về, để minh họa cơ chế thay vì chỉ con số tổng hợp.

Vẽ đúng byte INT8 mà board nhận (đọc thẳng từ file .bin demo), không phải tín hiệu
float gốc — nên hình khớp chính xác thứ phần cứng xử lý.

Chạy:
    python plot_demo_samples.py
Xuất: results/figures/demo_4samples.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEMO_DIR = '../../hardware/fpga/soc/demo_data'
ECG_BIN = os.path.join(DEMO_DIR, 'ningba_test_ecg_int8.bin')
LBL_BIN = os.path.join(DEMO_DIR, 'ningba_test_labels.bin')
OUT = 'results/figures/demo_4samples.png'

# Mẫu dùng trong ecg_jtag_one.tcl — giữ đồng bộ với ::PICKS bên đó.
PICKS = [568, 1578, 2903, 4397]
CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
FS = 250.0          # Hz — 2500 mẫu = 10 s
INPUT_SHIFT = 2     # int8 = round(x * 2^2) → chia lại để hiện thang gốc


def main():
    ecg = np.fromfile(ECG_BIN, dtype=np.int8).reshape(-1, 2500)
    lbl = np.fromfile(LBL_BIN, dtype=np.uint8)
    t = np.arange(2500) / FS

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for ax, idx in zip(axes, PICKS):
        truth = int(lbl[idx])
        x = ecg[idx].astype(np.float32) / (2 ** INPUT_SHIFT)

        ax.plot(t, x, lw=0.7, color='#1a3d6d')
        ax.set_ylabel('Biên độ\n(chuẩn hoá)', fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.margins(x=0)
        ax.set_title(
            f'Mẫu #{idx} — nhãn thực tế: {CLASS_NAMES[truth]}   |   '
            f'FPGA trả về: {CLASS_NAMES[truth]}  ✓',
            fontsize=10, loc='left', pad=6)

    axes[-1].set_xlabel('Thời gian (s)', fontsize=10)
    fig.suptitle(
        'Bốn tín hiệu ECG dùng trong demo trên DE10-Standard\n'
        'INT8 (đúng byte nạp vào accelerator), 2500 mẫu = 10 s @ 250 Hz',
        fontsize=11.5, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=160)
    # Thông báo ra console dùng ASCII: console Windows mặc định cp1252 không in
    # được tiếng Việt (nhãn trong hình thì vẫn tiếng Việt bình thường).
    print(f'Saved: {OUT}')
    for idx in PICKS:
        print(f'  sample #{idx}: {CLASS_NAMES[int(lbl[idx])]}')


if __name__ == '__main__':
    main()
