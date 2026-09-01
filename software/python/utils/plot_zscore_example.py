"""Plot one Chapman ECG record after the dataset preprocessing pipeline."""

import argparse
import os

import matplotlib
import numpy as np
import openpyxl
from scipy.signal import resample

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import CLASS_NAMES, RHYTHM_TO_4CLASS


def load_first_train_record(data_dir, seed=42, lead=1, target_fs=250):
    diag_path = os.path.join(data_dir, "Diagnostics.xlsx")
    workbook = openpyxl.load_workbook(diag_path, read_only=True)
    rows = workbook.active.iter_rows(values_only=True)
    next(rows)
    entries = [
        (row[0], RHYTHM_TO_4CLASS[row[1]], row[5])
        for row in rows
        if row[1] in RHYTHM_TO_4CLASS
    ]
    workbook.close()

    indices = np.random.RandomState(seed).permutation(len(entries))
    train_indices = indices[: int(0.8 * len(entries))]
    target_len = target_fs * 10

    for index in train_indices:
        filename, label, heart_rate = entries[index]
        csv_path = os.path.join(data_dir, f"{filename}.csv")
        if not os.path.exists(csv_path):
            continue

        ecg_12 = np.loadtxt(
            csv_path, delimiter=",", skiprows=1, encoding="utf-8-sig"
        )
        lead_signal = ecg_12[:, lead].astype(np.float64)
        if len(lead_signal) < int(3.85 * 500):
            continue

        signal = resample(lead_signal, target_len)
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        return filename, label, heart_rate, signal.astype(np.float32)

    raise RuntimeError("No valid training record was found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../../../data/Chapman")
    parser.add_argument(
        "--output", default="../results/example_record_zscore.png"
    )
    args = parser.parse_args()

    filename, label, heart_rate, signal = load_first_train_record(args.data_dir)
    time = np.arange(len(signal)) / 250.0

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(time, signal, color="#1565c0", linewidth=0.75)
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_title(
        "Chapman ECG - Lead II after resampling and Z-score"
        f" | Record {filename} | {CLASS_NAMES[label]}"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude (z-score)")
    ax.set_xlim(0, 10)
    ax.grid(True, alpha=0.22)
    ax.text(
        0.99,
        0.96,
        f"N={len(signal)}, fs=250 Hz\n"
        f"mean={signal.mean():.6f}, std={signal.std():.6f}\n"
        f"HR={heart_rate}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(f"Saved: {output}")
    print(
        f"Record={filename}, class={CLASS_NAMES[label]}, HR={heart_rate}, "
        f"mean={signal.mean():.8f}, std={signal.std():.8f}, "
        f"min={signal.min():.4f}, max={signal.max():.4f}"
    )


if __name__ == "__main__":
    main()
