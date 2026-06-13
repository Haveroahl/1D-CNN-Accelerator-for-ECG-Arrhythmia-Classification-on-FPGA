#!/usr/bin/env python3
# ecg_uart_host.py — PC host driver for the UART-driven ECG accelerator (Phase D).
#
# Replaces the slow JTAG/System-Console flow: instead of ~3 master_write_32 per
# byte over JTAG (~10 h for the full set), the FPGA's uart_wrapper auto-increments
# the SRAM address itself, so one CMD_LOAD ships all 2500 bytes in a single serial
# write. Full 1065-sample sweep drops to a few minutes (UART byte rate bound).
#
# Board:  uart_board_top  ->  ecg_uart_top -> uart_wrapper -> ecg_core (unchanged)
# Wire :  3.3V USB-serial adapter on GPIO_0 (adapter TX->pin1/RXD, RX->pin2/TXD, GND).
#
# Protocol (uart_wrapper.v, 8N1):
#   CMD_LOAD  0xA0 + 2500 data bytes  -> ACK 0x55 after the last byte
#   CMD_START 0xA1                    -> ACK 0x55 (pulses start, clears done)
#   CMD_STATUS0xA2                    -> 1 byte {5'b0, isram_free, done_latched, busy}
#   CMD_RESULT0xA3                    -> 1 byte {6'b0, class[1:0]}
#
# Usage:
#   python ecg_uart_host.py --port COM5                      # full Chapman set
#   python ecg_uart_host.py --port COM5 --max 3              # quick sanity (3 samples)
#   python ecg_uart_host.py --port COM5 --dataset ptbxl      # PTB-XL set
#
# Requires: pyserial  (pip install pyserial)

import argparse
import os
import sys
import time

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial not installed. Run:  pip install pyserial")

# ── protocol constants (match uart_wrapper.v) ──────────────────────────────
CMD_LOAD   = 0xA0
CMD_START  = 0xA1
CMD_STATUS = 0xA2
CMD_RESULT = 0xA3
RESP_ACK   = 0x55

SAMPLE_LEN = 2500
STATUS_DONE = 0x02   # bit1 = done_latched
STATUS_BUSY = 0x01   # bit0 = busy

CLASS_NAMES = {0: "AFIB", 1: "GSVT", 2: "SB", 3: "SR"}

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo_data")

DATASETS = {
    "chapman": ("chapman_test_ecg_int8.bin", "chapman_test_labels.bin"),
    "ptbxl":   ("ptbxl_test_ecg_int8.bin",   "ptbxl_test_labels.bin"),
}


def read_ack(ser, what):
    """Read one byte and require it to be RESP_ACK; raise on timeout/mismatch."""
    b = ser.read(1)
    if len(b) != 1:
        raise TimeoutError(f"no ACK after {what} (timeout). Check baud/wiring/clock.")
    if b[0] != RESP_ACK:
        raise ValueError(f"bad ACK after {what}: got 0x{b[0]:02X}, expected 0x55")


def load_sample(ser, sample_bytes):
    """CMD_LOAD + 2500 payload bytes in one write; wait for ACK."""
    assert len(sample_bytes) == SAMPLE_LEN
    ser.write(bytes([CMD_LOAD]) + sample_bytes)
    ser.flush()
    read_ack(ser, "CMD_LOAD")


def run_inference(ser, poll_timeout=5.0):
    """CMD_START, poll CMD_STATUS until done_latched, return predicted class."""
    ser.write(bytes([CMD_START]))
    ser.flush()
    read_ack(ser, "CMD_START")

    t0 = time.time()
    while True:
        ser.write(bytes([CMD_STATUS]))
        ser.flush()
        st = ser.read(1)
        if len(st) == 1 and (st[0] & STATUS_DONE):
            break
        if time.time() - t0 > poll_timeout:
            raise TimeoutError("inference did not assert done_latched")

    ser.write(bytes([CMD_RESULT]))
    ser.flush()
    res = ser.read(1)
    if len(res) != 1:
        raise TimeoutError("no result byte")
    return res[0] & 0x03


def main():
    ap = argparse.ArgumentParser(description="UART host driver for ECG accelerator")
    ap.add_argument("--port", required=True, help="serial port, e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200, help="baud rate (match RTL BAUD)")
    ap.add_argument("--dataset", choices=list(DATASETS), default="chapman")
    ap.add_argument("--max", type=int, default=0, help="max samples (0 = all)")
    ap.add_argument("--csv", default=None, help="write per-sample results to this CSV")
    args = ap.parse_args()

    ecg_name, lbl_name = DATASETS[args.dataset]
    ecg_path = os.path.join(DEMO, ecg_name)
    lbl_path = os.path.join(DEMO, lbl_name)
    for p in (ecg_path, lbl_path):
        if not os.path.exists(p):
            sys.exit(f"missing data file: {p}")

    with open(ecg_path, "rb") as f:
        ecg_all = f.read()
    with open(lbl_path, "rb") as f:
        lbl_all = f.read()

    n_total = len(ecg_all) // SAMPLE_LEN
    if len(ecg_all) % SAMPLE_LEN != 0:
        sys.exit(f"ECG file size {len(ecg_all)} not a multiple of {SAMPLE_LEN}")
    if len(lbl_all) < n_total:
        sys.exit(f"labels file too short: {len(lbl_all)} < {n_total}")

    n_run = n_total if args.max <= 0 else min(args.max, n_total)
    print(f"dataset={args.dataset}  total={n_total}  running={n_run}  "
          f"port={args.port}@{args.baud}")

    csv_f = open(args.csv, "w", newline="") if args.csv else None
    if csv_f:
        csv_f.write("sample,pred,truth,ok\n")

    correct = 0
    t_start = time.time()
    # 1s read timeout: CMD_LOAD ACK arrives after ~0.22s of TX at 115200 baud.
    with serial.Serial(args.port, args.baud, timeout=1.0) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        for s in range(n_run):
            off = s * SAMPLE_LEN
            sample = ecg_all[off:off + SAMPLE_LEN]
            load_sample(ser, sample)
            pred = run_inference(ser)
            truth = lbl_all[s]
            ok = (pred == truth)
            correct += ok
            if csv_f:
                csv_f.write(f"{s},{pred},{truth},{int(ok)}\n")
            if n_run <= 10 or s % 50 == 0:
                print(f"sample {s:4d} : pred={pred}({CLASS_NAMES[pred]}) "
                      f"truth={truth}({CLASS_NAMES.get(truth,'?')}) "
                      f"{'OK' if ok else 'X'}")

    if csv_f:
        csv_f.close()
    dt = time.time() - t_start
    acc = 100.0 * correct / n_run
    print("-" * 46)
    print(f"Accuracy: {correct}/{n_run} = {acc:.2f}%   ({dt:.1f}s, "
          f"{dt/n_run*1000:.0f} ms/sample)")


if __name__ == "__main__":
    main()
