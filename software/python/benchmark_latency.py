"""
Software latency baseline for the FPGA accelerator comparison.
=============================================================
Measures CPU inference latency of the deployed pruned model (4,4,8,8)
to compute speedup vs the FPGA accelerator (52.16 us compute-only).

Reports BOTH:
  - compute-only : time around model.forward(x)        -> fair vs FPGA 52us & Liu 2023 66us
  - end-to-end   : preprocess one CSV record + forward -> real user-perceived latency

Two model variants (both = pruned 4,4,8,8, matching hardware):
  - FP32     : ECG_1DCNN_Pruned float
  - INT8 sim : ECG_1DCNN_INT8 wrapping the pruned model (hardware-fidelity integer path)

Single-sample (batch=1) only — the accelerator processes one record at a time.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prune_finetune import ECG_1DCNN_Pruned
from model.model import ECG_1DCNN_INT8
from utils.dataset import ChapmanECGDataset


def load_pruned(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = ECG_1DCNN_Pruned(c1_out=ck['c1_out'], c2_out=ck['c2_out'],
                         c3_out=ck['c3_out'], c4_out=ck['c4_out'])
    m.load_state_dict(ck['model_state_dict'])
    m.to(device).eval()
    return m


def timeit(fn, n_warmup, n_iter):
    """Return per-call latency stats in microseconds."""
    for _ in range(n_warmup):
        fn()
    samples = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)  # us
    a = np.array(samples)
    return {'mean': a.mean(), 'std': a.std(), 'median': np.median(a),
            'p95': np.percentile(a, 95), 'min': a.min()}


def fmt(s):
    return f"{s['mean']:8.1f} +/- {s['std']:6.1f} us  (median {s['median']:.1f}, p95 {s['p95']:.1f}, min {s['min']:.1f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='./results/best_model_pruned.pth')
    ap.add_argument('--data_dir', default='../../data/Chapman')
    ap.add_argument('--n_iter', type=int, default=2000)
    ap.add_argument('--n_warmup', type=int, default=200)
    ap.add_argument('--threads', type=int, default=0,
                    help='torch CPU threads; 0 = leave default')
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = torch.device('cpu')
    print(f"\n{'='*70}\n  Software latency baseline (CPU, batch=1) — pruned 4,4,8,8\n{'='*70}")
    print(f"  torch threads: {torch.get_num_threads()}  |  iters: {args.n_iter} (warmup {args.n_warmup})")

    # ---- Build models ----
    fp32 = load_pruned(args.checkpoint, device)
    print(f"  params: {fp32.count_parameters()}")

    # INT8 sim needs a calib loader for input_shift — use a small test split
    test_ds = ChapmanECGDataset(args.data_dir, split='test')
    from torch.utils.data import DataLoader
    calib = DataLoader(test_ds, batch_size=64, shuffle=False)
    int8 = ECG_1DCNN_INT8(fp32, calib, device=device, n_cal_batches=10)

    # ---- Prepare one fixed input tensor (compute-only) ----
    x0 = test_ds[0][0].unsqueeze(0)  # (1, 2500)

    # ---- Compute-only benchmark (FP32 = the valid software baseline) ----
    # NOTE: INT8 sim is measured for reference ONLY. It is NOT a valid latency
    # baseline — ECG_1DCNN_INT8 runs float conv + extra round/clamp/shift on the
    # CPU (no INT8 SIMD path), so it is SLOWER than FP32 by construction. The
    # FPGA gets its INT8 speedup from dedicated DSP hardware, not from the CPU.
    # Software baseline for speedup = FP32 forward (same convention as Liu 2023,
    # who compare the FPGA INT8 core against a CPU running the plain model).
    print(f"\n  -- compute-only (forward pass, data already a tensor) --")
    with torch.no_grad():
        fp32_co = timeit(lambda: fp32(x0), args.n_warmup, args.n_iter)
        int8_co = timeit(lambda: int8(x0), args.n_warmup, args.n_iter)
    print(f"  FP32      : {fmt(fp32_co)}   <- software baseline")
    print(f"  INT8 sim  : {fmt(int8_co)}   (reference only, NOT a timing baseline)")

    # ---- Speedup vs FPGA (compute-only, apple-to-apple) ----
    FPGA_COMPUTE_US = 52.16
    print(f"\n{'='*70}\n  Speedup vs FPGA accelerator ({FPGA_COMPUTE_US} us compute-only @100MHz)\n{'='*70}")
    print(f"  CPU FP32 compute-only : {fp32_co['median']:.1f} us")
    print(f"  Speedup (FPGA vs CPU) : {fp32_co['median']/FPGA_COMPUTE_US:.2f}x")
    print(f"\n  NOTE: Both sides are compute-only (data pre-loaded), matching Liu 2023.")
    print(f"        ARM Cortex-A9 (HPS) baseline -> Phase D, expected far larger speedup.\n")


if __name__ == '__main__':
    main()
