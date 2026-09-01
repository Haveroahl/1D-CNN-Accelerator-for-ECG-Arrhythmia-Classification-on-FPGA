"""Benchmark: two software baselines vs the hardware accelerator.

Both software baselines compute the SAME bit-exact INT8 arithmetic the RTL
implements (acc → +bias → >>nb round-half-up → clamp → ReLU → pool → GAP → FC),
so every row of the table is the same work on a different platform:

  1. PyTorch INT8 (`int8_forward_golden`) — the reference the golden .mem files
     come from. Convenient, but ~94% of its wall-clock is framework dispatch
     overhead on a network this small, so on its own it flatters the hardware.
  2. Optimised C (`hardware/fpga/sw/niosv/cnn_sw.c`, -O2) — the honest baseline:
     plain integer loops, no framework, same CPU. This is the number to quote.

Reporting both is deliberate: it shows the speedup was not obtained by picking a
weak baseline, and it isolates how much of the "speedup" is really PyTorch
overhead. The C baseline is also the exact source file the Nios V firmware
compiles, so the two stay in sync.

Hardware side is the deterministic measured latency from tb_top.v run_inference:
5216 cycles / inference @ 100 MHz = 52.16 µs, identical for every input.

The C baseline is built and run automatically when a compiler is found; pass
--skip_c to report the PyTorch row only.

Usage:
    python bench_sw_vs_hw.py \
        --checkpoint ./results/qat_int8/model_qat_int8.pth \
        --data_dir   ../../data/Chapman
"""

import os
import re
import sys
import time
import shutil
import tempfile
import argparse
import subprocess
import statistics

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantization.qat_int8 import ECG_1DCNN_QAT
from utils.dataset import get_dataloaders
from generate_golden import int8_forward_golden

# Hardware reference (measured, deterministic — tb_top.v run_inference)
HW_CYCLES = 5216
HW_CLOCK_MHZ = 100.0
HW_LATENCY_US = HW_CYCLES / HW_CLOCK_MHZ  # 52.16 µs

# Measured on the synthesised design (Quartus PowerPlay driven by a
# full-inference VCD, 95.6% toggle coverage) — see PROJECT.md Phase C.
HW_TOTAL_MW = 623.0
HW_DYNAMIC_MW = 198.0

# The C baseline shares cnn_sw.c with the Nios V firmware, so both stay in sync.
NIOSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', 'hardware', 'fpga', 'sw', 'niosv')
# Prefer whatever is on PATH; the Quartus-bundled mingw is a last resort because
# its cc1.exe is a non-functional stub in some installs (it exists and exits 0
# but compiles nothing), so existence on disk is not proof of a usable compiler.
GCC_CANDIDATES = [
    'gcc', 'cc', 'clang',
    r'D:/altera_lite/25.1std/questa_fse/gcc-7.4.0-mingw64vc16/bin/gcc.exe',
]


def _can_compile(cc):
    """A compiler counts only if it actually produces an executable."""
    try:
        with tempfile.TemporaryDirectory() as d:
            src, exe = os.path.join(d, 'p.c'), os.path.join(d, 'p.exe')
            with open(src, 'w') as f:
                f.write('int main(void){return 0;}\n')
            r = subprocess.run([cc, '-O2', src, '-o', exe],
                               capture_output=True, text=True)
            return r.returncode == 0 and os.path.isfile(exe)
    except Exception:
        return False


def find_gcc():
    for c in GCC_CANDIDATES:
        path = c if os.path.isabs(c) else shutil.which(c)
        if path and os.path.isfile(path) and _can_compile(path):
            return path
    return None


def run_c_baseline(inp, preds, workdir):
    """Build and run the optimised-C baseline. Returns a stats dict or None."""
    gcc = find_gcc()
    if gcc is None:
        print("[WARN] no C compiler found - skipping the optimised-C baseline")
        return None

    os.makedirs(workdir, exist_ok=True)
    inp_bin = os.path.join(workdir, 'inp.bin')
    pred_bin = os.path.join(workdir, 'pred.bin')
    inp.astype(np.int8).tofile(inp_bin)
    preds.astype(np.int64).tofile(pred_bin)

    exe = os.path.join(workdir, 'bench_c.exe')
    cmd = [gcc, '-O2', '-I', NIOSV_DIR,
           os.path.join(NIOSV_DIR, 'bench_c_host.c'),
           os.path.join(NIOSV_DIR, 'cnn_sw.c'), '-o', exe]
    build = subprocess.run(cmd, capture_output=True, text=True)
    if build.returncode != 0:
        print(f"[WARN] C baseline build failed:\n{build.stderr[:500]}")
        return None

    res = subprocess.run([exe, inp_bin, pred_bin, str(len(preds))],
                         capture_output=True, text=True)
    out = res.stdout
    print(f"[INFO] C baseline: {out.splitlines()[0] if out else '(no output)'}")
    if res.returncode != 0 and 'BIT-EXACT PASS' not in out:
        print(f"[WARN] C baseline reported mismatches:\n{out[:400]}")
        return None

    m = re.search(r'median=([\d.]+) mean=([\d.]+) min=([\d.]+) '
                  r'max=([\d.]+) p95=([\d.]+)', out)
    if not m:
        print(f"[WARN] could not parse C baseline output:\n{out[:400]}")
        return None
    med, mean, mn, mx, p95 = (float(x) for x in m.groups())
    return {'median': med, 'mean': mean, 'min': mn, 'max': mx, 'p95': p95,
            'bit_exact': 'BIT-EXACT PASS' in out}


def run(args):
    device = torch.device('cpu')
    torch.set_num_threads(args.threads)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    is_pruned = 'c1_out' in ckpt
    qat_model = ECG_1DCNN_QAT(
        c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
        c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
    ) if is_pruned else ECG_1DCNN_QAT()
    qat_model.load_state_dict(ckpt['model_state_dict'])
    qat_model.eval()

    w_int8      = {k: np.array(v, dtype=np.int8)    for k, v in ckpt['w_int8'].items()}
    b_int8      = {k: np.array(v, dtype=np.float64) for k, v in ckpt['b_int8'].items()}
    nb          = ckpt['nb']
    w_shift     = ckpt['w_shift']
    input_shift = ckpt['input_shift_bits']

    _, _, test_loader = get_dataloaders(args.data_dir, batch_size=256, num_workers=0)
    all_x = torch.cat([b[0] for b in test_loader], dim=0)
    n_total = all_x.shape[0]
    n_run = min(args.num_samples, n_total) if args.num_samples > 0 else n_total

    print(f"[INFO] test samples available={n_total}, timing {n_run}, "
          f"threads={args.threads}")

    def infer(i):
        with torch.no_grad():
            return int8_forward_golden(
                qat_model, all_x[i].unsqueeze(0),
                w_int8, b_int8, nb, w_shift, input_shift, device)

    # Warmup (JIT/cache/thread pool)
    for i in range(min(20, n_run)):
        infer(i)

    # Time PyTorch and, in the same pass, keep the INT8 inputs and predicted
    # classes so the C baseline runs on exactly these inputs and is checked
    # against exactly these golden classes.
    per_sample_us = []
    c_inp = np.zeros((n_run, 2500), dtype=np.int8)
    c_pred = np.zeros(n_run, dtype=np.int64)
    for i in range(n_run):
        t0 = time.perf_counter()
        st = infer(i)
        per_sample_us.append((time.perf_counter() - t0) * 1e6)
        c_inp[i] = st['input_int8'].cpu().numpy().astype(np.int8)
        c_pred[i] = st['predicted_class']

    sw_median = statistics.median(per_sample_us)
    sw_mean   = statistics.mean(per_sample_us)
    sw_min    = min(per_sample_us)
    sw_max    = max(per_sample_us)
    sw_p95    = sorted(per_sample_us)[int(0.95 * (n_run - 1))]

    # Optimised-C baseline on the same CPU, same inputs, same golden classes.
    c = None
    if not args.skip_c:
        workdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'results', 'bench_c')
        c = run_c_baseline(c_inp, c_pred, workdir)

    hw_thr = 1e6 / HW_LATENCY_US
    hw_energy_dyn_uj   = HW_DYNAMIC_MW * HW_LATENCY_US / 1000.0
    hw_energy_total_uj = HW_TOTAL_MW   * HW_LATENCY_US / 1000.0

    def row(med, mean, mn, mx, p95, power_w):
        """Per-platform derived metrics (energy is an estimate for CPU rows)."""
        return {
            'med': med, 'mean': mean, 'min': mn, 'max': mx, 'p95': p95,
            'thr': 1e6 / med,
            'speedup': med / HW_LATENCY_US,
            'energy_uj': power_w * 1000.0 * med / 1000.0,
            'jitter': p95 - mn,
            'spread': mx / mn,
        }

    py = row(sw_median, sw_mean, sw_min, sw_max, sw_p95, args.cpu_power_w)
    cc = (row(c['median'], c['mean'], c['min'], c['max'], c['p95'],
              args.cpu_power_w) if c else None)

    def f(v, fmt, dash='n/a'):
        return dash if v is None else format(v, fmt)

    print()
    print("| Metric | PyTorch INT8 (CPU) | Optimised C -O2 (CPU) | Accelerator |")
    print("|---|---|---|---|")
    print(f"| Platform | {args.threads}-thread CPU, framework | "
          f"{args.threads}-thread CPU, plain loops | "
          f"Cyclone V FPGA @ {HW_CLOCK_MHZ:.0f} MHz |")
    print(f"| Latency / inference (median) | {py['med']:.1f} us | "
          f"{f(cc and cc['med'], '.1f')} us | "
          f"{HW_LATENCY_US:.2f} us ({HW_CYCLES} cycles) |")
    print(f"| Latency min / max | {py['min']:.1f} / {py['max']:.1f} us | "
          f"{f(cc and cc['min'], '.1f')} / {f(cc and cc['max'], '.1f')} us | "
          f"{HW_LATENCY_US:.2f} (fixed) |")
    print(f"| Throughput | {py['thr']:,.0f} inf/s | "
          f"{f(cc and cc['thr'], ',.0f')} inf/s | {hw_thr:,.0f} inf/s |")
    print(f"| **Speedup vs accelerator** | {py['speedup']:.1f}x slower | "
          f"**{f(cc and cc['speedup'], '.2f')}x slower** | **1x** |")
    print(f"| Energy / inference | ~{py['energy_uj']:,.0f} uJ (est.) | "
          f"~{f(cc and cc['energy_uj'], ',.0f')} uJ (est.) | "
          f"{hw_energy_total_uj:.1f} uJ total / {hw_energy_dyn_uj:.1f} uJ dyn |")
    if cc:
        print(f"| **Energy ratio** | ~{py['energy_uj']/hw_energy_total_uj:,.0f}x | "
              f"**~{cc['energy_uj']/hw_energy_total_uj:,.0f}x** | 1x |")
    print(f"| Jitter (max/min spread) | {py['spread']:.1f}x | "
          f"**{f(cc and cc['spread'], '.1f')}x** | **1.00x (0 jitter)** |")
    print()

    print("[HOW TO READ THIS]")
    print("  * The optimised-C column is the HONEST latency baseline. PyTorch's")
    print("    extra time is framework dispatch overhead, not useful computation,")
    print(f"    so the {py['speedup']:.0f}x figure overstates the hardware's advantage.")
    if cc:
        print(f"  * Quote {cc['speedup']:.2f}x for latency. The accelerator wins that")
        print(f"    while clocked at {HW_CLOCK_MHZ:.0f} MHz against a multi-GHz CPU.")
    print("  * The durable advantages are energy (~2 orders of magnitude) and")
    print("    determinism (fixed 5216 cycles vs CPU spread), not raw latency.")
    print()
    print(f"[NOTE] CPU rows timed over {n_run} samples on this machine; medians reject")
    print("       OS jitter but absolute values still vary with system load.")
    print(f"[NOTE] HW latency/power are MEASURED (tb_top.v; PowerPlay+VCD 95.6% toggle):"
          f" {HW_TOTAL_MW:.0f} mW total, {HW_DYNAMIC_MW:.0f} mW dynamic.")
    print(f"[NOTE] CPU energy is an ESTIMATE from an assumed {args.cpu_power_w:.0f} W"
          " package power - NOT measured. Report as order of magnitude only.")
    if c and not c['bit_exact']:
        print("[WARN] C baseline did NOT match golden - latency figures unreliable.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='./results/qat_int8/model_qat_int8.pth')
    p.add_argument('--data_dir',   default='../../data/Chapman')
    p.add_argument('--num_samples', type=int, default=500,
                   help='samples to time (0 = full test set)')
    p.add_argument('--threads',    type=int, default=1,
                   help='CPU threads (1 = fair edge-comparable single-core)')
    p.add_argument('--cpu_power_w', type=float, default=15.0,
                   help='assumed sustained CPU package power (W) for the energy '
                        'estimate; this is NOT measured')
    p.add_argument('--skip_c', action='store_true',
                   help='skip the optimised-C baseline (PyTorch row only)')
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
