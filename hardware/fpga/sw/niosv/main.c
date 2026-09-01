/*
 * main.c - Nios V/m firmware: drive the ECG CNN accelerator over Avalon-MM.
 *
 * Phase D (stage 2): demonstrate the verified ecg_core running on-chip,
 * controlled by a RISC-V soft-core CPU. The CPU loads each ECG sample into
 * the accelerator's input SRAM (one byte at a time through the avalon_slave
 * register map), starts inference, polls done, reads the predicted class,
 * and prints it. Three embedded Chapman samples (golden class 3/1/2) verify
 * the full CPU+accelerator datapath in both ModelSim simulation and on-board.
 *
 * Register map (avalon_slave.v), WORD address -> BYTE offset from ECG_BASE:
 *   word 0 (byte 0x00) W : sram_din      [7:0]
 *   word 1 (byte 0x04) W : sram_wr_addr  [11:0]
 *   word 2 (byte 0x08) W : sram_we       [0]
 *   word 3 (byte 0x0C) W : start         [0]  (also clears done_latched)
 *   word 4 (byte 0x10) R : status        {done_latched[1], busy[0]}
 *   word 5 (byte 0x14) R : result        [1:0]
 *
 * ECG_BASE = base address assigned to ecg_core_0.avs in nios_system.qsys.
 * Address map (Platform Designer): ecg_core_0.avs = 0x000A_0040.
 */

#include <stdio.h>
#include "io.h"          /* IORD_32DIRECT / IOWR_32DIRECT (Nios V HAL) */
#include "system.h"      /* generated base-address macros from BSP     */
#include "ecg_samples.h" /* ecg_data[N_SAMPLES][SAMPLE_LEN], ecg_golden[] */

/* Pure-C INT8 CNN inference (cnn_sw.c) — bit-exact software twin used to
 * benchmark the accelerator's speedup on the same soft-core clock. */
extern int cnn_sw_infer(const signed char *ecg);

/* Read the RISC-V machine cycle counter (CSR 0xB00). Nios V/m increments
 * mcycle once per core clock, so cycle deltas are a fair same-clock metric. */
static inline unsigned read_mcycle(void)
{
    unsigned c;
    __asm__ volatile ("csrr %0, mcycle" : "=r"(c));
    return c;
}

/* BSP system.h defines ECG_CORE_0_BASE (= 0xa0040) for the Qsys instance
 * ecg_core_0. Use it directly; fall back to the literal if absent. */
#ifdef ECG_CORE_0_BASE
#define ECG_BASE ECG_CORE_0_BASE
#else
#define ECG_BASE 0x000A0040u
#endif

/* Byte offsets of the six registers (word address * 4). */
#define A_DIN    0x00
#define A_ADDR   0x04
#define A_WE     0x08
#define A_START  0x0C
#define A_STAT   0x10
#define A_RES    0x14

/* Load one 2500-byte ECG sample into the accelerator's input SRAM. */
static void load_ecg(const signed char *ecg)
{
    int i;
    for (i = 0; i < SAMPLE_LEN; i++) {
        IOWR_32DIRECT(ECG_BASE, A_DIN,  (unsigned char)ecg[i] & 0xFF);
        IOWR_32DIRECT(ECG_BASE, A_ADDR, (unsigned)i);
        IOWR_32DIRECT(ECG_BASE, A_WE,   1u);
    }
}

/* Pulse start, poll done_latched (status bit 1), return predicted class. */
static int run_inference(void)
{
    unsigned status;
    long tries = 0;

    IOWR_32DIRECT(ECG_BASE, A_START, 1u);   /* start + clear done */

    do {
        status = IORD_32DIRECT(ECG_BASE, A_STAT);
        if (++tries > 1000000L) {
            printf("  TIMEOUT waiting for done (status=0x%x)\n", status);
            return -1;
        }
    } while ((status & 0x2u) == 0u);        /* wait for done_latched */

    return (int)(IORD_32DIRECT(ECG_BASE, A_RES) & 0x3u);
}

/* Accelerator compute-only latency, measured deterministically in the RTL
 * testbench (tb_top.v run_inference: START->done = 5216 core clocks). Same
 * 100 MHz clock domain as the Nios V core in nios_system, so cycle counts are
 * directly comparable. This is the pure inference cost, excluding the CPU's
 * byte-by-byte Avalon SRAM load (which is a CPU cost, not the accelerator's). */
#define HW_COMPUTE_CYCLES 5216u

/* Samples to benchmark. The software CNN has a data-independent control flow
 * (every conv position and tap is always evaluated), so its cycle count is
 * deterministic and one sample is representative. Keep this at 1 for RTL
 * simulation: one software inference already costs tens of ms of simulated
 * time, and Questa needs ~20 min of wall-clock per simulated 20 ms. */
#ifndef BENCH_SAMPLES
#define BENCH_SAMPLES 1
#endif

int main(void)
{
    int s, pred_hw, pred_sw, correct = 0, sw_ok = 0;
    unsigned c0, sw_cyc, sw_sum = 0;

    printf("\n=== ECG CNN: software (Nios V/m) vs accelerator ===\n");
    printf("ECG_BASE = 0x%08x\n", (unsigned)ECG_BASE);

    for (s = 0; s < BENCH_SAMPLES; s++) {
        /* --- Software INT8 CNN on the Nios V core, timed by mcycle --- */
        c0 = read_mcycle();
        pred_sw = cnn_sw_infer(ecg_data[s]);
        sw_cyc = read_mcycle() - c0;
        sw_sum += sw_cyc;

        /* --- Accelerator: drive over Avalon, verify same class --- */
        load_ecg(ecg_data[s]);
        pred_hw = run_inference();

        printf("sample %d : sw=%d hw=%d golden=%d  sw_cycles=%u  %s%s\n",
               s, pred_sw, pred_hw, ecg_golden[s], sw_cyc,
               (pred_hw == ecg_golden[s]) ? "HW_OK" : "HW_MISMATCH",
               (pred_sw == pred_hw) ? " SW==HW" : " SW!=HW");
        if (pred_hw == ecg_golden[s]) correct++;
        if (pred_sw == pred_hw)       sw_ok++;
    }

    unsigned sw_avg = sw_sum / BENCH_SAMPLES;
    printf("---------------------------------------------\n");
    printf("# ECG CNN - Software (Nios V/m)  ==== Executed cycles: %u\n", sw_avg);
    printf("# ECG CNN - Accelerator          ==== Executed cycles: %u\n",
           (unsigned)HW_COMPUTE_CYCLES);
    printf("# Speedup (SW/HW)                ==== %u.%02ux\n",
           sw_avg / HW_COMPUTE_CYCLES,
           (unsigned)(((unsigned long long)(sw_avg % HW_COMPUTE_CYCLES) * 100)
                      / HW_COMPUTE_CYCLES));
    printf("---------------------------------------------\n");
    printf("Result: HW %d/%d match golden ; SW==HW %d/%d\n",
           correct, BENCH_SAMPLES, sw_ok, BENCH_SAMPLES);

    while (1) { }   /* halt */
    return 0;
}
