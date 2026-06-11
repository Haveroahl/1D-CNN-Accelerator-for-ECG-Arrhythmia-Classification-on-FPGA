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

int main(void)
{
    int s, pred, correct = 0;

    printf("\n=== ECG CNN accelerator on Nios V/m (Phase D) ===\n");
    printf("ECG_BASE = 0x%08x\n", (unsigned)ECG_BASE);

    for (s = 0; s < N_SAMPLES; s++) {
        load_ecg(ecg_data[s]);
        pred = run_inference();
        printf("sample %d : pred=%d golden=%d %s\n",
               s, pred, ecg_golden[s],
               (pred == ecg_golden[s]) ? "OK" : "MISMATCH");
        if (pred == ecg_golden[s])
            correct++;
    }

    printf("---------------------------------------------\n");
    printf("Result: %d/%d match golden\n", correct, N_SAMPLES);

    while (1) { }   /* halt */
    return 0;
}
