// ecg_classify.c
// ============================================================================
// HPS (Cortex-A9 / Linux) user-space driver for the ECG CNN accelerator.
//
// Talks to the accelerator's avalon_slave over the HPS lightweight HPS-to-FPGA
// bridge (lwh2f, physical base 0xFF20_0000). The accelerator's avs base offset
// inside that bridge is whatever you assigned in Qsys Address Map (see README
// "Bước 6"). If you assigned 0x0, BRIDGE_BASE below is correct as-is.
//
// Register map (from RTL/avalon_slave.v). avalon_slave declares addressUnits =
// WORDS, so Avalon word address N == byte offset N*4 from the slave base:
//   word 0x00 (byte 0x00)  W : sram_din      [7:0]   one ECG byte to load
//   word 0x01 (byte 0x04)  W : sram_wr_addr  [11:0]  target index in input_sram
//   word 0x02 (byte 0x08)  W : sram_we       [0]     write-enable strobe
//   word 0x03 (byte 0x0C)  W : start         [0]     1 = kick off inference
//   word 0x04 (byte 0x10)  R : {done_latched, busy}  bit0=busy, bit1=done
//   word 0x05 (byte 0x14)  R : result        [1:0]   class 0..3
//
// Load protocol per ECG sample (2500 INT8 values): for each index i,
//   write sram_din = ecg[i]; write sram_wr_addr = i; pulse sram_we = 1.
// sram_we is self-clearing in avalon_slave (it deasserts the next cycle), so we
// write 1 each time; input_sram latches on the we=1 cycle.
//
// Build (cross-compile for ARMv7 Cortex-A9):
//   arm-linux-gnueabihf-gcc -O2 -Wall -o ecg_classify ecg_classify.c
// Run on the board (needs /dev/mem → root):
//   sudo ./ecg_classify sample0.bin
// where sample0.bin is 2500 raw int8 bytes (one ECG window).
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>

// ── Address map ─────────────────────────────────────────────────────────────
#define LWH2F_BASE   0xFF200000UL   // lightweight HPS-to-FPGA bridge base
#define LWH2F_SPAN   0x00001000UL   // map 4 KB (plenty for 6 registers)
#define AVS_BASE     0x00000000UL   // avs offset inside the bridge (Qsys-assigned)

// Register byte offsets (word address * 4)
#define REG_SRAM_DIN   (AVS_BASE + 0x00)
#define REG_SRAM_ADDR  (AVS_BASE + 0x04)
#define REG_SRAM_WE    (AVS_BASE + 0x08)
#define REG_START      (AVS_BASE + 0x0C)
#define REG_STATUS     (AVS_BASE + 0x10)   // bit0 busy, bit1 done
#define REG_RESULT     (AVS_BASE + 0x14)

#define ECG_LEN        2500

#define STATUS_BUSY    0x1
#define STATUS_DONE    0x2

static volatile uint8_t *g_bridge;   // mmap'd lwh2f window

static inline void reg_wr(uint32_t off, uint32_t val) {
    *(volatile uint32_t *)(g_bridge + off) = val;
}
static inline uint32_t reg_rd(uint32_t off) {
    return *(volatile uint32_t *)(g_bridge + off);
}

// Load one ECG byte into input_sram at index idx.
static void load_byte(uint32_t idx, uint8_t v) {
    reg_wr(REG_SRAM_DIN,  v);
    reg_wr(REG_SRAM_ADDR, idx);
    reg_wr(REG_SRAM_WE,   1);    // avalon_slave self-clears we next cycle
}

// Load full 2500-sample ECG window.
static void load_ecg(const int8_t *ecg) {
    for (uint32_t i = 0; i < ECG_LEN; i++)
        load_byte(i, (uint8_t)ecg[i]);
}

// Kick inference and block until done. Returns class 0..3.
static int run_inference(void) {
    reg_wr(REG_START, 1);                       // start pulse (self-clears in RTL)

    // Poll done. Inference is ~5216 cycles @100MHz ≈ 52 µs, so this spins briefly.
    // Guard with a generous timeout so a wiring/clock fault doesn't hang forever.
    uint32_t spins = 0;
    while (!(reg_rd(REG_STATUS) & STATUS_DONE)) {
        if (++spins > 100000000UL) {            // ~conservative escape hatch
            fprintf(stderr, "ERROR: inference timeout (done never asserted). "
                            "Check clock/reset/bridge wiring.\n");
            return -1;
        }
    }
    return (int)(reg_rd(REG_RESULT) & 0x3);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <ecg_sample.bin (2500 raw int8 bytes)>\n", argv[0]);
        return 1;
    }

    // ── Read the ECG sample file (exactly 2500 int8 bytes) ──────────────────
    int8_t ecg[ECG_LEN];
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }
    size_t n = fread(ecg, 1, ECG_LEN, f);
    fclose(f);
    if (n != ECG_LEN) {
        fprintf(stderr, "ERROR: expected %d bytes, read %zu\n", ECG_LEN, n);
        return 1;
    }

    // ── mmap the lightweight bridge ─────────────────────────────────────────
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (need root)"); return 1; }

    void *map = mmap(NULL, LWH2F_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd, LWH2F_BASE);
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
    g_bridge = (volatile uint8_t *)map;

    // ── Load → infer → read ─────────────────────────────────────────────────
    load_ecg(ecg);
    int cls = run_inference();

    if (cls >= 0) {
        static const char *names[4] = {"AFIB", "GSVT", "SB", "SR"};
        printf("class = %d (%s)\n", cls, names[cls]);
    }

    munmap(map, LWH2F_SPAN);
    close(fd);
    return (cls >= 0) ? 0 : 2;
}
