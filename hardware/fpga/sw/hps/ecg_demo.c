// ecg_demo.c
// ============================================================================
// HPS (Cortex-A9 / Linux) batch demo driver for the ECG CNN accelerator.
//
// Runs a whole test set through the accelerator, counts correct predictions
// vs. ground-truth labels, prints the accuracy on the console (UART), and
// reflects progress/result on the on-board HPS LED.
//
// This is the DEMO driver (batch + accuracy + LED). The single-sample driver
// ecg_classify.c is kept separate for one-shot classification.
//
// ── Accelerator register map (RTL/avalon_slave.v, addressUnits = WORDS) ──────
//   word 0x00 (byte 0x00) W : sram_din     [7:0]
//   word 0x01 (byte 0x04) W : sram_wr_addr [11:0]
//   word 0x02 (byte 0x08) W : sram_we      [0]
//   word 0x03 (byte 0x0C) W : start        [0]
//   word 0x04 (byte 0x10) R : {done_latched, busy}  bit0=busy, bit1=done
//   word 0x05 (byte 0x14) R : result       [1:0]
//
// ── Input files (from software/python/export_test_demo.py) ──────────────────
//   <set>_ecg_int8.bin : N * 2500 signed int8, sample-major (row i = sample i)
//   <set>_labels.bin   : N uint8 labels (0..3)
//   Expected accuracy: chapman_test ~94.65%, ptbxl_test ~77% (zero-shot).
//
// ── Displays ────────────────────────────────────────────────────────────────
//   7-SEGMENT (primary): accuracy % shown on HEX2/HEX1/HEX0 via a 7-bit Qsys PIO
//   driven into the seg7_acc.v module (RTL does BCD split + segment decode). HPS
//   writes the raw number 0..100; the displays show e.g. "094" or "100". Updated
//   live during the run and held at the final value. Set SEG7_PIO_BASE to the
//   PIO offset you assign in Qsys Address Map.
//
//   HPS LED (secondary, optional): single HPS_LED = GPIO53 = HPS GPIO1 bit 24,
//   used as a liveness/verdict indicator (toggles per sample; solid if pass).
//   Skipped automatically if the GPIO map fails.
//
// Build (cross-compile, ARMv7 Cortex-A9):
//   arm-linux-gnueabihf-gcc -O2 -Wall -o ecg_demo ecg_demo.c
// Run on the board (root for /dev/mem):
//   sudo ./ecg_demo chapman_test_ecg_int8.bin chapman_test_labels.bin
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <time.h>

// ── Accelerator bridge (lightweight HPS-to-FPGA) ────────────────────────────
#define LWH2F_BASE   0xFF200000UL
#define LWH2F_SPAN   0x00001000UL
#define AVS_BASE     0x00000000UL   // Qsys-assigned avs offset (see PHASE_D_STEPS §6)

#define REG_SRAM_DIN   (AVS_BASE + 0x00)
#define REG_SRAM_ADDR  (AVS_BASE + 0x04)
#define REG_SRAM_WE    (AVS_BASE + 0x08)
#define REG_START      (AVS_BASE + 0x0C)
#define REG_STATUS     (AVS_BASE + 0x10)
#define REG_RESULT     (AVS_BASE + 0x14)

// ── Accuracy 7-segment PIO (in the same lwh2f bridge, Qsys-assigned base) ────
// A 7-bit PIO output wired in Qsys to the seg7_acc module (HEX2/HEX1/HEX0).
// HPS writes the accuracy as a raw number 0..100; seg7_acc does the BCD + decode.
// ⚠️ Set SEG7_PIO_BASE to whatever offset you assign this PIO in Qsys Address Map.
#define SEG7_PIO_BASE  (AVS_BASE + 0x20)   // PIO data register (byte offset)

#define STATUS_DONE    0x2
#define ECG_LEN        2500
#define NUM_CLASSES    4
#define PASS_THRESHOLD 90.0        // % — solid LED if >= this

// ── HPS GPIO1 (Cyclone V HPS): HPS_LED = GPIO53 = GPIO1 bit 24 ───────────────
// HPS GPIO1 controller physical base. Registers (DesignWare GPIO):
//   gpio_swporta_dr  (0x00) data, gpio_swporta_ddr (0x04) direction (1=output).
#define HPS_GPIO1_BASE 0xFF709000UL
#define HPS_GPIO1_SPAN 0x00001000UL
#define GPIO_DR        0x00
#define GPIO_DDR       0x04
#define HPS_LED_BIT    (1u << 24)   // GPIO1[24] = GPIO53 = HPS_LED

static volatile uint8_t *g_bridge;   // accelerator avs window
static volatile uint8_t *g_gpio;     // HPS GPIO1 window (NULL if unavailable)

static inline void reg_wr(uint32_t off, uint32_t val) {
    *(volatile uint32_t *)(g_bridge + off) = val;
}
static inline uint32_t reg_rd(uint32_t off) {
    return *(volatile uint32_t *)(g_bridge + off);
}

// ── HPS LED helpers ─────────────────────────────────────────────────────────
static void led_init(void) {
    if (!g_gpio) return;
    uint32_t ddr = *(volatile uint32_t *)(g_gpio + GPIO_DDR);
    *(volatile uint32_t *)(g_gpio + GPIO_DDR) = ddr | HPS_LED_BIT;  // output
}
static void led_set(int on) {
    if (!g_gpio) return;
    uint32_t dr = *(volatile uint32_t *)(g_gpio + GPIO_DR);
    if (on) dr |= HPS_LED_BIT; else dr &= ~HPS_LED_BIT;
    *(volatile uint32_t *)(g_gpio + GPIO_DR) = dr;
}
static void led_toggle(void) {
    if (!g_gpio) return;
    uint32_t dr = *(volatile uint32_t *)(g_gpio + GPIO_DR);
    *(volatile uint32_t *)(g_gpio + GPIO_DR) = dr ^ HPS_LED_BIT;
}

// Write accuracy (0..100) to the 7-seg PIO; seg7_acc decodes to HEX2/1/0.
static void seg7_show(int acc_pct) {
    if (acc_pct < 0)   acc_pct = 0;
    if (acc_pct > 100) acc_pct = 100;
    reg_wr(SEG7_PIO_BASE, (uint32_t)acc_pct);
}

// ── Accelerator I/O ─────────────────────────────────────────────────────────
static void load_ecg(const int8_t *ecg) {
    for (uint32_t i = 0; i < ECG_LEN; i++) {
        reg_wr(REG_SRAM_DIN,  (uint8_t)ecg[i]);
        reg_wr(REG_SRAM_ADDR, i);
        reg_wr(REG_SRAM_WE,   1);   // self-clearing in avalon_slave
    }
}

static int run_inference(void) {
    reg_wr(REG_START, 1);
    uint32_t spins = 0;
    while (!(reg_rd(REG_STATUS) & STATUS_DONE)) {
        if (++spins > 100000000UL) {
            fprintf(stderr, "ERROR: inference timeout (done never asserted).\n");
            return -1;
        }
    }
    return (int)(reg_rd(REG_RESULT) & 0x3);
}

// ── Map a physical window; returns NULL on failure (non-fatal for GPIO) ──────
static volatile uint8_t *map_window(int fd, off_t base, size_t span) {
    void *m = mmap(NULL, span, PROT_READ | PROT_WRITE, MAP_SHARED, fd, base);
    return (m == MAP_FAILED) ? NULL : (volatile uint8_t *)m;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr,
            "usage: %s <set_ecg_int8.bin> <set_labels.bin>\n"
            "  e.g. %s chapman_test_ecg_int8.bin chapman_test_labels.bin\n",
            argv[0], argv[0]);
        return 1;
    }

    // ── Load labels (N = file size) ─────────────────────────────────────────
    FILE *fl = fopen(argv[2], "rb");
    if (!fl) { perror("fopen labels"); return 1; }
    fseek(fl, 0, SEEK_END);
    long N = ftell(fl);
    fseek(fl, 0, SEEK_SET);
    if (N <= 0) { fprintf(stderr, "ERROR: empty labels file\n"); fclose(fl); return 1; }
    uint8_t *labels = malloc((size_t)N);
    if (fread(labels, 1, (size_t)N, fl) != (size_t)N) {
        fprintf(stderr, "ERROR: short read on labels\n"); fclose(fl); free(labels); return 1;
    }
    fclose(fl);

    // ── Open ECG file (streamed sample-by-sample, no full load into RAM) ────
    FILE *fe = fopen(argv[1], "rb");
    if (!fe) { perror("fopen ecg"); free(labels); return 1; }
    fseek(fe, 0, SEEK_END);
    long ecg_bytes = ftell(fe);
    fseek(fe, 0, SEEK_SET);
    if (ecg_bytes != (long)N * ECG_LEN) {
        fprintf(stderr, "ERROR: ecg file = %ld bytes, expected N*%d = %ld\n",
                ecg_bytes, ECG_LEN, (long)N * ECG_LEN);
        fclose(fe); free(labels); return 1;
    }

    // ── mmap accelerator bridge (fatal) + HPS GPIO (optional) ───────────────
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem (need root)"); fclose(fe); free(labels); return 1; }

    g_bridge = map_window(fd, LWH2F_BASE, LWH2F_SPAN);
    if (!g_bridge) { perror("mmap bridge"); close(fd); fclose(fe); free(labels); return 1; }

    g_gpio = map_window(fd, HPS_GPIO1_BASE, HPS_GPIO1_SPAN);
    if (!g_gpio) fprintf(stderr, "[WARN] HPS GPIO map failed — LED disabled, console only.\n");
    led_init();
    led_set(0);

    // ── Batch loop ──────────────────────────────────────────────────────────
    int8_t ecg[ECG_LEN];
    long correct = 0, done = 0;
    long confusion[NUM_CLASSES][NUM_CLASSES] = {{0}};
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (long i = 0; i < N; i++) {
        if (fread(ecg, 1, ECG_LEN, fe) != ECG_LEN) {
            fprintf(stderr, "ERROR: short read on sample %ld\n", i); break;
        }
        load_ecg(ecg);
        int cls = run_inference();
        if (cls < 0) break;

        uint8_t y = labels[i];
        if (y < NUM_CLASSES) confusion[y][cls]++;
        if (cls == y) correct++;
        done++;

        led_toggle();   // liveness: one toggle per sample

        if ((i & 0x3F) == 0) {  // progress every 64 samples
            double racc = 100.0 * correct / done;
            seg7_show((int)(racc + 0.5));            // live running accuracy on HEX
            printf("\r  [%ld/%ld] running acc=%.2f%%   ", done, N, racc);
            fflush(stdout);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double acc = done ? 100.0 * correct / done : 0.0;
    double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    seg7_show((int)(acc + 0.5));   // final accuracy on HEX2/1/0

    printf("\n\n=== ECG accelerator demo ===\n");
    printf("samples : %ld\n", done);
    printf("correct : %ld\n", correct);
    printf("ACCURACY: %.2f%%\n", acc);
    if (done) printf("throughput (incl. HPS load): %.1f inf/s (%.2f ms/inf)\n",
                     done / secs, secs * 1000.0 / done);

    static const char *names[NUM_CLASSES] = {"AFIB", "GSVT", "SB", "SR"};
    printf("\nconfusion [true \\ pred]:   ");
    for (int p = 0; p < NUM_CLASSES; p++) printf("%6s", names[p]);
    printf("\n");
    for (int t = 0; t < NUM_CLASSES; t++) {
        printf("  %-6s              ", names[t]);
        for (int p = 0; p < NUM_CLASSES; p++) printf("%6ld", confusion[t][p]);
        printf("\n");
    }

    // ── Final LED verdict: solid if pass, slow-blink 10× if below threshold ─
    if (g_gpio) {
        if (acc >= PASS_THRESHOLD) {
            led_set(1);
            printf("\nLED: solid ON (accuracy >= %.0f%%)\n", PASS_THRESHOLD);
        } else {
            printf("\nLED: slow blink (accuracy < %.0f%%)\n", PASS_THRESHOLD);
            for (int k = 0; k < 20; k++) { led_toggle(); usleep(250000); }
            led_set(0);
        }
        munmap((void *)g_gpio, HPS_GPIO1_SPAN);
    }

    munmap((void *)g_bridge, LWH2F_SPAN);
    close(fd);
    fclose(fe);
    free(labels);
    return 0;
}
