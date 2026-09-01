/*
 * bench_c.c - Time the optimised C INT8 CNN on the host CPU.
 *
 * This is the fair software baseline: the same bit-exact INT8 arithmetic the
 * accelerator implements, compiled -O2 for the host, with no framework
 * overhead. Reading N samples of raw INT8 input from a file lets us reuse the
 * exact inputs the Python golden was generated from.
 *
 * Usage: bench_c <inp.bin> <pred.bin> <N>
 *   inp.bin  : N * 2500 int8 samples
 *   pred.bin : N int64 golden predicted classes
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern int cnn_sw_infer(const signed char *ecg);

/* Monotonic microsecond clock: QPC on Windows, CLOCK_MONOTONIC elsewhere. */
#ifdef _WIN32
#include <windows.h>
static double now_us(void)
{
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e6 / (double)freq.QuadPart;
}
#else
static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}
#endif

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: %s inp.bin pred.bin N\n", argv[0]); return 2; }
    int N = atoi(argv[3]);

    signed char *inp = malloc((size_t)N * 2500);
    long long   *gold = malloc((size_t)N * sizeof(long long));
    FILE *fi = fopen(argv[1], "rb"), *fp = fopen(argv[2], "rb");
    if (!fi || !fp) { fprintf(stderr, "cannot open inputs\n"); return 2; }
    if (fread(inp, 1, (size_t)N * 2500, fi) != (size_t)N * 2500) { fprintf(stderr, "short read inp\n"); return 2; }
    if (fread(gold, sizeof(long long), (size_t)N, fp) != (size_t)N) { fprintf(stderr, "short read pred\n"); return 2; }
    fclose(fi); fclose(fp);

    /* Warm up caches / branch predictors before timing. */
    for (int i = 0; i < (N < 20 ? N : 20); i++) cnn_sw_infer(inp + (size_t)i * 2500);

    double *us = malloc((size_t)N * sizeof(double));
    int mism = 0;
    for (int i = 0; i < N; i++) {
        double t0 = now_us();
        int pred = cnn_sw_infer(inp + (size_t)i * 2500);
        us[i] = now_us() - t0;
        if (pred != (int)gold[i]) {
            if (mism < 5) printf("MISMATCH i=%d c=%d gold=%lld\n", i, pred, gold[i]);
            mism++;
        }
    }

    double sum = 0, mn = us[0], mx = us[0];
    for (int i = 0; i < N; i++) { sum += us[i]; if (us[i] < mn) mn = us[i]; if (us[i] > mx) mx = us[i]; }
    qsort(us, N, sizeof(double), cmp_double);
    double med = us[N / 2], p95 = us[(int)(0.95 * (N - 1))];

    printf("samples=%d  mismatches=%d %s\n", N, mism, mism ? "FAIL" : "BIT-EXACT PASS");
    printf("latency_us median=%.1f mean=%.1f min=%.1f max=%.1f p95=%.1f\n",
           med, sum / N, mn, mx, p95);
    printf("throughput_inf_s=%.0f\n", 1e6 / med);
    return mism ? 1 : 0;
}
