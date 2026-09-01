/*
 * cnn_sw.c - Pure-C INT8 ECG CNN inference for Nios V/m.
 *
 * Bit-exact software twin of the accelerator datapath (and of the Python
 * int8_forward_golden reference): for each conv layer
 *     acc(int32) = sum(x * w) over in_ch * tap  (+ pre-scaled bias)
 *     out = clamp( round_half_up(acc, nb), -127, 127 )   [+ ReLU on conv4]
 *     maxpool K=5 S=5
 * then GAP = floor(sum/4), FC (nb=0, pre-scaled bias), argmax.
 *
 * round_half_up(acc, n) = (acc + (1<<(n-1))) >> n  (arithmetic shift; RISC-V
 * srai gives floor for negatives, matching torch.floor((x+2^(n-1))/2^n)).
 *
 * Buffers are sized for the largest layer (in_len<=2500, out_ch<=8) and reused.
 */

#include "ecg_weights.h"

#define KSZ  5
#define PAD  2
#define POOL 5

/* Ping / pong activation buffers: [channel][length]. Conv1 input is 2500
 * (single channel); after pool1 -> 500, pool2 -> 100, pool3 -> 20, pool4 -> 4. */
static signed char buf_a[8][2500];
static signed char buf_b[8][2500];

static int clamp127(int v)
{
    if (v > 127)  return 127;
    if (v < -127) return -127;
    return v;
}

static int round_shift(int acc, int nb)
{
    if (nb <= 0) return acc;
    return (acc + (1 << (nb - 1))) >> nb;   /* arithmetic, floor for negatives */
}

/*
 * One conv+bias+rescale+(relu)+maxpool layer.
 *   in  : src[in_ch][in_len]
 *   out : dst[out_ch][in_len/POOL]
 *   w   : [out_ch][in_ch][KSZ] row-major ; bias : [out_ch] (acc domain)
 */
static void conv_pool(const signed char *w, const int *bias,
                      signed char src[][2500], signed char dst[][2500],
                      int in_ch, int out_ch, int in_len, int nb, int relu)
{
    int oc, t, ic, kk, p, j;
    int out_len = in_len / POOL;

    for (oc = 0; oc < out_ch; oc++) {
        const signed char *w_oc = w + oc * in_ch * KSZ;
        /* pooled output position p covers conv positions [p*POOL .. p*POOL+4] */
        for (p = 0; p < out_len; p++) {
            int pmax = -128;
            for (j = 0; j < POOL; j++) {
                t = p * POOL + j;                /* conv output index */
                int acc = bias[oc];
                for (ic = 0; ic < in_ch; ic++) {
                    const signed char *w_ic = w_oc + ic * KSZ;
                    for (kk = 0; kk < KSZ; kk++) {
                        int idx = t + kk - PAD;   /* padded conv1d, pad=2 */
                        if (idx >= 0 && idx < in_len)
                            acc += (int)src[ic][idx] * (int)w_ic[kk];
                    }
                }
                int o = clamp127(round_shift(acc, nb));
                if (relu && o < 0) o = 0;
                if (o > pmax) pmax = o;
            }
            dst[oc][p] = (signed char)pmax;
        }
    }
}

/*
 * Full INT8 inference on one 2500-sample INT8 ECG. Returns argmax class 0..3.
 * Bit-exact with int8_forward_golden / the RTL accelerator.
 */
int cnn_sw_infer(const signed char *ecg)
{
    int c, i, best, bestv;

    /* Load input into buf_a[0] (channel 0). */
    for (i = 0; i < 2500; i++)
        buf_a[0][i] = ecg[i];

    /* Conv1: 1x2500 -> 4x500 (no relu) */
    conv_pool(w_conv1, b_conv1, buf_a, buf_b, 1, 4, 2500, NB_C1, 0);
    /* Conv2: 4x500  -> 4x100 (no relu) */
    conv_pool(w_conv2, b_conv2, buf_b, buf_a, 4, 4, 500,  NB_C2, 0);
    /* Conv3: 4x100  -> 8x20  (no relu) */
    conv_pool(w_conv3, b_conv3, buf_a, buf_b, 4, 8, 100,  NB_C3, 0);
    /* Conv4: 8x20   -> 8x4   (ReLU) */
    conv_pool(w_conv4, b_conv4, buf_b, buf_a, 8, 8, 20,   NB_C4, 1);

    /* GAP: floor(sum/4) per channel -> gap[8] */
    int gap[8];
    for (c = 0; c < 8; c++) {
        int s = 0;
        for (i = 0; i < 4; i++) s += buf_a[c][i];
        gap[c] = s >> 2;                 /* floor(sum/4); sum>=0 after ReLU */
    }

    /* FC: logits[o] = sum(gap * w_fc) + b_fc[o] ; argmax */
    best = 0; bestv = -0x7fffffff;
    for (c = 0; c < 4; c++) {
        int acc = b_fc[c];
        for (i = 0; i < 8; i++)
            acc += gap[i] * (int)w_fc[c * 8 + i];
        if (acc > bestv) { bestv = acc; best = c; }
    }
    return best;
}
