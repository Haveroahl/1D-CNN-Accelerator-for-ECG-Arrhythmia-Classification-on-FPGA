"""
Pan-Tompkins R-peak / HR estimator — matching Liu 2023 (fphys-14-1079503).
==========================================================================
Implements exactly the HR estimator the reference paper fully-maps on FPGA,
so this doubles as the golden reference for a future hr_estimator.v.

Liu 2023 formulation (eqs 5-9):
  DIFF:  D[t] = x[t] - x[t-1]
  RECT:  R[t] = |D[t]|
  INTE:  S[t] = sum_{w=0..W} R[t+w]          (W = 16)
  THR  = 0.375 * max(S)  = (Smax>>2) + (Smax>>3)   (self-adaptive)
  R-peak where S[t] > THR, with refractory period 0.24 s (=60 samples @250Hz)
  BPM  = 60 * fs * (N-1) / sum(RR)  = 60*fs*(N-1)/(peak_last - peak_first)

Integer-friendly: no float except the input (which HW would have as INT).
"""
import numpy as np

FS = 250
W_INTE = 16
REFRACTORY_S = 0.24
GAMMA_SHIFTS = (2, 3)   # 0.375 = 1/4 + 1/8


def pan_tompkins_hr(x, fs=FS, w=W_INTE, refractory_s=REFRACTORY_S):
    """Return (bpm, peak_indices). x = 1-D ECG (any scale)."""
    # DIFF
    d = np.diff(x, prepend=x[0])
    # RECT
    r = np.abs(d)
    # INTE — sliding-window sum of width w (causal-ish, matches S[t]=sum R[t+w])
    kernel = np.ones(w + 1)
    s = np.convolve(r, kernel, mode='same')
    # self-adaptive threshold: 0.375 * max(S) via shifts (emulate HW)
    smax = s.max()
    thr = (smax / (1 << GAMMA_SHIFTS[0])) + (smax / (1 << GAMMA_SHIFTS[1]))
    # candidate peaks: S > THR, enforce refractory period
    refr = int(refractory_s * fs)
    above = s > thr
    peaks = []
    last = -refr - 1
    i = 0
    n = len(s)
    while i < n:
        if above[i]:
            # take local max within the contiguous above-threshold run
            j = i
            while j < n and above[j]:
                j += 1
            seg_peak = i + int(np.argmax(s[i:j]))
            if seg_peak - last >= refr:
                peaks.append(seg_peak)
                last = seg_peak
            i = j
        else:
            i += 1
    peaks = np.array(peaks)
    if len(peaks) < 2:
        return np.nan, peaks
    span = peaks[-1] - peaks[0]            # = sum of RR intervals (samples)
    bpm = 60.0 * fs * (len(peaks) - 1) / span
    return float(bpm), peaks


if __name__ == '__main__':
    # quick self-test on a synthetic 75 bpm signal
    t = np.arange(2500) / FS
    sig = np.zeros_like(t)
    for k in range(int(10 * 75 / 60)):
        c = int(k * FS * 60 / 75)
        if c < len(sig):
            sig[c] = 1.0
    bpm, pk = pan_tompkins_hr(sig)
    print(f"synthetic 75 bpm -> detected {bpm:.1f} bpm, {len(pk)} peaks")
