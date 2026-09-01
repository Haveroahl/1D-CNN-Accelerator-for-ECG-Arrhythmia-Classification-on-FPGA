"""
Probe: does the model use RR-interval (rhythm timing) or morphology?
======================================================================
Two quantitative tests on the pruned float model (best_model_pruned.pth):

  1. HR correlation — correlate GAP features + per-class logit with the
     ground-truth VentricularRate (hr). If the model encodes heart rate,
     some feature dim should track hr.

  2. Beat-shuffle — detect R-peaks, segment into beats, randomly permute
     beat order (destroys RR-interval / rhythm regularity, keeps per-beat
     morphology). Measure accuracy drop. Small drop => model is a
     bag-of-morphology, not an RR detector.

Read-only: loads existing checkpoint + Chapman test split. No training.
"""
import sys, os
import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import ChapmanECGDataset, CLASS_NAMES
from prune_finetune import ECG_1DCNN_Pruned

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Chapman')
CKPT = os.path.join(os.path.dirname(__file__), 'results', 'best_model_pruned.pth')


def load_model():
    m = ECG_1DCNN_Pruned(c1_out=4, c2_out=4, c3_out=8, c4_out=8)
    sd = torch.load(CKPT, map_location=DEVICE)
    sd = sd.get('model_state_dict', sd.get('state_dict', sd)) if isinstance(sd, dict) else sd
    m.load_state_dict(sd)
    m.eval().to(DEVICE)
    return m


def get_gap_feature(model, x):
    """Run forward up to GAP, return (gap_feat (B,C), logits (B,4))."""
    with torch.no_grad():
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = model.pool1(model.conv1(x))
        h = model.pool2(model.conv2(h))
        h = model.pool3(model.conv3(h))
        h = model.pool4(torch.relu(model.conv4(h)))
        gap = model.gap(h).squeeze(-1)         # (B, C)
        logits = model.fc(gap)
    return gap.cpu().numpy(), logits.cpu().numpy()


def main():
    print(f"[INFO] device={DEVICE}  ckpt={os.path.basename(CKPT)}")
    ds = ChapmanECGDataset(DATA_DIR, split='test', seed=42)
    N = len(ds)
    print(f"[INFO] test records: {N}")

    X = np.stack([ds.records[i] for i in range(N)])          # (N, 2500) z-scored
    y = np.array(ds.labels)
    hr = np.array([h if h not in (None, 0) else np.nan for h in ds.heart_rates], dtype=float)

    model = load_model()
    xb = torch.from_numpy(X).float().to(DEVICE)

    # ---- baseline accuracy + features ----
    gap, logits = get_gap_feature(model, xb)
    pred = logits.argmax(1)
    base_acc = (pred == y).mean()
    print(f"\n=== Baseline (float pruned) ===")
    print(f"accuracy = {base_acc*100:.2f}%   (n={N})")

    # ================================================================
    # TEST 1 — HR correlation
    # ================================================================
    print(f"\n=== TEST 1: correlation with ground-truth VentricularRate ===")
    valid = ~np.isnan(hr)
    print(f"records with valid HR: {valid.sum()}/{N}")
    hrv = hr[valid]
    # correlate each GAP feature dim with HR
    print(f"\n{'feature':<14}{'pearson_r':>11}{'spearman':>11}")
    best = (0, '')
    for c in range(gap.shape[1]):
        r, _ = pearsonr(gap[valid, c], hrv)
        rho, _ = spearmanr(gap[valid, c], hrv)
        if abs(r) > abs(best[0]):
            best = (r, f'gap[{c}]')
        print(f"gap[{c}]{'':<8}{r:>11.3f}{rho:>11.3f}")
    # logits too
    for c in range(logits.shape[1]):
        r, _ = pearsonr(logits[valid, c], hrv)
        rho, _ = spearmanr(logits[valid, c], hrv)
        print(f"logit[{CLASS_NAMES[c]}]{'':<3}{r:>11.3f}{rho:>11.3f}")
    # best linear combo of GAP -> HR (R^2 of linear regression)
    A = np.concatenate([gap[valid], np.ones((valid.sum(), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(A, hrv, rcond=None)
    pred_hr = A @ coef
    ss_res = ((hrv - pred_hr) ** 2).sum()
    ss_tot = ((hrv - hrv.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"\nbest single feature: {best[1]} (|r|={abs(best[0]):.3f})")
    print(f"linear regression GAP(all dims) -> HR :  R^2 = {r2:.3f}")
    print(f"  (R^2 near 1 => HR is linearly decodable from features;")
    print(f"   R^2 near 0 => model barely encodes heart rate)")

    # ================================================================
    # TEST 2 — beat-shuffle (destroy RR-interval, keep morphology)
    # ================================================================
    print(f"\n=== TEST 2: beat-order shuffle (breaks RR-interval, keeps beat shape) ===")
    rng = np.random.default_rng(0)

    def shuffle_beats(sig):
        """Detect R-peaks, cut at midpoints, permute segment order, re-concat."""
        # R-peaks: prominence on rectified signal; 250 Hz, min RR ~0.3s = 75 samp
        peaks, _ = find_peaks(np.abs(sig), distance=60, prominence=np.std(sig) * 1.2)
        if len(peaks) < 3:
            return sig.copy(), 0
        # cut points = midpoints between consecutive peaks
        cuts = ((peaks[:-1] + peaks[1:]) // 2).tolist()
        bounds = [0] + cuts + [len(sig)]
        segs = [sig[bounds[i]:bounds[i+1]] for i in range(len(bounds) - 1)]
        order = rng.permutation(len(segs))
        out = np.concatenate([segs[i] for i in order])
        # pad/crop to original length
        if len(out) < len(sig):
            out = np.pad(out, (0, len(sig) - len(out)))
        return out[:len(sig)].astype(np.float32), len(segs)

    Xs = np.zeros_like(X)
    nseg = []
    for i in range(N):
        Xs[i], k = shuffle_beats(X[i])
        nseg.append(k)
    nseg = np.array(nseg)
    print(f"mean beats detected/record: {nseg.mean():.1f} (min {nseg.min()}, max {nseg.max()})")

    xs = torch.from_numpy(Xs).float().to(DEVICE)
    _, logits_s = get_gap_feature(model, xs)
    pred_s = logits_s.argmax(1)
    shuf_acc = (pred_s == y).mean()
    agree = (pred_s == pred).mean()
    print(f"\naccuracy after beat-shuffle = {shuf_acc*100:.2f}%   (baseline {base_acc*100:.2f}%)")
    print(f"drop = {(base_acc - shuf_acc)*100:.2f} pp")
    print(f"prediction agreement with original = {agree*100:.2f}%")

    # per-class breakdown (AFIB irregularity should be the one most hurt
    # if model used RR; SB/SR depend on rate)
    print(f"\nper-class accuracy  (baseline -> shuffled):")
    for c in range(4):
        m = y == c
        if m.sum() == 0:
            continue
        b = (pred[m] == c).mean()
        s = (pred_s[m] == c).mean()
        print(f"  {CLASS_NAMES[c]:<5} n={m.sum():4d}   {b*100:6.2f}% -> {s*100:6.2f}%   (d= {(b-s)*100:+.2f})")

    print(f"\n=== interpretation ===")
    print(f"Large accuracy drop => model relies on RR-interval / rhythm timing.")
    print(f"Small drop          => model is bag-of-morphology; RR barely used.")


if __name__ == '__main__':
    main()
