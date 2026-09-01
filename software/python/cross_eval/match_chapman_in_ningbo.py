"""Identify which Ningbo (JS*) records are actually Chapman-Shaoxing duplicates.

The PhysioNet CSN archive concatenates Chapman-Shaoxing + Ningbo under one JS
numbering. The Chapman half is the data the model was TRAINED on, so it must be
excluded from the cross-dataset test. We can't trust the JS<=10646 cut (it gives
10,247 != the true 10,646 Chapman records), so we match by WAVEFORM.

Method (scale-invariant, robust to uV/mV unit + tiny float diff):
  - For each record, take lead II, z-score normalise (mean0/std1), resample to a
    fixed 512-pt shape vector, round to 3 decimals -> fingerprint key.
  - A Ningbo record whose fingerprint matches ANY Chapman fingerprint is a
    Chapman duplicate -> record its JS id.
Outputs data/ningba_chapman_jsids.txt (one JS id per line) for the preprocessor.

Usage:  python cross_eval/match_chapman_in_ningbo.py
"""
import os, glob, numpy as np, wfdb
from scipy.signal import resample

DATA = r'd:\Thesis101\data'
N = 256  # fingerprint length


def zfp(sig):
    sig = np.asarray(sig, np.float64)
    s = sig.std()
    if s < 1e-9:
        return None
    z = (sig - sig.mean()) / s
    r = resample(z, N)
    return tuple(np.round(r, 2))


def main():
    # 1. Chapman fingerprints (lead II = column 1 of csv)
    chap = sorted(glob.glob(os.path.join(DATA, 'Chapman', 'MUSE_*.csv')))
    print(f'[INFO] hashing {len(chap)} Chapman records ...')
    chap_fp = {}
    for i, f in enumerate(chap):
        a = np.genfromtxt(f, delimiter=',', skip_header=1)
        fp = zfp(a[:, 1])
        if fp is not None:
            chap_fp[fp] = os.path.basename(f)
        if (i + 1) % 2000 == 0:
            print(f'   chapman {i+1}/{len(chap)}')
    print(f'[INFO] {len(chap_fp)} unique Chapman fingerprints')

    # 2. Scan Ningbo, flag matches
    heas = glob.glob(os.path.join(DATA, 'ningba', 'WFDBRecords', '**', '*.hea'),
                     recursive=True)
    print(f'[INFO] scanning {len(heas)} JS records ...')
    matched = []
    for i, h in enumerate(heas):
        try:
            r = wfdb.rdrecord(h[:-4])
            fp = zfp(r.p_signal[:, 1])
        except Exception:
            continue
        if fp is not None and fp in chap_fp:
            jsid = int(os.path.basename(h)[2:-4])
            matched.append(jsid)
        if (i + 1) % 5000 == 0:
            print(f'   ningbo {i+1}/{len(heas)}  matched={len(matched)}')

    matched.sort()
    out = os.path.join(DATA, 'ningba_chapman_jsids.txt')
    with open(out, 'w') as fo:
        fo.write('\n'.join(str(j) for j in matched) + '\n')
    print(f'\n[RESULT] Chapman duplicates found in Ningbo: {len(matched)}')
    if matched:
        print(f'         JS id range: {matched[0]} .. {matched[-1]}')
        print(f'         max JS id <= 10646? {matched[-1] <= 10646}')
    print(f'[INFO] saved: {out}')


if __name__ == '__main__':
    main()
