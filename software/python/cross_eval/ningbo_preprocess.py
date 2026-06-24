"""Ningbo (Chapman-Ningbo / Shaoxing) Dataset — Preprocess for 4-class ECG.

Ningbo: ~45k 12-lead ECG records, 500 Hz, 10s (5000 samples), WFDB .hea/.mat.
Labels: SNOMED-CT codes in the `#Dx:` line of each .hea (multi-label).
Source: PhysioNet "A large scale 12-lead electrocardiogram database ...".

Pipeline (identical numeric path to ptbxl_preprocess.py / Chapman):
  1. Walk WFDBRecords/**/*.hea, read #Dx SNOMED codes
  2. Map to 4-class via dominant rhythm code (rhythm > morphology priority)
  3. Load waveform via wfdb, Lead II = channel 1
  4. Downsample 500 -> 250 Hz (5000 -> 2500) on float64
  5. Z-score normalize per record, cast float32 last
  6. Write <out>/<CLASS>/<recname>.npy  (by-class tree, no split)

4-class mapping (Chapman taxonomy, AFL grouped into AFIB):
Matches Chapman RHYTHM_TO_4CLASS (utils/dataset.py, Zheng et al. 2020):
  AFIB(0): AFIB,AF  GSVT(1): ST,SVT,AT,AVNRT,AVRT,SAAWR  SB(2): SB  SR(3): SR,SA
  AFIB (0): Atrial Fibrillation 164889003, Atrial Flutter 164890007 (=Chapman AF)
  GSVT (1): Sinus Tachycardia 427084000, Supraventricular Tach 426761007,
            AT 713422000, AVNRT 233896004, AVRT 233897008, SAAWR 195101003
  SB   (2): Sinus Bradycardia 426177001
  SR   (3): Sinus Rhythm 426783006, Sinus Irregularity 427393009 (=Chapman SA)

Usage:
    python cross_eval/ningbo_preprocess.py --data_dir d:\\Thesis101\\data
"""

import os
import glob
import argparse
import numpy as np
from scipy.signal import resample
from collections import Counter

# SNOMED rhythm code -> 4-class. AFL (164890007) -> AFIB per user decision.
SNOMED_TO_4CLASS = {
    '164889003': 0,   # Atrial Fibrillation
    '164890007': 0,   # Atrial Flutter -> AFIB
    '427084000': 1,   # Sinus Tachycardia        -> GSVT (Chapman 'ST')
    '426761007': 1,   # Supraventricular Tach.    -> GSVT (Chapman 'SVT')
    '713422000': 1,   # Atrial Tachycardia        -> GSVT (Chapman 'AT')
    '233896004': 1,   # AVNRT                     -> GSVT
    '233897008': 1,   # AVRT                      -> GSVT
    '195101003': 1,   # SA node wandering / SAAWR -> GSVT
    '426177001': 2,   # Sinus Bradycardia         -> SB
    '426783006': 3,   # Sinus Rhythm              -> SR
    '427393009': 3,   # Sinus Irregularity        -> SR (Chapman 'SA', NOT GSVT)
}

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']
ORIG_FS     = 500
TARGET_FS   = 250
TARGET_LEN  = TARGET_FS * 10   # 2500
LEAD_IDX    = 1                # Lead II


# The PhysioNet "large-scale 12-lead ECG" archive concatenates two source
# databases under one JS* numbering: Chapman-Shaoxing = JS00001..JS10646
# (= the 10,646 records the model was TRAINED on), Ningbo = JS10647 onward.
# We must exclude the Chapman half to avoid train/test leakage in the
# cross-dataset study. Boundary confirmed by label distribution (AFIB=1780,
# SVT=587 in the low half match Chapman; SI/AFL explosion in the high half).
CHAPMAN_MAX_JSID = 10646


def jsid(hea_path):
    """Numeric JS id from a record filename (JS00001.hea -> 1)."""
    return int(os.path.splitext(os.path.basename(hea_path))[0][2:])


def parse_dx(hea_path):
    """Return list of SNOMED codes from the #Dx: line, or [] if absent."""
    with open(hea_path) as f:
        for line in f:
            if line.startswith('#Dx:'):
                return [c.strip() for c in line.split(':', 1)[1].strip().split(',')]
    return []


def get_label(codes):
    """4-class from SNOMED code list, keeping only unambiguous records.

    A record is kept ONLY if every mappable rhythm code resolves to the SAME
    4-class (i.e. a single, clear rhythm). Records with no mappable rhythm code,
    or with conflicting rhythm classes (e.g. both AFIB and a GSVT code), are
    dropped (return None) — per the requirement to discard ambiguous labels."""
    classes = {SNOMED_TO_4CLASS[c] for c in codes if c in SNOMED_TO_4CLASS}
    if len(classes) != 1:
        return None
    return classes.pop()


def preprocess_signal(sig):
    """Downsample 500->250 Hz, z-score, float32 (2500,)."""
    down = resample(sig, TARGET_LEN)
    mu   = np.mean(down)
    std  = np.std(down) + 1e-8
    return ((down - mu) / std).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=r'd:\Thesis101\data')
    args = p.parse_args()

    ningbo_dir = os.path.join(args.data_dir, 'ningba')
    out_dir    = os.path.join(args.data_dir, 'ningba_by_class')
    for cname in CLASS_NAMES:
        os.makedirs(os.path.join(out_dir, cname), exist_ok=True)

    import wfdb

    heas = glob.glob(os.path.join(ningbo_dir, 'WFDBRecords', '**', '*.hea'),
                     recursive=True)
    print(f"[INFO] found {len(heas)} .hea records")

    dist = Counter()
    skipped_nolabel = 0
    skipped_err = 0
    skipped_chapman = 0

    for i, hea in enumerate(heas):
        if jsid(hea) <= CHAPMAN_MAX_JSID:     # Chapman half — model trained on it
            skipped_chapman += 1
            continue
        codes = parse_dx(hea)
        label = get_label(codes)
        if label is None:
            skipped_nolabel += 1
            continue
        try:
            rec  = wfdb.rdrecord(hea[:-4])           # strip .hea
            sig  = rec.p_signal[:, LEAD_IDX].astype(np.float64)
            proc = preprocess_signal(sig)
        except Exception:
            skipped_err += 1
            continue
        recname = os.path.splitext(os.path.basename(hea))[0]
        np.save(os.path.join(out_dir, CLASS_NAMES[label], recname + '.npy'), proc)
        dist[label] += 1
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1}/{len(heas)} processed")

    print(f"\n[INFO] written to {out_dir}")
    for c in range(4):
        print(f"  {CLASS_NAMES[c]}: {dist[c]}")
    print(f"  total labeled: {sum(dist.values())}")
    print(f"  skipped (Chapman half JS<={CHAPMAN_MAX_JSID}): {skipped_chapman}")
    print(f"  skipped (no rhythm label): {skipped_nolabel}")
    print(f"  skipped (load error):      {skipped_err}")


if __name__ == '__main__':
    main()
