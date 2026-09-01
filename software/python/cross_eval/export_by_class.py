"""Export Chapman + PTB-XL records as per-class .npy files.

Each record -> data/<dataset>_by_class/<CLASS>/<id>.npy  (float32, shape (2500,))
Both datasets go through the SAME preprocessed pipeline:
  Lead II -> downsample 500->250 Hz -> per-record z-score -> float32.

  - Chapman: loaded on-the-fly via ChapmanECGDataset(split='all')
  - PTB-XL : read from the already-preprocessed ptbxl_dataset.npz (train+val+test)

Usage:
    python cross_eval/export_by_class.py --data_dir d:/Thesis101/data
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.dataset import ChapmanECGDataset

CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']   # index = label


def ensure_dirs(root):
    for c in CLASS_NAMES:
        os.makedirs(os.path.join(root, c), exist_ok=True)


def export_chapman(data_dir, out_root):
    print('=== Chapman ===')
    ensure_dirs(out_root)
    ds = ChapmanECGDataset(os.path.join(data_dir, 'Chapman'), split='all', seed=42)
    counts = {c: 0 for c in CLASS_NAMES}
    for i in range(len(ds)):
        sig = ds.records[i].astype(np.float32)        # (2500,) float32
        label = ds.labels[i]
        cname = CLASS_NAMES[label]
        np.save(os.path.join(out_root, cname, f'chapman_{i:05d}.npy'), sig)
        counts[cname] += 1
    print('  saved:', counts, '-> total', sum(counts.values()))


def export_ptbxl(data_dir, out_root):
    print('=== PTB-XL ===')
    ensure_dirs(out_root)
    npz = np.load(os.path.join(data_dir, 'ptbxl_processed', 'ptbxl_dataset.npz'))
    X = np.concatenate([npz['X_train'], npz['X_val'], npz['X_test']]).astype(np.float32)
    y = np.concatenate([npz['y_train'], npz['y_val'], npz['y_test']]).astype(int)
    counts = {c: 0 for c in CLASS_NAMES}
    for i in range(len(X)):
        cname = CLASS_NAMES[y[i]]
        np.save(os.path.join(out_root, cname, f'ptbxl_{i:05d}.npy'), X[i])
        counts[cname] += 1
    print('  saved:', counts, '-> total', sum(counts.values()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='d:/Thesis101/data')
    args = p.parse_args()

    export_chapman(args.data_dir, os.path.join(args.data_dir, 'chapman_by_class'))
    export_ptbxl(args.data_dir,   os.path.join(args.data_dir, 'ptbxl_by_class'))
    print('\nDone.')


if __name__ == '__main__':
    main()