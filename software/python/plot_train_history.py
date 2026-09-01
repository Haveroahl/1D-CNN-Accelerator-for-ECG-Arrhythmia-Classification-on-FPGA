"""Plot training history from train_history.json.

Two-panel figure: (a) Loss History, (b) Accuracy History — train vs validation.
Matplotlib default style (C0/C1 colors, boxed axes, framed legend).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--history', default='./results/ningba/train_history.json')
    p.add_argument('--output_dir', default='./results/figures')
    p.add_argument('--tag', default='ningba')
    args = p.parse_args()

    with open(args.history) as f:
        h = json.load(f)

    ep = range(len(h['train_loss']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6), dpi=200)

    ax1.plot(ep, h['train_loss'], label='Train Loss')
    ax1.plot(ep, h['val_loss'], label='Val Loss')
    ax1.set_title('Loss History')
    ax1.legend()

    ax2.plot(ep, h['train_acc'], label='Train Acc')
    ax2.plot(ep, h['val_acc'], label='Val Acc')
    ax2.set_title('Accuracy History')
    ax2.set_ylim(top=1.0)
    ax2.legend()

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.join(args.output_dir, f'train_history_{args.tag}')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base}.{ext}', bbox_inches='tight')
    print(f'saved {base}.png / .pdf  ({len(h["train_loss"])} epochs; '
          f'final loss train {h["train_loss"][-1]:.4f} / val {h["val_loss"][-1]:.4f}; '
          f'final acc train {h["train_acc"][-1]*100:.2f}% / '
          f'val {h["val_acc"][-1]*100:.2f}%)')


if __name__ == '__main__':
    main()
