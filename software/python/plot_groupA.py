"""Figures for the Chapter 4 group-A analyses.

  fig_ci        bootstrap 95% CI, float32 vs INT8 (ningba + georgia)
  fig_bitwidth  accuracy vs bit-width with the weight-ROM cost on a second panel
  fig_layererr  per-stage SQNR and the share of activations quantized to 0
  fig_hrerr     error rate vs heart-rate band, with the 60 bpm SB/SR threshold
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
OUT = os.path.join(R, 'figures')
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  {name}.png')


def fig_ci():
    ni = json.load(open(os.path.join(R, 'ningba', 'stats', 'ningba_stats.json')))
    ge = json.load(open(os.path.join(R, 'georgia', 'stats', 'georgia_stats.json')))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    metrics = [('acc', 'Accuracy'), ('f1_macro', 'F1-macro'), ('macro_auc', 'macro-AUC')]
    labels = ['Chapman-Ningbo\nfloat32', 'Chapman-Ningbo\nINT8',
              'Georgia\nfloat32', 'Georgia\nINT8']
    for ax, (key, title) in zip(axes, metrics):
        pts, los, his = [], [], []
        for src in (ni, ge):
            for mdl in ('float32', 'int8'):
                c = src['ci'][mdl][key]
                pts.append(c['point']); los.append(c['lo']); his.append(c['hi'])
        pts, los, his = map(np.array, (pts, los, his))
        x = np.arange(4)
        ax.errorbar(x, pts, yerr=[pts - los, his - pts], fmt='o',
                    capsize=5, color='C0', markersize=6)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    save(fig, 'fig_ci_bootstrap')


def fig_bitwidth():
    d = json.load(open(os.path.join(R, 'ningba', 'bitwidth', 'bitwidth_sweep.json')))
    rows = d['rows']
    names = [r['name'] for r in rows]
    acc = [r['acc'] * 100 for r in rows]
    rom = [r['hw']['weight_rom_bits'] / 1024 for r in rows]
    x = np.arange(len(rows))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.bar(x, acc, color=['C7', 'C3', 'C0', 'C2'], width=0.6)
    for i, v in enumerate(acc):
        a1.text(i, v + 1.5, f'{v:.2f}', ha='center', fontsize=8)
    a1.set_xticks(x); a1.set_xticklabels(names)
    a1.set_ylabel('Accuracy (%)'); a1.set_ylim(0, 105)
    a1.set_title('(a) Accuracy vs bit-width')
    a2.bar(x, rom, color=['C7', 'C3', 'C0', 'C2'], width=0.6)
    for i, v in enumerate(rom):
        a2.text(i, v + 0.4, f'{v:.1f}', ha='center', fontsize=8)
    a2.set_xticks(x); a2.set_xticklabels(names)
    a2.set_ylabel('Weight ROM (kbit)')
    a2.set_title('(b) Weight storage cost')
    save(fig, 'fig_bitwidth')


def fig_layererr():
    d = json.load(open(os.path.join(R, 'ningba', 'quant_error',
                                    'layer_quant_error.json')))
    a = d['activations']
    ks = ['pool1', 'pool2', 'pool3', 'pool4', 'gap', 'logits']
    lbl = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'GAP', 'FC']
    sq = [a[k]['sqnr_db'] for k in ks]
    nr = [a[k]['nrmse'] * 100 for k in ks]
    x = np.arange(len(ks))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.bar(x, sq, color='C0', width=0.6)
    for i, v in enumerate(sq):
        a1.text(i, v + 0.4, f'{v:.1f}', ha='center', fontsize=8)
    a1.set_xticks(x); a1.set_xticklabels(lbl)
    a1.set_ylabel('SQNR (dB)'); a1.set_title('(a) Signal-to-quantization-noise')
    a2.bar(x, nr, color='C1', width=0.6)
    for i, v in enumerate(nr):
        a2.text(i, v + 0.2, f'{v:.1f}', ha='center', fontsize=8)
    a2.set_xticks(x); a2.set_xticklabels(lbl)
    a2.set_ylabel('NRMSE (%)'); a2.set_title('(b) Relative error')
    save(fig, 'fig_layer_quant_error')


def fig_hrerr():
    d = json.load(open(os.path.join(R, 'ningba', 'rr_analysis', 'rr_analysis.json')))
    b = d['hr_bins']
    lbl = [f"{x['lo']}-{x['hi']}" for x in b]
    err = [x['err_pct'] for x in b]
    n = [x['n'] for x in b]
    x = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x, err, color='C0', width=0.65)
    for i, (v, c) in enumerate(zip(err, n)):
        ax.text(i, v + 0.3, f'{v:.1f}\n(n={c})', ha='center', fontsize=7)
    # 60 bpm SB/SR threshold sits between the 55-60 and 60-65 bins
    thr = next(i for i, x_ in enumerate(b) if x_['lo'] == 60) - 0.5
    ax.axvline(thr, color='C3', ls='--', lw=1.5)
    ax.text(thr + 0.1, max(err) * 0.92, 'SB/SR threshold\n60 bpm',
            color='C3', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(lbl, rotation=30, fontsize=8)
    ax.set_xlabel('Heart rate (bpm)'); ax.set_ylabel('Error rate (%)')
    ax.set_title('Error rate by heart-rate band (Chapman-Ningbo test, INT8)')
    save(fig, 'fig_hr_error')


if __name__ == '__main__':
    print('writing figures ->', OUT)
    fig_ci(); fig_bitwidth(); fig_layererr(); fig_hrerr()
