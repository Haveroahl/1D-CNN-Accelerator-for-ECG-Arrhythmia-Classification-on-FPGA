"""Vẽ hình so sánh mô hình trước và sau cắt tỉa (và sau lượng tử hoá).

Số liệu lấy từ Bảng 4.1 và Bảng 3.2 của khoá luận.

Sinh 2 file PNG 300 dpi nền trắng:
  pruning_accuracy.png  — cột kép: độ chính xác + số tham số, kèm mức chênh
  pruning_channels.png  — số kênh từng tầng trước/sau cắt tỉa

Chạy:  python plot_pruning_compare.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'figures')
os.makedirs(OUT, exist_ok=True)

TEAL   = '#1B6B7A'
TEAL_L = '#7FB3BE'
ORANGE = '#C4661A'
PULSE  = '#B02840'
GREY   = '#5C6470'
GREY2  = '#858C97'
PANEL  = '#F1F0EC'
RULE   = '#DCDEE2'

plt.rcParams['font.family'] = ['DejaVu Sans']

# ── Bảng 4.1 ────────────────────────────────────────────────────────────
NAMES  = ['Float32\nđầy đủ', 'Float32\nsau cắt tỉa', 'INT8\nkhớp bit']
PARAMS = [1244, 640, 640]
ACC    = [95.35, 95.03, 94.27]
F1     = [0.9478, 0.9446, 0.9356]


# ══════════════ HÌNH 1 — độ chính xác + tham số ══════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.0), dpi=300,
                               gridspec_kw={'width_ratios': [1.15, 1]})
fig.patch.set_facecolor('white')

# ---- trái: độ chính xác ----
x = np.arange(3)
colors = [TEAL, TEAL, ORANGE]
bars = ax1.bar(x, ACC, width=.56, color=colors, zorder=3)

for xi, v in zip(x, ACC):
    ax1.text(xi, v + .07, f'{v:.2f}%', ha='center', va='bottom',
             fontsize=13, fontweight='bold', color=GREY if xi < 2 else ORANGE)

# mũi tên chênh lệch
ax1.annotate('', xy=(1, 95.03), xytext=(0, 95.35),
             arrowprops=dict(arrowstyle='->', color=PULSE, lw=1.6,
                             shrinkA=3, shrinkB=3, connectionstyle='arc3,rad=-.25'))
ax1.text(.5, 94.62, '−0,32 điểm', ha='center', va='center', fontsize=11,
         fontweight='bold', color=PULSE,
         bbox=dict(boxstyle='round,pad=.32', fc='white', ec=PULSE, lw=.9))

ax1.annotate('', xy=(2, 94.27), xytext=(1, 95.03),
             arrowprops=dict(arrowstyle='->', color=GREY2, lw=1.6,
                             shrinkA=3, shrinkB=14, connectionstyle='arc3,rad=-.32'))
ax1.text(1.62, 93.70, '−0,76 điểm', ha='center', va='center', fontsize=11,
         fontweight='bold', color=GREY,
         bbox=dict(boxstyle='round,pad=.32', fc='white', ec=GREY2, lw=.9))

ax1.set_xticks(x)
ax1.set_xticklabels(NAMES, fontsize=11.5, color=GREY)
ax1.set_ylim(93.35, 95.95)
ax1.set_ylabel('Độ chính xác (%)', fontsize=11.5, color=GREY)
ax1.set_title('Độ chính xác qua từng bước',
              fontsize=13, fontweight='bold', color=GREY, pad=14)
ax1.grid(axis='y', color=RULE, lw=.8, zorder=0)
ax1.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    ax1.spines[s].set_visible(False)
ax1.spines['bottom'].set_color(RULE)
ax1.tick_params(axis='y', labelsize=10, colors=GREY2, length=0)
ax1.tick_params(axis='x', length=0)

# ---- phải: số tham số ----
bars2 = ax2.bar(x, PARAMS, width=.56, color=[TEAL, ORANGE, ORANGE], zorder=3)
for xi, v in zip(x, PARAMS):
    ax2.text(xi, v + 22, f'{v:,}'.replace(',', '.'), ha='center', va='bottom',
             fontsize=13, fontweight='bold', color=GREY if xi == 0 else ORANGE)

ax2.annotate('', xy=(1, 640), xytext=(0, 1244),
             arrowprops=dict(arrowstyle='->', color=PULSE, lw=1.8,
                             shrinkA=6, shrinkB=6, connectionstyle='arc3,rad=.30'))
ax2.text(.30, 905, '−48,55 %', ha='center', va='center', fontsize=12,
         fontweight='bold', color=PULSE,
         bbox=dict(boxstyle='round,pad=.34', fc='white', ec=PULSE, lw=1.1))

ax2.set_xticks(x)
ax2.set_xticklabels(NAMES, fontsize=11.5, color=GREY)
ax2.set_ylim(0, 1480)
ax2.set_ylabel('Số tham số', fontsize=11.5, color=GREY)
ax2.set_title('Số tham số của mô hình',
              fontsize=13, fontweight='bold', color=GREY, pad=14)
ax2.grid(axis='y', color=RULE, lw=.8, zorder=0)
ax2.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    ax2.spines[s].set_visible(False)
ax2.spines['bottom'].set_color(RULE)
ax2.tick_params(axis='y', labelsize=10, colors=GREY2, length=0)
ax2.tick_params(axis='x', length=0)

fig.suptitle('SO SÁNH MÔ HÌNH TRƯỚC VÀ SAU CẮT TỈA KÊNH',
             fontsize=15, fontweight='bold', color=ORANGE, y=.985)
fig.text(.5, .022,
         'Cắt tỉa loại gần một nửa số tham số nhưng chỉ mất 0,32 điểm phần trăm độ chính xác.',
         ha='center', fontsize=10.5, color=GREY)
fig.subplots_adjust(top=.82, bottom=.17, wspace=.28, left=.075, right=.975)

p1 = os.path.join(OUT, 'pruning_accuracy.png')
fig.savefig(p1, facecolor='white', bbox_inches='tight')
plt.close(fig)


# ══════════════ HÌNH 2 — số kênh từng tầng ══════════════
LAYERS = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'FC']
BEFORE = [4, 8, 8, 16, 16]
AFTER  = [4, 4, 8, 8, 8]

fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=300)
fig.patch.set_facecolor('white')

xx = np.arange(len(LAYERS))
w = .35
b1 = ax.bar(xx - w/2, BEFORE, w, label='Trước cắt tỉa', color=TEAL_L, zorder=3)
b2 = ax.bar(xx + w/2, AFTER,  w, label='Sau cắt tỉa',  color=TEAL,   zorder=3)

for xi, (a, b) in enumerate(zip(BEFORE, AFTER)):
    ax.text(xi - w/2, a + .28, str(a), ha='center', va='bottom',
            fontsize=11.5, fontweight='bold', color=GREY)
    ax.text(xi + w/2, b + .28, str(b), ha='center', va='bottom',
            fontsize=11.5, fontweight='bold', color=TEAL)
    if a != b:
        ax.text(xi, -1.55, f'−{a-b}', ha='center', va='center', fontsize=11,
                fontweight='bold', color=PULSE)

# đường ngưỡng 8 đơn vị tính
ax.axhline(8, color=ORANGE, lw=1.4, ls='--', zorder=2)
ax.text(-.34, 8.55, 'tám đơn vị tính của mạch', ha='left', va='bottom',
        fontsize=10, fontweight='bold', color=ORANGE,
        bbox=dict(boxstyle='round,pad=.22', fc='white', ec='none'))

ax.set_xticks(xx)
ax.set_xticklabels(LAYERS, fontsize=12, color=GREY)
ax.set_ylim(-2.6, 19.6)
ax.set_ylabel('Số kênh', fontsize=11.5, color=GREY)
ax.set_title('SỐ KÊNH MỖI TẦNG TRƯỚC VÀ SAU CẮT TỈA',
             fontsize=14, fontweight='bold', color=ORANGE, pad=16)
ax.grid(axis='y', color=RULE, lw=.8, zorder=0)
ax.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_position(('data', 0))
ax.spines['bottom'].set_color(RULE)
ax.tick_params(axis='y', labelsize=10, colors=GREY2, length=0)
ax.tick_params(axis='x', length=0, pad=24)
ax.set_yticks([0, 4, 8, 12, 16])
ax.legend(fontsize=11, frameon=False, loc='upper left',
          labelcolor=GREY, bbox_to_anchor=(.005, .995), ncol=2,
          columnspacing=1.6, handlelength=1.3)

fig.text(.5, .015,
         'Cấu hình (4, 8, 8, 16) → (4, 4, 8, 8): mọi tầng đều về luỹ thừa của hai '
         'và không vượt quá tám đơn vị tính của mạch.',
         ha='center', fontsize=10, color=GREY)
fig.subplots_adjust(top=.86, bottom=.17, left=.09, right=.97)

p2 = os.path.join(OUT, 'pruning_channels.png')
fig.savefig(p2, facecolor='white', bbox_inches='tight')
plt.close(fig)

print('Đã lưu:')
print(' ', p1)
print(' ', p2)
