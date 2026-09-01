"""Vẽ hình minh hoạ cấu trúc confusion matrix 4x4 cho slide.

Sinh 2 file PNG (300 dpi, nền trắng, dán thẳng vào PowerPoint):
  cm_structure.png  — ma trận 4x4, ký hiệu n_ij, tô đậm đường chéo
  cm_tpfn.png       — phân rã TP/FN/FP/TN khi xét lớp AFIB

Chạy:  python plot_cm_theory.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'figures')
os.makedirs(OUT, exist_ok=True)

TEAL   = '#1B6B7A'
PANEL  = '#F1F0EC'
ORANGE = '#C4661A'
PULSE  = '#B02840'
PULSE_S= '#F7E4E8'
INK    = '#12161C'
GREY   = '#5C6470'
GREY2  = '#858C97'

CLASSES = ['AFIB', 'GSVT', 'SB', 'SR']
plt.rcParams['font.family'] = ['DejaVu Sans']


def draw_grid(ax, cell_face, cell_text, text_color, edge=None):
    """Vẽ lưới 4x4. cell_face/cell_text/text_color là mảng 4x4."""
    n = 4
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle(
                (j, n - 1 - i), 1, 1,
                facecolor=cell_face[i][j],
                edgecolor=(edge[i][j] if edge else 'white'),
                linewidth=(1.6 if edge and edge[i][j] != 'white' else 2.2),
                linestyle=('--' if edge and edge[i][j] == PULSE and i > 0 else '-'),
                zorder=1))
            ax.text(j + .5, n - 1 - i + .5, cell_text[i][j],
                    ha='center', va='center',
                    fontsize=15, fontweight='bold',
                    color=text_color[i][j], zorder=2)

    # nhãn cột (lớp dự đoán)
    for j, c in enumerate(CLASSES):
        ax.text(j + .5, n + .16, c, ha='center', va='bottom',
                fontsize=11.5, fontweight='bold', color=GREY)
    # nhãn hàng (lớp thật)
    for i, c in enumerate(CLASSES):
        ax.text(-.14, n - 1 - i + .5, c, ha='right', va='center',
                fontsize=11.5, fontweight='bold', color=GREY)

    ax.text(n / 2, n + .62, 'LỚP DỰ ĐOÁN', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=ORANGE)
    ax.text(-.95, n / 2, 'LỚP THẬT', ha='center', va='center',
            rotation=90, fontsize=11, fontweight='bold',
            color=ORANGE)

    ax.set_xlim(-1.15, n + .05)
    ax.set_ylim(-.05, n + .95)
    ax.set_aspect('equal')
    ax.axis('off')


# ══════════════ HÌNH 1 — cấu trúc chung ══════════════
fig, ax = plt.subplots(figsize=(7.0, 5.4), dpi=300)
fig.patch.set_facecolor('white')

face = [[TEAL if i == j else PANEL for j in range(4)] for i in range(4)]
txt  = [[f'n{chr(0x2080+i+1)}{chr(0x2080+j+1)}' for j in range(4)] for i in range(4)]
col  = [['white' if i == j else GREY for j in range(4)] for i in range(4)]

draw_grid(ax, face, txt, col)
ax.set_title('CẤU TRÚC CONFUSION MATRIX 4 × 4',
             fontsize=14, fontweight='bold', color=ORANGE, pad=26)
fig.text(.5, .035,
         'Phần tử $n_{ij}$ = số mẫu thuộc lớp thật $i$ được dự đoán là lớp $j$.\n'
         'Bốn ô trên đường chéo là số mẫu phân loại đúng. Tổng toàn ma trận = 4.973.',
         ha='center', fontsize=10, color=GREY, linespacing=1.5)
fig.subplots_adjust(top=.88, bottom=.16, left=.13, right=.97)
p1 = os.path.join(OUT, 'cm_structure.png')
fig.savefig(p1, facecolor='white', bbox_inches='tight')
plt.close(fig)


# ══════════════ HÌNH 2 — phân rã TP/FN/FP/TN ══════════════
fig, ax = plt.subplots(figsize=(7.0, 5.4), dpi=300)
fig.patch.set_facecolor('white')

face = [[PANEL] * 4 for _ in range(4)]
txt  = [['TN'] * 4 for _ in range(4)]
col  = [[GREY2] * 4 for _ in range(4)]
edge = [['white'] * 4 for _ in range(4)]

# hàng 0 = lớp thật AFIB
face[0][0], txt[0][0], col[0][0] = TEAL, 'TP', 'white'
for j in range(1, 4):
    face[0][j], txt[0][j], col[0][j], edge[0][j] = PULSE_S, 'FN', PULSE, PULSE
# cột 0 = dự đoán AFIB
for i in range(1, 4):
    face[i][0], txt[i][0], col[i][0], edge[i][0] = PULSE_S, 'FP', PULSE, PULSE

draw_grid(ax, face, txt, col, edge)
ax.set_title('PHÂN RÃ TP · FP · FN · TN — KHI XÉT LỚP AFIB',
             fontsize=13.5, fontweight='bold', color=ORANGE, pad=26)

fig.text(.5, .035,
         'TP: một ô duy nhất  ·  FN: phần còn lại của HÀNG (bỏ sót)\n'
         'FP: phần còn lại của CỘT (báo động sai)  ·  TN: chín ô còn lại\n'
         'Với bốn lớp, mỗi lớp có một bộ TP/FP/FN/TN riêng — không phải bốn ô cố định.',
         ha='center', fontsize=9.5, color=GREY, linespacing=1.55)
fig.subplots_adjust(top=.88, bottom=.19, left=.13, right=.97)
p2 = os.path.join(OUT, 'cm_tpfn.png')
fig.savefig(p2, facecolor='white', bbox_inches='tight')
plt.close(fig)

print('Đã lưu:')
print(' ', p1)
print(' ', p2)
