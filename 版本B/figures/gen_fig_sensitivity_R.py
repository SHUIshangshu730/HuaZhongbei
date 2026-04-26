import os, sys
sys.path.insert(0, '.')
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS, smart_labels, auto_legend
setup_style()
import numpy as np
import matplotlib.pyplot as plt
import json

with open('figures/sensitivity_results.json') as f:
    sens = json.load(f)

fig, ax = plt.subplots(figsize=(7, 4.5))

params_to_plot = [
    ('R', '圆柱半径 $R$ (mm)', 0),
    ('alpha', '俯仰角 $\\alpha$ (°)', 1),
    ('phi', '方位角 $\\phi$ (rad)', 2),
]

annot_xs, annot_ys, annot_texts, annot_colors = [], [], [], []
markers = ['o', 's', '^']

for i, (key, label, ci) in enumerate(params_to_plot):
    if key not in sens:
        continue
    x = np.array(sens[key]['values'])
    y = np.array(sens[key]['ssim'])
    # normalize x to [0,1] for unified axis
    x_norm = (x - x.min()) / (x.max() - x.min())
    ax.plot(x_norm, y, f'{markers[i]}-', color=PALETTE[ci], linewidth=2,
            markersize=6, markeredgecolor='white', markeredgewidth=1.2,
            label=label, zorder=3)
    ax.fill_between(x_norm, y.min()-0.02, y, alpha=0.07, color=PALETTE[ci], linewidth=0)
    max_idx = np.argmax(y)
    annot_xs.append(x_norm[max_idx]); annot_ys.append(y[max_idx])
    annot_texts.append(f'峰值 {y[max_idx]:.3f}')
    annot_colors.append(PALETTE[ci])

smart_labels(ax, annot_xs, annot_ys, annot_texts,
             colors=annot_colors, fontweight='bold', fontsize=8,
             offset=(0, 12),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, linewidth=0.5),
             arrowprops=dict(arrowstyle='->', lw=1.0, color=COLORS['ref_line']))

ax.set_xlabel('参数归一化值（0=最小，1=最大）', fontsize=11)
ax.set_ylabel('SSIM 值', fontsize=11)
auto_legend(ax)
ax.grid(alpha=0.15, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, 'figures/fig_sensitivity_R.pdf')
