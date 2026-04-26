import os, sys
sys.path.insert(0, '.')
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS, _lighten
setup_style()
import numpy as np
import matplotlib.pyplot as plt
import json

with open('figures/problem1_results.json') as f:
    p1 = json.load(f)

categories = ['SSIM', 'PSNR (dB)', '雅可比CV', '边界惩罚']
mona = [p1['fig3']['ssim'], p1['fig3']['psnr']/30, p1['fig3']['jacobian_cv'], p1['fig3']['boundary_penalty']]
cat  = [p1['fig4']['ssim'], p1['fig4']['psnr']/30, p1['fig4']['jacobian_cv'], p1['fig4']['boundary_penalty']]
# normalize PSNR back for display
mona_raw = [p1['fig3']['ssim'], p1['fig3']['psnr'], p1['fig3']['jacobian_cv'], p1['fig3']['boundary_penalty']]
cat_raw  = [p1['fig4']['ssim'], p1['fig4']['psnr'], p1['fig4']['jacobian_cv'], p1['fig4']['boundary_penalty']]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Left: SSIM & PSNR grouped bar
ax = axes[0]
metrics = ['SSIM', 'PSNR (dB)']
mona_vals = [p1['fig3']['ssim'], p1['fig3']['psnr']]
cat_vals  = [p1['fig4']['ssim'], p1['fig4']['psnr']]
x = np.arange(len(metrics))
w = 0.32
bars1 = ax.bar(x - w/2, mona_vals, w, color=_lighten(PALETTE[0], 0.4),
               edgecolor=PALETTE[0], linewidth=1.5, label='图3 蒙娜丽莎')
bars2 = ax.bar(x + w/2, cat_vals,  w, color=_lighten(PALETTE[1], 0.4),
               edgecolor=PALETTE[1], linewidth=1.5, label='图4 卡通猫')
for bar, v in zip(list(bars1)+list(bars2), mona_vals+cat_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{v:.2f}', ha='center', va='bottom', fontsize=8, color=COLORS['text'])
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylabel('指标值', fontsize=11)
ax.legend(fontsize=9, frameon=True, edgecolor=COLORS['grid'])
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.axhline(0, color=COLORS['ref_line'], linewidth=0.8, linestyle='--', alpha=0.5)

# Right: Jacobian CV & boundary penalty
ax2 = axes[1]
metrics2 = ['雅可比CV', '边界惩罚']
mona_v2 = [p1['fig3']['jacobian_cv'], p1['fig3']['boundary_penalty']]
cat_v2  = [p1['fig4']['jacobian_cv'], p1['fig4']['boundary_penalty']]
x2 = np.arange(len(metrics2))
bars3 = ax2.bar(x2 - w/2, mona_v2, w, color=_lighten(PALETTE[0], 0.4),
                edgecolor=PALETTE[0], linewidth=1.5, label='图3 蒙娜丽莎')
bars4 = ax2.bar(x2 + w/2, cat_v2,  w, color=_lighten(PALETTE[1], 0.4),
                edgecolor=PALETTE[1], linewidth=1.5, label='图4 卡通猫')
for bar, v in zip(list(bars3)+list(bars4), mona_v2+cat_v2):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f'{v:.3f}', ha='center', va='bottom', fontsize=8, color=COLORS['text'])
ax2.set_xticks(x2); ax2.set_xticklabels(metrics2, fontsize=10)
ax2.set_ylabel('指标值', fontsize=11)
ax2.legend(fontsize=9, frameon=True, edgecolor=COLORS['grid'])
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

for i, ax in enumerate(axes):
    ax.text(-0.08, 1.05, f'({chr(97+i)})', transform=ax.transAxes,
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['bg_box'],
                      edgecolor=COLORS['grid'], alpha=0.9))
fig.tight_layout(pad=2.0)
save_fig(fig, 'figures/fig_ssim_comparison.pdf')
