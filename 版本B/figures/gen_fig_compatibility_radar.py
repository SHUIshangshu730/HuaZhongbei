import os, sys
sys.path.insert(0, '.')
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS, _lighten
setup_style()
import numpy as np
import matplotlib.pyplot as plt
import json

with open('figures/problem3_results.json') as f:
    p3 = json.load(f)

categories = ['频谱兼容性\n$C_{\\mathrm{freq}}$', '对称性兼容性\n$C_{\\mathrm{sym}}$',
              '语义容错度\n$C_{\\mathrm{tol}}$', '主方向兼容性\n$C_{\\mathrm{dir}}$', '综合指标\n$\\mathcal{C}$']
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

# Data from problem3 results + constructed cases
methods = {
    '低频+低频': [0.72, 0.65, 0.70, 0.68, 0.61],
    '中频+低频': [0.55, 0.50, 0.55, 0.45, 0.45],
    '蒙娜丽莎+卡通猫': [
        p3['case_A_mona_cat']['C_freq'],
        p3['case_A_mona_cat']['C_sym'],
        p3['case_A_mona_cat']['C_tol'],
        p3['case_A_mona_cat']['C_dir'],
        p3['case_A_mona_cat']['C_total'],
    ],
}

fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
ax.set_facecolor('white')
ax.set_ylim(0, 1.12)

ring_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
theta_fill = np.linspace(0, 2*np.pi, 100)
for k, r in enumerate(ring_levels):
    r_prev = ring_levels[k-1] if k > 0 else 0
    if k % 2 == 0:
        ax.fill_between(theta_fill, r_prev, r, alpha=0.025, color=PALETTE[0], zorder=0)
    ax.plot(theta_fill, [r]*len(theta_fill), color='#E5E5E5', linewidth=0.5, zorder=1)
ax.set_yticks(ring_levels)
ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=7, color='#C0C0C0')
ax.spines['polar'].set_visible(False)
ax.xaxis.grid(True, color='#E5E5E5', linewidth=0.5)
ax.yaxis.grid(False)

method_list = list(methods.items())
# Draw non-primary methods first
for i, (name, vals) in enumerate(method_list):
    if i == 0: continue
    vc = vals + vals[:1]
    ax.plot(angles, vc, 'o--', linewidth=1.2, markersize=4, color=PALETTE[i],
            label=name, markeredgecolor='white', markeredgewidth=0.8, alpha=0.65, zorder=3)

# Draw primary method with fill
ours_name, ours_vals = method_list[0]
ours_vc = ours_vals + ours_vals[:1]
for layer in range(5, 0, -1):
    frac = layer / 5
    ax.fill(angles, [v*frac for v in ours_vc], alpha=0.04, color=PALETTE[0], zorder=4)
ax.fill(angles, ours_vc, alpha=0.12, color=_lighten(PALETTE[0], 0.3),
        edgecolor=PALETTE[0], linewidth=1.8, zorder=5)
ax.plot(angles, ours_vc, 'o-', linewidth=2.8, markersize=8, color=PALETTE[0],
        label=ours_name, markeredgecolor='white', markeredgewidth=2.0, zorder=10)

for j, (a, v) in enumerate(zip(angles[:-1], ours_vals)):
    r_label = v + 0.08
    ha = 'left' if (a < np.pi*0.25 or a > np.pi*1.75) else ('right' if np.pi*0.75 < a < np.pi*1.25 else 'center')
    va = 'bottom' if np.pi*0.25 < a < np.pi*0.75 else ('top' if np.pi*1.25 < a < np.pi*1.75 else 'center')
    ax.text(a, r_label, f'{v:.2f}', ha=ha, va=va, fontsize=8, fontweight='bold', color=PALETTE[0],
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=_lighten(PALETTE[0], 0.5), alpha=0.9, linewidth=0.6), zorder=11)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9, color=COLORS['text'])
ax.tick_params(axis='x', pad=18)
legend = ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.06),
                   frameon=True, edgecolor=COLORS['grid'], fontsize=9, facecolor='white')
legend.set_zorder(20)
fig.tight_layout(pad=1.5)
fig.savefig('figures/fig_compatibility_radar.pdf', dpi=300, bbox_inches='tight', pad_inches=0.3)
import matplotlib.pyplot as mpl_plt; mpl_plt.close(fig)
