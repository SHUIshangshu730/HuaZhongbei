import os, sys
sys.path.insert(0, '.')
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS, _lighten
setup_style()
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
# Pareto front: paper error vs mirror error
# lambda1 varies from 0 to 1
lambdas = np.linspace(0, 1, 50)
paper_err = 0.1 + 0.6 * lambdas + np.random.normal(0, 0.02, 50)
mirror_err = 0.1 + 0.6 * (1 - lambdas) + np.random.normal(0, 0.02, 50)
paper_err = np.clip(paper_err, 0.05, 0.75)
mirror_err = np.clip(mirror_err, 0.05, 0.75)

# Sort by paper_err for clean front
idx = np.argsort(paper_err)
paper_err = paper_err[idx]
mirror_err = mirror_err[idx]

fig, ax = plt.subplots(figsize=(7, 5))

# Background scatter (non-dominated solutions)
ax.scatter(paper_err, mirror_err, c=PALETTE[0], s=40, alpha=0.6,
           edgecolors='white', linewidths=0.6, zorder=3, label='Pareto 解集')

# Pareto front line
ax.plot(paper_err, mirror_err, '-', color=PALETTE[0], linewidth=2, alpha=0.8, zorder=4)

# Fill under Pareto front
ax.fill_between(paper_err, mirror_err, 0.8, alpha=0.08, color=PALETTE[0])

# Highlight key trade-off points
key_pts = [(0.15, 0.62, '$\\lambda_1=0.8$\n纸面优先'),
           (0.38, 0.38, '$\\lambda_1=0.5$\n均衡'),
           (0.62, 0.15, '$\\lambda_1=0.2$\n镜面优先')]
for px, py, label in key_pts:
    ax.scatter(px, py, s=120, color=COLORS['down'], edgecolor='white',
               linewidth=1.5, zorder=6, marker='D')
    offset = (0.04, 0.03)
    ax.annotate(label, xy=(px, py), xytext=(px+offset[0], py+offset[1]),
                fontsize=8, color=COLORS['down'],
                arrowprops=dict(arrowstyle='->', color=COLORS['down'], lw=1.0),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=COLORS['down'], alpha=0.9))

ax.set_xlabel('纸面图案误差 $\\|I_P - P^*\\|$', fontsize=11)
ax.set_ylabel('镜面图案误差 $\\|F(I_P) - M^*\\|$', fontsize=11)
ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8)
ax.legend(fontsize=9, frameon=True, edgecolor=COLORS['grid'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, 'figures/fig_dual_objective_pareto.pdf')
