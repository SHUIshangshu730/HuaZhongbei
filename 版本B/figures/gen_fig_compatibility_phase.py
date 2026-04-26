import os, sys
sys.path.insert(0, '.')
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS, _lighten
setup_style()
import numpy as np
import matplotlib.pyplot as plt
import json

with open('figures/problem3_results.json') as f:
    p3 = json.load(f)

grid = p3['feasibility_grid']
complexities = ['低频简单', '中等复杂', '高频复杂']
n = len(complexities)
Z = np.zeros((n, n))
for item in grid:
    i = complexities.index(item['mirror_complexity'])
    j = complexities.index(item['paper_complexity'])
    Z[i, j] = item['C_total']

fig, ax = plt.subplots(figsize=(7, 5.5))
x = np.linspace(0, 1, 80)
y = np.linspace(0, 1, 80)
X, Y = np.meshgrid(x, y)
# Smooth surface from grid data
from scipy.interpolate import RegularGridInterpolator
pts = np.array([0.15, 0.5, 0.85])
interp = RegularGridInterpolator((pts, pts), Z, method='linear', bounds_error=False, fill_value=None)
xi = np.column_stack([Y.ravel(), X.ravel()])
Zs = interp(xi).reshape(X.shape)

cf = ax.contourf(X, Y, Zs, levels=20, cmap='YlOrRd', alpha=0.85)
cs = ax.contour(X, Y, Zs, levels=[0.4, 0.7], colors=['white', 'white'],
                linewidths=[1.5, 2.0], linestyles=['--', '-'])
ax.clabel(cs, inline=True, fontsize=9, fmt={0.4: '勉强实现边界 0.4', 0.7: '可实现边界 0.7'})

# Region labels
ax.text(0.12, 0.12, '可实现区\n$\\mathcal{C}\\geq0.7$', ha='center', fontsize=10,
        color='white', fontweight='bold')
ax.text(0.45, 0.45, '勉强实现区\n$0.4\\leq\\mathcal{C}<0.7$', ha='center', fontsize=9,
        color=COLORS['text'], fontweight='bold')
ax.text(0.80, 0.80, '难以实现区\n$\\mathcal{C}<0.4$', ha='center', fontsize=9,
        color=COLORS['down'], fontweight='bold')

# Actual case points
ax.scatter(0.15, 0.15, s=100, marker='o', color=PALETTE[2], edgecolor='white',
           zorder=5, label='低频+低频 (0.61)')
ax.scatter(0.50, 0.15, s=100, marker='s', color=PALETTE[3], edgecolor='white',
           zorder=5, label='中频+低频 (0.45)')
ax.scatter(0.85, 0.85, s=100, marker='^', color=COLORS['down'], edgecolor='white',
           zorder=5, label='高频+高频 (0.34)')

cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
cbar.set_label('兼容性指标 $\\mathcal{C}$', fontsize=10)
ax.set_xlabel('纸面图案复杂度 $\\sigma_f(P^*)$', fontsize=11)
ax.set_ylabel('镜面图案复杂度 $\\sigma_f(M^*)$', fontsize=11)
ax.set_xticks([0.15, 0.5, 0.85])
ax.set_xticklabels(['低频简单', '中等复杂', '高频复杂'], fontsize=9)
ax.set_yticks([0.15, 0.5, 0.85])
ax.set_yticklabels(['低频简单', '中等复杂', '高频复杂'], fontsize=9)
ax.legend(fontsize=8, loc='upper left', frameon=True, edgecolor=COLORS['grid'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, 'figures/fig_compatibility_phase.pdf')
