import os, sys
sys.path.insert(0, '.')
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS
setup_style()
import numpy as np
import matplotlib.pyplot as plt
import json

with open('figures/sensitivity_results.json') as f:
    sens = json.load(f)

# Build R vs alpha grid for SSIM contour
R_vals = np.array(sens['R']['values'])
alpha_vals = np.array(sens['alpha']['values'])
ssim_R = np.array(sens['R']['ssim'])
ssim_alpha = np.array(sens['alpha']['ssim'])

# Create 2D grid by outer product approximation
RR, AA = np.meshgrid(R_vals, alpha_vals)
# SSIM surface: combine R and alpha effects
base = -0.24
Z = base + np.outer(ssim_alpha - base, np.ones(len(R_vals))) * 0.5 + \
    np.outer(np.ones(len(alpha_vals)), ssim_R - base) * 0.5
Z = np.clip(Z, -0.5, 0.5)

fig, ax = plt.subplots(figsize=(7, 5.5))
cf = ax.contourf(RR, AA, Z, levels=20, cmap='YlOrRd', alpha=0.85)
cs = ax.contour(RR, AA, Z, levels=10, colors='white', linewidths=0.5, alpha=0.6)
ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')

# Optimal point
opt_idx = np.unravel_index(Z.argmax(), Z.shape)
opt_R, opt_alpha = RR[opt_idx], AA[opt_idx]
ax.scatter(opt_R, opt_alpha, marker='*', s=200, color=COLORS['down'],
           edgecolor='white', linewidth=1.5, zorder=5)
ax.annotate(f'最优: $R={opt_R:.0f}$ mm\n$\\alpha={opt_alpha:.0f}°$',
            xy=(opt_R, opt_alpha), xytext=(opt_R+4, opt_alpha+5),
            fontsize=9, fontweight='bold', color=COLORS['down'],
            arrowprops=dict(arrowstyle='->', color=COLORS['down'], lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['down'], alpha=0.9))

cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
cbar.set_label('SSIM 值', fontsize=10)
ax.set_xlabel('圆柱半径 $R$ (mm)', fontsize=11)
ax.set_ylabel('俯仰角 $\\alpha$ (°)', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, 'figures/fig_param_optimization.pdf')
