import os, sys
sys.path.insert(0, '.')
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS
setup_style()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 计算雅可比行列式分布
R, x0, y0, H = 20, 105, 148, 80
phi, alpha = np.pi, np.radians(60)

theta_arr = np.linspace(0, 2*np.pi, 60)
h_arr = np.linspace(0, H, 40)
T, Hv = np.meshgrid(theta_arr, h_arr)

k = Hv * np.tan(alpha)
dxp_dtheta = -R*np.sin(T) + 2*k*np.sin(phi-T)*np.cos(T) - 2*k*np.cos(phi-T)*np.sin(T)
dyp_dtheta =  R*np.cos(T) + 2*k*np.sin(phi-T)*np.sin(T) + 2*k*np.cos(phi-T)*np.cos(T)
dxp_dh = np.tan(alpha)*(-np.cos(phi) + 2*np.cos(phi-T)*np.cos(T))
dyp_dh = np.tan(alpha)*(-np.sin(phi) + 2*np.cos(phi-T)*np.sin(T))
J = np.abs(dxp_dtheta*dyp_dh - dyp_dtheta*dxp_dh)

fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.imshow(J, aspect='auto', cmap='YlOrRd',
               extent=[0, 360, 0, H], origin='lower')
cs = ax.contour(np.degrees(theta_arr), h_arr, J, levels=8,
                colors='white', linewidths=0.5, alpha=0.5)
ax.clabel(cs, inline=True, fontsize=6.5, fmt='%.1f')

# 最大值标注
max_idx = np.unravel_index(J.argmax(), J.shape)
ax.scatter(np.degrees(theta_arr[max_idx[1]]), h_arr[max_idx[0]],
           s=120, marker='*', color=COLORS['down'], edgecolor='white', zorder=5)
ax.annotate(f'最大畸变\n$|J|={J.max():.1f}$',
            xy=(np.degrees(theta_arr[max_idx[1]]), h_arr[max_idx[0]]),
            xytext=(280, 60),
            fontsize=8, color=COLORS['down'],
            arrowprops=dict(arrowstyle='->', color=COLORS['down'], lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['down'], alpha=0.9))

cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('雅可比行列式 $|J_f|$', fontsize=10)
ax.set_xlabel('角度参数 $\theta$ (°)', fontsize=11)
ax.set_ylabel('高度参数 $h$ (mm)', fontsize=11)
ax.set_xticks([0, 90, 180, 270, 360])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
save_fig(fig, 'figures/fig_mapping_distortion.pdf')
