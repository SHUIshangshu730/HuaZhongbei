import os, sys
sys.path.insert(0, '.')
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS
setup_style()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 5.5))
ax = fig.add_subplot(111, projection='3d')

# 圆柱参数
R, H = 20, 80
x0, y0 = 105, 148
theta = np.linspace(0, 2*np.pi, 60)
h_vals = np.linspace(0, H, 20)
T, Hv = np.meshgrid(theta, h_vals)
X_cyl = x0 + R*np.cos(T)
Y_cyl = y0 + R*np.sin(T)
Z_cyl = Hv

# 纸面（A4局部）
px = np.linspace(60, 150, 10)
py = np.linspace(100, 200, 10)
PX, PY = np.meshgrid(px, py)
PZ = np.zeros_like(PX)

ax.plot_surface(PX, PY, PZ, alpha=0.15, color=PALETTE[2], edgecolor='none')
ax.plot_surface(X_cyl, Y_cyl, Z_cyl, cmap='coolwarm', alpha=0.55, edgecolor='none',
                rstride=2, cstride=2)

# 反射光路示意（3条）
phi, alpha_rad = np.pi, np.radians(60)
for th in [np.pi*0.8, np.pi, np.pi*1.2]:
    Mx = x0 + R*np.cos(th)
    My = y0 + R*np.sin(th)
    Mz = H/2
    n = np.array([np.cos(th), np.sin(th), 0])
    v = np.array([np.sin(alpha_rad)*np.cos(phi), np.sin(alpha_rad)*np.sin(phi), np.cos(alpha_rad)])
    r = v - 2*np.dot(v, n)*n
    t = Mz / np.cos(alpha_rad)
    Px = Mx - t*(np.sin(alpha_rad)*np.cos(phi) - 2*np.sin(alpha_rad)*np.cos(phi-th)*np.cos(th))
    Py = My - t*(np.sin(alpha_rad)*np.sin(phi) - 2*np.sin(alpha_rad)*np.cos(phi-th)*np.sin(th))
    ax.plot([Mx, Px], [My, Py], [Mz, 0], '--', color=PALETTE[1], linewidth=1.2, alpha=0.8)
    ax.scatter([Mx], [My], [Mz], s=30, color=PALETTE[0], zorder=5)
    ax.scatter([Px], [Py], [0], s=20, color=PALETTE[3], zorder=5)

# 观察者方向箭头
ax.quiver(x0, y0-60, H+20, 0, 30, -20, color=COLORS['down'], linewidth=2, arrow_length_ratio=0.3)
ax.text(x0, y0-55, H+35, '观察者视线 $\mathbf{v}$', fontsize=9, color=COLORS['down'])

ax.set_xlabel('$x$ (mm)', fontsize=9, labelpad=5)
ax.set_ylabel('$y$ (mm)', fontsize=9, labelpad=5)
ax.set_zlabel('$z$ (mm)', fontsize=9, labelpad=5)
ax.view_init(elev=25, azim=210)
ax.tick_params(labelsize=7)
ax.set_xlim(60, 150); ax.set_ylim(80, 200); ax.set_zlim(0, H+30)
fig.tight_layout()
save_fig(fig, 'figures/fig_coordinate_system.pdf')
