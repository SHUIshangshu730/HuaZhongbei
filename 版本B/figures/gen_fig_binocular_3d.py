"""Enhanced 3D binocular observation geometry diagram"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from _utils.plot_utils import setup_style, PALETTE, COLORS
    setup_style()
except:
    PALETTE = ['#7AAEC8','#E8945A','#7BC8A4','#9B8EC4','#E0A0A0','#F0C05A','#8FAEC0','#A8C4D8']
    COLORS = {'primary':'#7AAEC8','secondary':'#E8945A','accent':'#7BC8A4','down':'#E0A0A0'}

fig = plt.figure(figsize=(10, 7.5))
ax = fig.add_subplot(111, projection='3d')

# Cylinder parameters
R, H = 20, 80
x0, y0 = 105, 148

# Paper plane
px = np.linspace(50, 160, 10)
py = np.linspace(90, 210, 10)
PX, PY = np.meshgrid(px, py)
PZ = np.zeros_like(PX)
ax.plot_surface(PX, PY, PZ, alpha=0.12, color=PALETTE[2], edgecolor='none')

# Cylinder surface
theta = np.linspace(0, 2*np.pi, 60)
h_vals = np.linspace(0, H, 20)
T, Hv = np.meshgrid(theta, h_vals)
X_cyl = x0 + R*np.cos(T)
Y_cyl = y0 + R*np.sin(T)
Z_cyl = Hv
ax.plot_surface(X_cyl, Y_cyl, Z_cyl, cmap='coolwarm', alpha=0.45, edgecolor='none', rstride=2, cstride=2)

# Observer eyes (binocular)
d_IPD = 32
eye_z = 200
eye_y = 80
eye_x = x0
E_L = np.array([eye_x - d_IPD, eye_y, eye_z])
E_R = np.array([eye_x + d_IPD, eye_y, eye_z])
E_C = (E_L + E_R) / 2

ax.scatter(*E_L, s=120, color=PALETTE[1], marker='o', zorder=10, edgecolors='white', linewidth=0.8)
ax.scatter(*E_R, s=120, color=PALETTE[3], marker='o', zorder=10, edgecolors='white', linewidth=0.8)
ax.text(E_L[0]-8, E_L[1]-5, E_L[2]+8, r'$E_L$', fontsize=10, color=PALETTE[1], fontweight='bold')
ax.text(E_R[0]+3, E_R[1]-5, E_R[2]+8, r'$E_R$', fontsize=10, color=PALETTE[3], fontweight='bold')

# IPD line
ax.plot([E_L[0], E_R[0]], [E_L[1], E_R[1]], [E_L[2], E_R[2]], '-', color='gray', linewidth=1.2, alpha=0.6)
ax.text(eye_x, eye_y-8, eye_z+3, r'$d_{\mathrm{IPD}}$', fontsize=8, color='gray', ha='center')

# Reflection point on cylinder front
theta_M = np.pi
h_M = H * 0.5
M = np.array([x0 + R*np.cos(theta_M), y0 + R*np.sin(theta_M), h_M])
n_hat = np.array([np.cos(theta_M), np.sin(theta_M), 0])

ax.scatter(*M, s=80, color=PALETTE[0], marker='D', zorder=10, edgecolors='white', linewidth=0.8)
ax.text(M[0]-12, M[1], M[2]+5, r'$M(\theta,h)$', fontsize=8, color=PALETTE[0])

# Left eye reflection path
ax.plot([E_L[0], M[0]], [E_L[1], M[1]], [E_L[2], M[2]], '--', color=PALETTE[1], linewidth=1.5, alpha=0.8)
v_L = (E_L - M) / np.linalg.norm(E_L - M)
r_L = v_L - 2 * np.dot(v_L, n_hat) * n_hat
t_L = -M[2] / r_L[2] if r_L[2] != 0 else 100
P_L = M + t_L * r_L
ax.plot([M[0], P_L[0]], [M[1], P_L[1]], [M[2], 0], '-', color=PALETTE[1], linewidth=1.5, alpha=0.8)
ax.scatter(P_L[0], P_L[1], 0, s=60, color=PALETTE[1], marker='v', zorder=8)
ax.text(P_L[0]-5, P_L[1]-8, -3, r'$P_L$', fontsize=9, color=PALETTE[1], fontweight='bold')

# Right eye reflection path
ax.plot([E_R[0], M[0]], [E_R[1], M[1]], [E_R[2], M[2]], '--', color=PALETTE[3], linewidth=1.5, alpha=0.8)
v_R = (E_R - M) / np.linalg.norm(E_R - M)
r_R = v_R - 2 * np.dot(v_R, n_hat) * n_hat
t_R = -M[2] / r_R[2] if r_R[2] != 0 else 100
P_R = M + t_R * r_R
ax.plot([M[0], P_R[0]], [M[1], P_R[1]], [M[2], 0], '-', color=PALETTE[3], linewidth=1.5, alpha=0.8)
ax.scatter(P_R[0], P_R[1], 0, s=60, color=PALETTE[3], marker='v', zorder=8)
ax.text(P_R[0]+3, P_R[1]-8, -3, r'$P_R$', fontsize=9, color=PALETTE[3], fontweight='bold')

# Disparity annotation
ax.plot([P_L[0], P_R[0]], [P_L[1], P_R[1]], [2, 2], '-', color=PALETTE[4], linewidth=2, alpha=0.9)
mid_x = (P_L[0] + P_R[0]) / 2
mid_y = (P_L[1] + P_R[1]) / 2
ax.text(mid_x, mid_y+5, 6, r'$\Delta P$', fontsize=10, color=PALETTE[4], fontweight='bold', ha='center')

# Virtual image behind mirror
V_img = np.array([x0 + 2*R*np.cos(theta_M), y0 + 2*R*np.sin(theta_M), h_M])
ax.scatter(*V_img, s=100, color=PALETTE[5], marker='*', zorder=10)
ax.text(V_img[0]-15, V_img[1], V_img[2]+5, 'Virtual\nimage', fontsize=7, color=PALETTE[5], ha='center')
ax.plot([E_L[0], V_img[0]], [E_L[1], V_img[1]], [E_L[2], V_img[2]], ':', color=PALETTE[5], linewidth=0.8, alpha=0.5)
ax.plot([E_R[0], V_img[0]], [E_R[1], V_img[1]], [E_R[2], V_img[2]], ':', color=PALETTE[5], linewidth=0.8, alpha=0.5)

# Normal vector at M
n_end = M + 25 * n_hat
ax.plot([M[0], n_end[0]], [M[1], n_end[1]], [M[2], n_end[2]], '-', color='gray', linewidth=1.5, alpha=0.7)
ax.text(n_end[0]-5, n_end[1]-5, n_end[2]+3, r'$\mathbf{n}$', fontsize=9, color='gray')

# Second reflection point for depth variation
theta_M2 = np.pi * 0.75
h_M2 = H * 0.7
M2 = np.array([x0 + R*np.cos(theta_M2), y0 + R*np.sin(theta_M2), h_M2])
n_hat2 = np.array([np.cos(theta_M2), np.sin(theta_M2), 0])
ax.scatter(*M2, s=50, color=PALETTE[6], marker='D', zorder=9, alpha=0.7)

v_L2 = (E_L - M2) / np.linalg.norm(E_L - M2)
r_L2 = v_L2 - 2 * np.dot(v_L2, n_hat2) * n_hat2
t_L2 = -M2[2] / r_L2[2] if r_L2[2] != 0 else 100
P_L2 = M2 + t_L2 * r_L2
ax.plot([E_L[0], M2[0]], [E_L[1], M2[1]], [E_L[2], M2[2]], '--', color=PALETTE[1], linewidth=0.8, alpha=0.4)
ax.plot([M2[0], P_L2[0]], [M2[1], P_L2[1]], [M2[2], 0], '-', color=PALETTE[1], linewidth=0.8, alpha=0.4)

v_R2 = (E_R - M2) / np.linalg.norm(E_R - M2)
r_R2 = v_R2 - 2 * np.dot(v_R2, n_hat2) * n_hat2
t_R2 = -M2[2] / r_R2[2] if r_R2[2] != 0 else 100
P_R2 = M2 + t_R2 * r_R2
ax.plot([E_R[0], M2[0]], [E_R[1], M2[1]], [E_R[2], M2[2]], '--', color=PALETTE[3], linewidth=0.8, alpha=0.4)
ax.plot([M2[0], P_R2[0]], [M2[1], P_R2[1]], [M2[2], 0], '-', color=PALETTE[3], linewidth=0.8, alpha=0.4)

ax.set_xlabel('$x$ (mm)', fontsize=9, labelpad=8)
ax.set_ylabel('$y$ (mm)', fontsize=9, labelpad=8)
ax.set_zlabel('$z$ (mm)', fontsize=9, labelpad=8)
ax.view_init(elev=22, azim=215)
ax.tick_params(labelsize=7)
ax.set_xlim(50, 160)
ax.set_ylim(70, 220)
ax.set_zlim(-5, 210)

fig.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'fig_binocular_3d.pdf'), dpi=300, bbox_inches='tight')
print("OK: fig_binocular_3d.pdf generated")
