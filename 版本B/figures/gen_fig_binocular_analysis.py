"""Binocular disparity and comfort zone analysis figure"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from _utils.plot_utils import setup_style, PALETTE
    setup_style()
except:
    PALETTE = ['#7AAEC8','#E8945A','#7BC8A4','#9B8EC4','#E0A0A0','#F0C05A','#8FAEC0','#A8C4D8']

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# ===== Panel (a): Binocular observation geometry =====
ax = axes[0]
ax.set_xlim(-80, 80)
ax.set_ylim(-20, 160)
ax.set_aspect('equal')

# Paper plane
ax.plot([-70, 70], [0, 0], 'k-', linewidth=2)
ax.text(0, -12, 'Paper plane (z=0)', ha='center', fontsize=8, style='italic')

# Cylinder mirror
theta_cyl = np.linspace(0, 2*np.pi, 100)
R_cyl = 15
ax.plot(R_cyl*np.cos(theta_cyl), R_cyl*np.sin(theta_cyl),
        color=PALETTE[0], linewidth=2)
ax.fill(R_cyl*np.cos(theta_cyl), R_cyl*np.sin(theta_cyl),
        color=PALETTE[0], alpha=0.15)
ax.text(0, 0, 'R', ha='center', va='center', fontsize=9, fontweight='bold')

# Left and right eyes
d_eye = 32
eye_h = 140
eye_L = (-d_eye, eye_h)
eye_R = (d_eye, eye_h)

ax.plot(*eye_L, 'o', color=PALETTE[1], markersize=10, zorder=5)
ax.plot(*eye_R, 'o', color=PALETTE[3], markersize=10, zorder=5)
ax.text(eye_L[0]-8, eye_L[1]+5, r'$E_L$', fontsize=9, color=PALETTE[1], fontweight='bold')
ax.text(eye_R[0]+3, eye_R[1]+5, r'$E_R$', fontsize=9, color=PALETTE[3], fontweight='bold')

# IPD annotation
ax.annotate('', xy=(d_eye, eye_h-8), xytext=(-d_eye, eye_h-8),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
ax.text(0, eye_h-14, r'$d_{IPD}$', ha='center', fontsize=8, color='gray')

# Reflection points
refl_L = (-10, 12)
refl_R = (10, 12)
ax.plot(*refl_L, 's', color=PALETTE[1], markersize=6, zorder=5)
ax.plot(*refl_R, 's', color=PALETTE[3], markersize=6, zorder=5)

# Lines from eyes to reflection points
ax.plot([eye_L[0], refl_L[0]], [eye_L[1], refl_L[1]], '--', color=PALETTE[1], linewidth=1, alpha=0.7)
ax.plot([eye_R[0], refl_R[0]], [eye_R[1], refl_R[1]], '--', color=PALETTE[3], linewidth=1, alpha=0.7)

# Reflected rays to paper
paper_L = (-35, 0)
paper_R = (35, 0)
ax.plot([refl_L[0], paper_L[0]], [refl_L[1], paper_L[1]], '-', color=PALETTE[1], linewidth=1.2)
ax.plot([refl_R[0], paper_R[0]], [refl_R[1], paper_R[1]], '-', color=PALETTE[3], linewidth=1.2)

ax.plot(*paper_L, 'v', color=PALETTE[1], markersize=7)
ax.plot(*paper_R, 'v', color=PALETTE[3], markersize=7)
ax.text(paper_L[0], paper_L[1]-10, r'$P_L$', ha='center', fontsize=8, color=PALETTE[1])
ax.text(paper_R[0], paper_R[1]-10, r'$P_R$', ha='center', fontsize=8, color=PALETTE[3])

# Disparity annotation
ax.annotate('', xy=(paper_R[0], 5), xytext=(paper_L[0], 5),
            arrowprops=dict(arrowstyle='<->', color=PALETTE[4], lw=1.5))
ax.text(0, 8, r'$\Delta x$', ha='center', fontsize=9, color=PALETTE[4], fontweight='bold')

# Virtual image
ax.plot(0, 50, '*', color=PALETTE[5], markersize=12, zorder=5)
ax.text(5, 52, 'Virtual\nimage', fontsize=7, color=PALETTE[5])

ax.set_xlabel('x (mm)', fontsize=9)
ax.set_ylabel('z (mm)', fontsize=9)
ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

# ===== Panel (b): Disparity vs distance =====
ax = axes[1]

d_IPD = 64  # mm
L_obs = np.linspace(300, 800, 100)

d_virtual_list = [20, 40, 60, 80]
for i, d_v in enumerate(d_virtual_list):
    disparity_angle = d_IPD * d_v / (L_obs * (L_obs - d_v)) * 180 / np.pi * 60
    ax.plot(L_obs, disparity_angle, color=PALETTE[i], linewidth=1.8,
            label=r'$d_v=%d$ mm' % d_v)

ax.axhline(y=60, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.text(310, 62, 'Diplopia threshold (60 arcmin)', fontsize=7, color='red')
ax.axhline(y=20, color=PALETTE[2], linestyle=':', linewidth=1, alpha=0.7)
ax.text(310, 22, 'Comfort zone (20 arcmin)', fontsize=7, color=PALETTE[2])

ax.fill_between(L_obs, 0, 20, alpha=0.08, color=PALETTE[2])

ax.set_xlabel('Observation distance L (mm)', fontsize=9)
ax.set_ylabel('Binocular disparity (arcmin)', fontsize=9)
ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
ax.set_ylim(0, 100)
ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

# ===== Panel (c): Depth jump risk heatmap =====
ax = axes[2]

R_vals = np.linspace(10, 40, 30)
alpha_vals = np.linspace(30, 75, 30)
R_grid, A_grid = np.meshgrid(R_vals, alpha_vals)

d_depth_range = R_grid / (2 * np.cos(np.radians(A_grid))) * 1.5
L_ref = 500
disparity_var = d_IPD * d_depth_range / (L_ref**2) * 180 / np.pi * 60

vmax_val = float(disparity_var.max())
im = ax.contourf(R_grid, A_grid, disparity_var, levels=np.linspace(0, vmax_val, 16), cmap='RdYlGn_r')
cb = plt.colorbar(im, ax=ax, label='Disparity variation (arcmin)')
cb.ax.tick_params(labelsize=7)

cs = ax.contour(R_grid, A_grid, disparity_var, levels=[20, 60],
                colors=['green', 'red'], linewidths=[1.5, 1.5], linestyles=['--', '-'])
ax.clabel(cs, fmt={20.0: 'Comfort\n20\'', 60.0: 'Diplopia\n60\''}, fontsize=7, inline=True)

ax.set_xlabel('Cylinder radius R (mm)', fontsize=9)
ax.set_ylabel(r'Elevation angle $\alpha$ (deg)', fontsize=9)
ax.text(0.02, 0.98, '(c)', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'fig_binocular_analysis.pdf'),
            dpi=300, bbox_inches='tight')
print("OK: fig_binocular_analysis.pdf generated")
