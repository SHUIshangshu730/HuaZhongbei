"""Visual comfort zones and virtual image depth consistency diagram"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from _utils.plot_utils import setup_style, PALETTE, COLORS
    setup_style()
except:
    PALETTE = ['#7AAEC8','#E8945A','#7BC8A4','#9B8EC4','#E0A0A0','#F0C05A','#8FAEC0','#A8C4D8']
    COLORS = {'primary':'#7AAEC8','secondary':'#E8945A','accent':'#7BC8A4'}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
ax.set_xlim(-100, 100)
ax.set_ylim(-30, 280)
ax.set_aspect('equal')
ax.axis('off')

ax.plot([-90, 90], [0, 0], 'k-', linewidth=2.5)
ax.text(0, -18, 'Paper plane ($z=0$)', ha='center', fontsize=8, style='italic', color='#555')

cyl_R = 18
cyl_H = 70
rect = Rectangle((-cyl_R, 0), 2*cyl_R, cyl_H, linewidth=1.5,
                 edgecolor=PALETTE[0], facecolor=PALETTE[0], alpha=0.2)
ax.add_patch(rect)
ax.text(0, cyl_H/2, 'Cylinder\nmirror', ha='center', va='center', fontsize=7,
        color=PALETTE[0], fontweight='bold')

eye_y = 240
d_half = 28
E_L = (-d_half, eye_y)
E_R = (d_half, eye_y)
E_C = (0, eye_y)

for ex, ey, label, col in [(E_L[0], E_L[1], r'$E_L$', PALETTE[1]),
                             (E_R[0], E_R[1], r'$E_R$', PALETTE[3])]:
    ax.plot(ex, ey, 'o', color=col, markersize=12, zorder=10,
            markeredgecolor='white', markeredgewidth=1)
    ax.text(ex + (8 if ex > 0 else -18), ey+5, label, fontsize=10,
            color=col, fontweight='bold')

ax.annotate('', xy=(d_half, eye_y-12), xytext=(-d_half, eye_y-12),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2))
ax.text(0, eye_y-22, r'$d_{\mathrm{IPD}} \approx 64$ mm', ha='center', fontsize=7, color='gray')

zones = [
    (2, PALETTE[2], 0.25, r'Clear $2^\circ$'),
    (5, PALETTE[0], 0.15, r'Effective $5^\circ$'),
    (15, PALETTE[5], 0.08, r'Comfortable $15^\circ$'),
]

view_dir_angle = -90
for half_deg, color, alpha, label in reversed(zones):
    angle_L = view_dir_angle - half_deg
    angle_R = view_dir_angle + half_deg
    cone_len = 220
    x_L = E_C[0] + cone_len * np.cos(np.radians(angle_L))
    y_L = E_C[1] + cone_len * np.sin(np.radians(angle_L))
    x_R = E_C[0] + cone_len * np.cos(np.radians(angle_R))
    y_R = E_C[1] + cone_len * np.sin(np.radians(angle_R))
    triangle = plt.Polygon([(E_C[0], E_C[1]), (x_L, y_L), (x_R, y_R)],
                           alpha=alpha, color=color, zorder=1)
    ax.add_patch(triangle)

for i, (half_deg, color, alpha, label) in enumerate(zones):
    y_label = 180 - i * 22
    ax.text(65, y_label, label, fontsize=7, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8,
                      edgecolor=color, linewidth=0.5))

M_top = (0, cyl_H)
ax.plot([E_L[0], M_top[0]], [E_L[1], M_top[1]], '--', color=PALETTE[1], linewidth=1, alpha=0.6)
ax.plot([E_R[0], M_top[0]], [E_R[1], M_top[1]], '--', color=PALETTE[3], linewidth=1, alpha=0.6)

P_L_paper = (-45, 0)
P_R_paper = (45, 0)
ax.plot([M_top[0]-5, P_L_paper[0]], [M_top[1], P_L_paper[1]], '-', color=PALETTE[1], linewidth=1.2, alpha=0.7)
ax.plot([M_top[0]+5, P_R_paper[0]], [M_top[1], P_R_paper[1]], '-', color=PALETTE[3], linewidth=1.2, alpha=0.7)

ax.plot(*P_L_paper, 'v', color=PALETTE[1], markersize=8, zorder=8)
ax.plot(*P_R_paper, 'v', color=PALETTE[3], markersize=8, zorder=8)
ax.text(P_L_paper[0]-3, P_L_paper[1]-14, r'$P_L$', fontsize=8, color=PALETTE[1], ha='center')
ax.text(P_R_paper[0]+3, P_R_paper[1]-14, r'$P_R$', fontsize=8, color=PALETTE[3], ha='center')

ax.annotate('', xy=(P_R_paper[0], 6), xytext=(P_L_paper[0], 6),
            arrowprops=dict(arrowstyle='<->', color=PALETTE[4], lw=1.8))
ax.text(0, 10, r'$\Delta P$', ha='center', fontsize=10, color=PALETTE[4], fontweight='bold')

ax.annotate('', xy=(-75, 0), xytext=(-75, eye_y),
            arrowprops=dict(arrowstyle='<->', color='#888', lw=1))
ax.text(-82, eye_y/2, r'$L$', fontsize=10, color='#888', ha='center', rotation=90)

ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

# ===== Panel (b): Virtual image depth and fusion zones =====
ax = axes[1]
ax.set_xlim(-60, 100)
ax.set_ylim(-30, 260)
ax.set_aspect('equal')
ax.axis('off')

ax.plot([-50, 90], [0, 0], 'k-', linewidth=2.5)
ax.text(20, -18, 'Paper plane', ha='center', fontsize=8, style='italic', color='#555')

cyl_x_center = 20
rect2 = Rectangle((cyl_x_center - cyl_R, 0), 2*cyl_R, cyl_H, linewidth=1.5,
                   edgecolor=PALETTE[0], facecolor=PALETTE[0], alpha=0.2)
ax.add_patch(rect2)

eye2_y = 230
E2_L = (cyl_x_center - d_half, eye2_y)
E2_R = (cyl_x_center + d_half, eye2_y)

ax.plot(*E2_L, 'o', color=PALETTE[1], markersize=10, zorder=10, markeredgecolor='white')
ax.plot(*E2_R, 'o', color=PALETTE[3], markersize=10, zorder=10, markeredgecolor='white')
ax.text(E2_L[0]-15, E2_L[1]+3, r'$E_L$', fontsize=9, color=PALETTE[1])
ax.text(E2_R[0]+5, E2_R[1]+3, r'$E_R$', fontsize=9, color=PALETTE[3])

vi_depths = [
    (30, '$d_{v1}$', PALETTE[2]),
    (70, '$d_{v2}$', PALETTE[4]),
]

for d_v, label, color in vi_depths:
    vi_y = cyl_H / 2
    vi_x = cyl_x_center - cyl_R - d_v
    ax.plot(vi_x, vi_y, '*', color=color, markersize=14, zorder=10)
    ax.text(vi_x - 5, vi_y + 12, label, fontsize=7, color=color, ha='center', fontweight='bold')
    ax.plot([E2_L[0], vi_x], [E2_L[1], vi_y], ':', color=color, linewidth=0.8, alpha=0.5)
    ax.plot([E2_R[0], vi_x], [E2_R[1], vi_y], ':', color=color, linewidth=0.8, alpha=0.5)

ax.annotate('', xy=(cyl_x_center - cyl_R - 70, cyl_H/2 - 8),
            xytext=(cyl_x_center - cyl_R - 30, cyl_H/2 - 8),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
ax.text(cyl_x_center - cyl_R - 50, cyl_H/2 - 20, r'$\Delta d_v$', fontsize=10,
        color='red', ha='center', fontweight='bold')

# Fusion zone legend
comfort_txt = "Comfort: " + r"$\eta < 20'$"
fusible_txt = "Fusible: " + r"$20' \leq \eta < 60'$"
diplopia_txt = "Diplopia: " + r"$\eta \geq 60'$"

zone_data = [
    (160, comfort_txt, PALETTE[2], 0.15),
    (130, fusible_txt, PALETTE[5], 0.12),
    (100, diplopia_txt, '#E05555', 0.12),
]

for y_pos, text, color, alpha in zone_data:
    box = Rectangle((52, y_pos - 10), 46, 20, linewidth=1.2,
                     edgecolor=color, facecolor=color, alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(75, y_pos, text, fontsize=6.5, color=color, ha='center', va='center', fontweight='bold')

ax.annotate(r'$\eta = \frac{d_{\mathrm{IPD}} \cdot d_v}{L(L-d_v)}$',
            xy=(55, 75), fontsize=9, color='#444',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F7FA', edgecolor='#CCC', alpha=0.9))

ax.text(55, 55, r'$\Delta d_v < 22.7$ mm', fontsize=8, color='red',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0F0', edgecolor='red', alpha=0.8))

ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'fig_visual_zones.pdf'),
            dpi=300, bbox_inches='tight')
print("OK: fig_visual_zones.pdf generated")
