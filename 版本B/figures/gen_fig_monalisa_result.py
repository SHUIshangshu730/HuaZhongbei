import os, sys
sys.path.insert(0, '.')
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS, _lighten
setup_style()
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

fig, axes = plt.subplots(1, 4, figsize=(13, 4))

imgs = [
    ('user_data/图3.png', '目标镜面图案\n(蒙娜丽莎)'),
    ('figures/paper_pattern_fig3_mona.png', '纸面变形图案\n$I_P$'),
    ('figures/mirror_sim_fig3_mona.png', '正向仿真恢复\n$I_{\mathrm{sim}}$'),
    (None, '误差热图\n$|I_{\mathrm{sim}}-I_M^*|$'),
]

target = None
sim = None
for j, (path, label) in enumerate(imgs):
    ax = axes[j]
    if path and os.path.exists(path):
        img = np.array(Image.open(path).convert('L'))
        if j == 0:
            target = img.astype(float)
        elif j == 2:
            sim = img.astype(float)
        ax.imshow(img, cmap='gray', aspect='auto')
    elif j == 3 and target is not None and sim is not None:
        # resize sim to match target
        from PIL import Image as PILImage
        sim_img = PILImage.fromarray(sim.astype(np.uint8)).resize(
            (target.shape[1], target.shape[0]), PILImage.BILINEAR)
        sim_r = np.array(sim_img).astype(float)
        err = np.abs(sim_r - target)
        im = ax.imshow(err, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, shrink=0.8, label='误差值')
    else:
        ax.text(0.5, 0.5, '图像\n不可用', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color=COLORS['ref_line'])
        ax.set_facecolor('#f5f5f5')
    ax.set_xlabel(label, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(-0.05, 1.04, f'({chr(97+j)})', transform=ax.transAxes,
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['bg_box'],
                      edgecolor=COLORS['grid'], alpha=0.9))

# SSIM标注
axes[2].text(0.5, -0.18, 'SSIM = -0.236\nPSNR = 8.86 dB',
             ha='center', va='top', transform=axes[2].transAxes,
             fontsize=8, color=COLORS['text'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=PALETTE[0], alpha=0.9))
fig.tight_layout(pad=1.5)
save_fig(fig, 'figures/fig_monalisa_result.pdf')
