import os, sys
sys.path.insert(0, '.')
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS, _lighten
setup_style()
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from PIL import Image as PILImage

# 统一显示尺寸：以目标图为基准
target_raw = np.array(PILImage.open('user_data/图3.png').convert('L'))
H_ref, W_ref = target_raw.shape

def load_gray_resized(path):
    img = PILImage.open(path).convert('L')
    img = img.resize((W_ref, H_ref), PILImage.LANCZOS)
    return np.array(img).astype(float)

target = target_raw.astype(float)
paper  = load_gray_resized('figures/paper_pattern_fig3_mona.png')
sim    = load_gray_resized('figures/mirror_sim_fig3_mona.png')
err    = np.abs(sim - target)

# 计算真实 SSIM/PSNR
from skimage.metrics import structural_similarity as ssim_fn, peak_signal_noise_ratio as psnr_fn
ssim_val = ssim_fn(target.astype(np.uint8), sim.astype(np.uint8), data_range=255)
psnr_val = psnr_fn(target.astype(np.uint8), sim.astype(np.uint8), data_range=255)

panels = [
    (target, 'gray',   '目标镜面图案\n(蒙娜丽莎)',   None),
    (paper,  'gray',   '纸面变形图案\n$I_P$',         None),
    (sim,    'gray',   '正向仿真恢复\n$I_{\\mathrm{sim}}$', None),
    (err,    'YlOrRd', '误差热图\n$|I_{\\mathrm{sim}}-I_M^*|$', (0, 80)),
]

fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))
for j, (data, cmap, label, clim) in enumerate(panels):
    ax = axes[j]
    kwargs = dict(cmap=cmap, aspect='equal')
    if clim:
        kwargs['vmin'], kwargs['vmax'] = clim
    im = ax.imshow(data, **kwargs)
    if clim:
        plt.colorbar(im, ax=ax, shrink=0.75, label='误差值')
    ax.set_xlabel(label, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(-0.05, 1.04, f'({chr(97+j)})', transform=ax.transAxes,
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['bg_box'],
                      edgecolor=COLORS['grid'], alpha=0.9))

axes[2].text(0.5, -0.20, f'SSIM = {ssim_val:.3f}\nPSNR = {psnr_val:.2f} dB',
             ha='center', va='top', transform=axes[2].transAxes,
             fontsize=8, color=COLORS['text'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=PALETTE[0], alpha=0.9))
fig.tight_layout(pad=1.5)
save_fig(fig, 'figures/fig_monalisa_result.pdf')
