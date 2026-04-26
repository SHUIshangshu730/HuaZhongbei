import os, sys
sys.path.insert(0, '.')
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS, _lighten
setup_style()
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

cases = [
    ('figures/case1_mirror_target.png', 'figures/case1_paper_art.png', 'figures/case1_mirror_sim.png', '案例1：文字"ART"'),
    ('figures/case2_mirror_target.png', 'figures/case2_paper_art.png', 'figures/case2_mirror_sim.png', '案例2：笑脸图案'),
    ('figures/case3_mirror_target.png', 'figures/case3_paper_art.png', 'figures/case3_mirror_sim.png', '案例3：星形万花筒'),
]
ssims = [0.315, 0.339, 0.321]

fig, axes = plt.subplots(3, 3, figsize=(10, 8))
col_labels = ['目标镜面图案', '纸面艺术图案', '仿真恢复图案']

for i, (mt, pa, ms, case_name) in enumerate(cases):
    for j, (path, col_label) in enumerate(zip([mt, pa, ms], col_labels)):
        ax = axes[i, j]
        if os.path.exists(path):
            img = np.array(Image.open(path).convert('RGB'))
            ax.imshow(img, aspect='auto')
        else:
            ax.text(0.5, 0.5, '不可用', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color=COLORS['ref_line'])
            ax.set_facecolor('#f5f5f5')
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(col_label, fontsize=10, fontweight='bold', color=COLORS['text'])
        if j == 0:
            ax.set_ylabel(case_name, fontsize=9, rotation=90, labelpad=5)
        if j == 2:
            ax.text(1.02, 0.5, f'SSIM\n{ssims[i]:.3f}', transform=ax.transAxes,
                    fontsize=8, va='center', ha='left', color=PALETTE[i],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=PALETTE[i], alpha=0.9))

fig.tight_layout(pad=1.5, h_pad=0.8, w_pad=0.5)
save_fig(fig, 'figures/fig_dual_meaning_cases.pdf')
