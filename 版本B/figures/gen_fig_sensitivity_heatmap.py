import os, sys
sys.path.insert(0, '.')
from _utils.plot_utils import setup_style, save_fig, PALETTE, COLORS
setup_style()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

with open('figures/sensitivity_results.json') as f:
    sens = json.load(f)

params = ['R', 'H', 'phi', 'alpha']
param_labels = ['半径 $R$', '高度 $H$', '方位角 $\\phi$', '俯仰角 $\\alpha$']
metrics = ['SSIM均值', 'SSIM标准差', '归一化灵敏度']

data = np.zeros((len(params), len(metrics)))
for i, key in enumerate(params):
    if key not in sens:
        data[i] = [0, 0, 0]
        continue
    y = np.array(sens[key]['ssim'])
    x = np.array(sens[key]['values'])
    data[i, 0] = np.mean(y)
    data[i, 1] = np.std(y)
    # normalized sensitivity: (dy/dx) * (x_mean/y_mean)
    dy = np.gradient(y, x)
    x_mean = np.mean(x)
    y_mean = np.mean(np.abs(y)) if np.mean(np.abs(y)) > 0 else 1e-6
    data[i, 2] = np.mean(np.abs(dy)) * x_mean / y_mean

fig, ax = plt.subplots(figsize=(7, 4))
data_min, data_max = data.min(), data.max()
sns.heatmap(data, annot=False, cmap='YlOrRd',
            xticklabels=metrics, yticklabels=param_labels, ax=ax,
            linewidths=0.8, linecolor='white',
            cbar_kws={'shrink': 0.8, 'label': '指标值'})

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        norm_val = (val - data_min) / (data_max - data_min + 1e-9)
        txt_color = 'white' if norm_val > 0.65 else COLORS['text']
        ax.text(j + 0.5, i + 0.5, f'{val:.3f}', ha='center', va='center',
                fontsize=9, color=txt_color,
                fontweight='bold' if norm_val > 0.8 else 'normal')

# Highlight max sensitivity
max_idx = np.unravel_index(data[:, 2].argmax(), (len(params),))
ax.add_patch(plt.Rectangle((2, max_idx[0]), 1, 1,
             fill=False, edgecolor=COLORS['down'], linewidth=2.5))

ax.set_xticklabels(metrics, rotation=0, fontsize=9)
ax.set_yticklabels(param_labels, rotation=0, fontsize=9)
fig.tight_layout()
save_fig(fig, 'figures/fig_sensitivity_heatmap.pdf')
