"""
灵敏度分析：关键参数对SSIM的影响
"""
import numpy as np
from PIL import Image
import json, os, sys
sys.path.insert(0, '.')
from code.problem1 import generate_paper_pattern, forward_map, compute_ssim

def sensitivity_single_param(img_arr, base_params, param_name, values):
    R, x0, y0, H, phi, alpha = base_params
    idx = {"R":0,"x0":1,"y0":2,"H":3,"phi":4,"alpha":5}[param_name]
    ssims = []
    for v in values:
        p = list(base_params)
        p[idx] = v
        try:
            paper = generate_paper_pattern(img_arr, *p, dpi=3.0)
            sim = forward_map(paper, *p, dpi=3.0)
            sim_r = np.array(Image.fromarray(sim).resize((img_arr.shape[1], img_arr.shape[0])))
            s = compute_ssim(sim_r, img_arr)
        except:
            s = 0.0
        ssims.append(float(s))
    return ssims

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print("=== 灵敏度分析 ===")

    img = np.array(Image.open("user_data/图3.png").convert('RGB').resize((150, 150)))
    base = (20.0, 105.0, 148.0, 80.0, np.pi, np.radians(60.0))
    R0, x0_0, y0_0, H0, phi0, alpha0 = base

    results = {}

    R_vals = np.linspace(0.5*R0, 2*R0, 11).tolist()
    ssim_R = sensitivity_single_param(img, base, "R", R_vals)
    results["R"] = {"values": R_vals, "ssim": ssim_R, "label": "圆柱半径R (mm)"}
    print(f"R: min={min(ssim_R):.3f}, max={max(ssim_R):.3f}")

    H_vals = np.linspace(0.5*H0, 2*H0, 11).tolist()
    ssim_H = sensitivity_single_param(img, base, "H", H_vals)
    results["H"] = {"values": H_vals, "ssim": ssim_H, "label": "圆柱高度H (mm)"}
    print(f"H: min={min(ssim_H):.3f}, max={max(ssim_H):.3f}")

    phi_vals = np.linspace(phi0 - np.radians(30), phi0 + np.radians(30), 13).tolist()
    ssim_phi = sensitivity_single_param(img, base, "phi", phi_vals)
    results["phi"] = {"values": [float(np.degrees(v)) for v in phi_vals],
                      "ssim": ssim_phi, "label": "方位角φ (°)"}
    print(f"phi: min={min(ssim_phi):.3f}, max={max(ssim_phi):.3f}")

    alpha_vals = np.linspace(alpha0 - np.radians(15), alpha0 + np.radians(15), 11).tolist()
    ssim_alpha = sensitivity_single_param(img, base, "alpha", alpha_vals)
    results["alpha"] = {"values": [float(np.degrees(v)) for v in alpha_vals],
                        "ssim": ssim_alpha, "label": "俯仰角α (°)"}
    print(f"alpha: min={min(ssim_alpha):.3f}, max={max(ssim_alpha):.3f}")

    x0_vals = np.linspace(x0_0 - 10, x0_0 + 10, 11).tolist()
    ssim_x0 = sensitivity_single_param(img, base, "x0", x0_vals)
    results["x0"] = {"values": x0_vals, "ssim": ssim_x0, "label": "圆柱位置x0 (mm)"}
    print(f"x0: min={min(ssim_x0):.3f}, max={max(ssim_x0):.3f}")

    # 归一化灵敏度系数
    base_ssim = sensitivity_single_param(img, base, "R", [R0])[0]
    sens_coeffs = {}
    for pname, pval, pvals, ssims in [
        ("R", R0, R_vals, ssim_R),
        ("H", H0, H_vals, ssim_H),
        ("alpha", alpha0, alpha_vals, ssim_alpha),
        ("x0", x0_0, x0_vals, ssim_x0),
    ]:
        vals_arr = np.array(pvals)
        ssim_arr = np.array(ssims)
        dssim_dp = np.gradient(ssim_arr, vals_arr)
        mid = len(dssim_dp)//2
        S = dssim_dp[mid] * pval / (base_ssim + 1e-10)
        sens_coeffs[pname] = float(S)
    results["sensitivity_coefficients"] = sens_coeffs
    print(f"\n归一化灵敏度系数: {sens_coeffs}")

    with open("figures/sensitivity_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("灵敏度分析结果已保存到 figures/sensitivity_results.json")
