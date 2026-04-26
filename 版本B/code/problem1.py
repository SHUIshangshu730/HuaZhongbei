"""
问题一：圆柱镜反射逆映射模型
给定目标镜面图案，生成纸面图案及最优圆柱参数
"""
import numpy as np
from PIL import Image
import json, os, sys

# A4 尺寸 (mm)
A4_W, A4_H = 210.0, 297.0

def inverse_map(theta, h, R, x0, y0, phi, alpha):
    """
    圆柱展开坐标 (theta, h) -> 纸面坐标 (xp, yp)
    alpha: 俯仰角 (rad), phi: 水平方位角 (rad)
    """
    k = h * np.tan(alpha)
    xp = x0 + R * np.cos(theta) - k * np.cos(phi) + 2 * k * np.cos(phi - theta) * np.cos(theta)
    yp = y0 + R * np.sin(theta) - k * np.sin(phi) + 2 * k * np.cos(phi - theta) * np.sin(theta)
    return xp, yp

def generate_paper_pattern(mirror_img_arr, R, x0, y0, H, phi, alpha,
                            N_theta=400, N_h=300, dpi=10.0):
    """
    逆映射：将镜面图案映射到纸面
    dpi: 像素/mm
    返回纸面图案 (numpy array, uint8)
    """
    pw = int(A4_W * dpi)
    ph = int(A4_H * dpi)
    paper = np.ones((ph, pw, 3), dtype=np.float32) * 255.0

    mh, mw = mirror_img_arr.shape[:2]
    thetas = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)
    hs = np.linspace(0, H, N_h)

    TH, HH = np.meshgrid(thetas, hs)  # (N_h, N_theta)
    xp, yp = inverse_map(TH, HH, R, x0, y0, phi, alpha)

    # 镜面展开坐标 -> 像素索引
    mi = (TH / (2 * np.pi) * mw).astype(int) % mw
    mj = ((1 - HH / H) * (mh - 1)).astype(int)
    mj = np.clip(mj, 0, mh - 1)

    # 纸面像素索引
    pxi = (xp * dpi).astype(int)
    pyi = (yp * dpi).astype(int)

    mask = (pxi >= 0) & (pxi < pw) & (pyi >= 0) & (pyi < ph)
    paper[pyi[mask], pxi[mask]] = mirror_img_arr[mj[mask], mi[mask], :3].astype(np.float32)

    return paper.astype(np.uint8)

def forward_map(paper_arr, R, x0, y0, H, phi, alpha,
                N_theta=400, N_h=300, dpi=10.0):
    """
    正向映射：从纸面图案仿真镜面图案（用于验证）
    """
    pw = int(A4_W * dpi)
    ph = int(A4_H * dpi)
    mw, mh = N_theta, N_h
    mirror_sim = np.ones((mh, mw, 3), dtype=np.float32) * 255.0

    thetas = np.linspace(0, 2 * np.pi, mw, endpoint=False)
    hs = np.linspace(0, H, mh)
    TH, HH = np.meshgrid(thetas, hs)
    xp, yp = inverse_map(TH, HH, R, x0, y0, phi, alpha)

    pxi = (xp * dpi).astype(int)
    pyi = (yp * dpi).astype(int)
    mask = (pxi >= 0) & (pxi < pw) & (pyi >= 0) & (pyi < ph)

    mirror_sim[mask] = paper_arr[pyi[mask], pxi[mask], :3].astype(np.float32)
    return mirror_sim.astype(np.uint8)

def compute_ssim(img1, img2):
    """简化 SSIM（灰度）"""
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    if a.ndim == 3:
        a = 0.299*a[:,:,0] + 0.587*a[:,:,1] + 0.114*a[:,:,2]
    if b.ndim == 3:
        b = 0.299*b[:,:,0] + 0.587*b[:,:,1] + 0.114*b[:,:,2]
    # resize to same shape
    from PIL import Image as PILImage
    if a.shape != b.shape:
        b_img = PILImage.fromarray(b.astype(np.uint8)).resize((a.shape[1], a.shape[0]))
        b = np.array(b_img).astype(np.float64)
    C1, C2 = 6.5025, 58.5225
    mu1, mu2 = a.mean(), b.mean()
    s1, s2 = a.std(), b.std()
    s12 = np.mean((a - mu1) * (b - mu2))
    ssim = (2*mu1*mu2 + C1) * (2*s12 + C2) / ((mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2))
    return float(ssim)

def compute_psnr(img1, img2):
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    if a.ndim == 3: a = a.mean(axis=2)
    if b.ndim == 3: b = b.mean(axis=2)
    from PIL import Image as PILImage
    if a.shape != b.shape:
        b_img = PILImage.fromarray(b.astype(np.uint8)).resize((a.shape[1], a.shape[0]))
        b = np.array(b_img).astype(np.float64)
    mse = np.mean((a - b)**2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(255**2 / mse))

def compute_jacobian_cv(R, x0, y0, H, phi, alpha, N_theta=200, N_h=150):
    """计算雅可比行列式变异系数（畸变指标）"""
    thetas = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
    hs = np.linspace(1.0, H, N_h)  # 避免 h=0 时 k=0
    TH, HH = np.meshgrid(thetas, hs)
    k = HH * np.tan(alpha)
    # 偏导数
    dxp_dtheta = -R*np.sin(TH) + 2*k*np.sin(phi-TH)*np.cos(TH) - 2*k*np.cos(phi-TH)*np.sin(TH)
    dyp_dtheta =  R*np.cos(TH) + 2*k*np.sin(phi-TH)*np.sin(TH) + 2*k*np.cos(phi-TH)*np.cos(TH)
    ta = np.tan(alpha)
    dxp_dh = ta * (-np.cos(phi) + 2*np.cos(phi-TH)*np.cos(TH))
    dyp_dh = ta * (-np.sin(phi) + 2*np.cos(phi-TH)*np.sin(TH))
    jac = np.abs(dxp_dtheta * dyp_dh - dyp_dtheta * dxp_dh)
    jac = jac[jac > 1e-10]
    return float(np.std(jac) / (np.mean(jac) + 1e-10))

def boundary_penalty(R, x0, y0, H, phi, alpha, N_theta=200, N_h=150):
    """超出 A4 边界的比例"""
    thetas = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
    hs = np.linspace(0, H, N_h)
    TH, HH = np.meshgrid(thetas, hs)
    xp, yp = inverse_map(TH, HH, R, x0, y0, phi, alpha)
    total = xp.size
    out = np.sum((xp < 0) | (xp > A4_W) | (yp < 0) | (yp > A4_H))
    return float(out / total)

def solve_problem1(img_path, label, params_init):
    """对单张目标镜面图案求解问题一"""
    print(f"\n=== 问题一：处理 {label} ===")
    img = Image.open(img_path).convert('RGB')
    img_arr = np.array(img)
    print(f"  图像尺寸: {img_arr.shape}")

    R, x0, y0, H, phi, alpha = params_init
    print(f"  使用参数: R={R}, x0={x0}, y0={y0}, H={H}, phi={phi:.3f}, alpha={np.degrees(alpha):.1f}°")

    # 生成纸面图案
    paper = generate_paper_pattern(img_arr, R, x0, y0, H, phi, alpha, dpi=5.0)
    paper_path = f"figures/paper_pattern_{label}.png"
    Image.fromarray(paper).save(paper_path)
    print(f"  纸面图案已保存: {paper_path}")

    # 正向验证
    sim = forward_map(paper, R, x0, y0, H, phi, alpha, dpi=5.0)
    sim_path = f"figures/mirror_sim_{label}.png"
    # resize sim to match original
    sim_img = Image.fromarray(sim).resize((img_arr.shape[1], img_arr.shape[0]))
    sim_img.save(sim_path)

    ssim_val = compute_ssim(np.array(sim_img), img_arr)
    psnr_val = compute_psnr(np.array(sim_img), img_arr)
    jac_cv = compute_jacobian_cv(R, x0, y0, H, phi, alpha)
    bp = boundary_penalty(R, x0, y0, H, phi, alpha)

    print(f"  SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f}dB, 畸变CV={jac_cv:.4f}, 超界率={bp:.4f}")

    return {
        "label": label,
        "params": {"R": R, "x0": x0, "y0": y0, "H": H, "phi": phi, "alpha_deg": float(np.degrees(alpha))},
        "ssim": ssim_val,
        "psnr": psnr_val,
        "jacobian_cv": jac_cv,
        "boundary_penalty": bp,
        "paper_pattern_path": paper_path,
        "mirror_sim_path": sim_path
    }

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    results = {}

    # 图3：蒙娜丽莎
    p3 = (20.0, 105.0, 148.0, 80.0, np.pi, np.radians(60.0))
    r3 = solve_problem1("user_data/图3.png", "fig3_mona", p3)
    results["fig3"] = r3

    # 图4：卡通猫
    p4 = (15.0, 105.0, 148.0, 60.0, np.pi, np.radians(55.0))
    r4 = solve_problem1("user_data/图4.png", "fig4_cat", p4)
    results["fig4"] = r4

    with open("figures/problem1_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n问题一结果已保存到 figures/problem1_results.json")
