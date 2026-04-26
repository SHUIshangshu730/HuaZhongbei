"""
问题三：双目标兼容性理论模型
计算兼容性指标，分析可行域
"""
import numpy as np
from PIL import Image
import json, os, sys
sys.path.insert(0, '.')
from code.problem1 import inverse_map, generate_paper_pattern, forward_map, compute_ssim, A4_W, A4_H

def compute_freq_complexity(img_arr):
    """空间频率标准差（图案复杂度）"""
    gray = img_arr.astype(np.float64)
    if gray.ndim == 3:
        gray = 0.299*gray[:,:,0] + 0.587*gray[:,:,1] + 0.114*gray[:,:,2]
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift)**2
    h, w = gray.shape
    u = np.fft.fftfreq(h) * h
    v = np.fft.fftfreq(w) * w
    V, U = np.meshgrid(v, u)
    freq = np.sqrt(U**2 + V**2)
    freq_shift = np.fft.fftshift(freq)
    weights = power / (power.sum() + 1e-10)
    mean_freq = np.sum(freq_shift * weights)
    std_freq = np.sqrt(np.sum((freq_shift - mean_freq)**2 * weights))
    return float(std_freq)

def compute_symmetry(img_arr, n_angles=8):
    """旋转对称性指标 eta_sym"""
    gray = img_arr.astype(np.float64)
    if gray.ndim == 3:
        gray = 0.299*gray[:,:,0] + 0.587*gray[:,:,1] + 0.114*gray[:,:,2]
    pil = Image.fromarray(gray.astype(np.uint8))
    scores = []
    for k in range(1, n_angles):
        angle = 360.0 * k / n_angles
        rot = np.array(pil.rotate(angle, expand=False)).astype(np.float64)
        corr = np.corrcoef(gray.flatten(), rot.flatten())[0, 1]
        scores.append(max(0.0, corr))
    return float(np.mean(scores))

def compute_main_direction(img_arr):
    """图案主方向角（梯度PCA）"""
    gray = img_arr.astype(np.float64)
    if gray.ndim == 3:
        gray = 0.299*gray[:,:,0] + 0.587*gray[:,:,1] + 0.114*gray[:,:,2]
    gy, gx = np.gradient(gray)
    cov = np.array([[np.mean(gx*gx), np.mean(gx*gy)],
                    [np.mean(gx*gy), np.mean(gy*gy)]])
    vals, vecs = np.linalg.eigh(cov)
    main_vec = vecs[:, np.argmax(vals)]
    return float(np.arctan2(main_vec[1], main_vec[0]))

def compute_tolerance(img_arr, n_trials=5):
    """语义容错度：随机仿射变换后的SSIM均值"""
    from PIL import Image as PILImage
    pil = PILImage.fromarray(img_arr)
    scores = []
    np.random.seed(42)
    for _ in range(n_trials):
        angle = np.random.uniform(-15, 15)
        scale = np.random.uniform(0.85, 1.15)
        w, h = pil.size
        rotated = pil.rotate(angle, expand=False)
        new_w, new_h = int(w*scale), int(h*scale)
        resized = rotated.resize((new_w, new_h))
        # crop/pad back to original size
        result = PILImage.new('RGB', (w, h), (128, 128, 128))
        paste_x = max(0, (w - new_w)//2)
        paste_y = max(0, (h - new_h)//2)
        result.paste(resized.crop((0, 0, min(new_w, w), min(new_h, h))), (paste_x, paste_y))
        scores.append(compute_ssim(np.array(result), img_arr))
    return float(np.mean(scores))

def compute_freq_compatibility(P_arr, M_inv_arr):
    """频谱兼容性 C_freq"""
    def get_spectrum(arr):
        gray = arr.astype(np.float64)
        if gray.ndim == 3:
            gray = 0.299*gray[:,:,0] + 0.587*gray[:,:,1] + 0.114*gray[:,:,2]
        return np.abs(np.fft.fft2(gray))
    sp = get_spectrum(P_arr)
    sm = get_spectrum(M_inv_arr)
    # resize to same shape
    if sp.shape != sm.shape:
        sm_img = Image.fromarray(sm.astype(np.uint8)).resize((sp.shape[1], sp.shape[0]))
        sm = np.array(sm_img).astype(np.float64)
    num = np.sum(sp * sm)
    denom = np.sqrt(np.sum(sp**2) * np.sum(sm**2)) + 1e-10
    return float(num / denom)

def compatibility_index(P_arr, M_arr, R=15.0, x0=105.0, y0=148.0, H=50.0,
                        phi=np.pi, alpha=None):
    """计算综合兼容性指标 C(P*, M*)"""
    if alpha is None:
        alpha = np.radians(60.0)
    # 将 M* 逆映射到纸面
    M_inv = generate_paper_pattern(M_arr, R, x0, y0, H, phi, alpha, dpi=3.0)

    C_freq = compute_freq_compatibility(P_arr, M_inv)
    C_sym = 0.5 * (compute_symmetry(P_arr) + compute_symmetry(M_arr))
    C_tol = 0.5 * (compute_tolerance(P_arr) + compute_tolerance(M_arr))
    theta_P = compute_main_direction(P_arr)
    theta_M = compute_main_direction(M_arr)
    C_dir = float(np.cos(theta_P - theta_M)**2)

    C = 0.25 * C_freq + 0.25 * C_sym + 0.25 * C_tol + 0.25 * C_dir
    return {
        "C_freq": C_freq, "C_sym": C_sym, "C_tol": C_tol, "C_dir": C_dir,
        "C_total": C,
        "feasibility": "可实现" if C >= 0.7 else ("勉强实现" if C >= 0.4 else "难以实现")
    }

def feasibility_grid_analysis():
    """可行域网格分析：不同复杂度组合的兼容性"""
    from PIL import Image as PILImage, ImageDraw
    results = []
    complexities = ["低频简单", "中等复杂", "高频复杂"]
    # 生成三类测试图案
    def make_low_freq(sz=100):
        img = np.zeros((sz, sz, 3), dtype=np.uint8)
        cx, cy = sz//2, sz//2
        for r in range(10, sz//2, 15):
            for i in range(sz):
                for j in range(sz):
                    if abs(np.sqrt((i-cy)**2+(j-cx)**2) - r) < 3:
                        img[i,j] = [200, 100, 50]
        return img

    def make_mid_freq(sz=100):
        x = np.linspace(0, 4*np.pi, sz)
        y = np.linspace(0, 4*np.pi, sz)
        X, Y = np.meshgrid(x, y)
        v = (np.sin(X) * np.cos(Y) * 127 + 128).astype(np.uint8)
        return np.stack([v, v//2+64, 255-v], axis=2)

    def make_high_freq(sz=100):
        np.random.seed(0)
        v = np.random.randint(0, 255, (sz, sz, 3), dtype=np.uint8)
        return v

    patterns = [make_low_freq(), make_mid_freq(), make_high_freq()]
    for i, (P, cP) in enumerate(zip(patterns, complexities)):
        for j, (M, cM) in enumerate(zip(patterns, complexities)):
            c = compatibility_index(P, M)
            results.append({
                "paper_complexity": cP,
                "mirror_complexity": cM,
                "C_total": c["C_total"],
                "feasibility": c["feasibility"]
            })
            print(f"  纸面={cP}, 镜面={cM}: C={c['C_total']:.3f} ({c['feasibility']})")
    return results

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    print("=== 问题三：双目标兼容性分析 ===")

    # 具体案例：用图3/图4测试兼容性
    results = {}
    try:
        img3 = np.array(Image.open("user_data/图3.png").convert('RGB').resize((200, 200)))
        img4 = np.array(Image.open("user_data/图4.png").convert('RGB').resize((200, 200)))

        print("\n案例A：纸面=图3(蒙娜丽莎), 镜面=图4(卡通猫)")
        cA = compatibility_index(img3, img4)
        print(f"  C_total={cA['C_total']:.4f}, 可行性={cA['feasibility']}")
        results["case_A_mona_cat"] = cA

        print("\n案例B：纸面=图4(卡通猫), 镜面=图3(蒙娜丽莎)")
        cB = compatibility_index(img4, img3)
        print(f"  C_total={cB['C_total']:.4f}, 可行性={cB['feasibility']}")
        results["case_B_cat_mona"] = cB
    except Exception as e:
        print(f"  图像加载失败: {e}")

    print("\n可行域网格分析：")
    grid = feasibility_grid_analysis()
    results["feasibility_grid"] = grid

    # 兼容性条件总结
    results["conditions"] = {
        "condition1": "低频优先：图案主要语义集中在低频分量（空间频率<5 cycles/cm）",
        "condition2": "对称性适配：旋转对称图案与圆柱反射机制天然兼容",
        "condition3": "语义容错：抽象图案容错度高，可承受较大几何畸变",
        "condition4": "主方向一致：两图案主方向夹角越小，兼容性越高",
        "threshold_feasible": 0.7,
        "threshold_marginal": 0.4
    }

    with open("figures/problem3_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n问题三结果已保存到 figures/problem3_results.json")
