"""
问题二：纸面图案与镜面图案同时有意义的可行性分析
构造3个具体案例并生成可视化图片
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json, os, sys
sys.path.insert(0, '.')
from code.problem1 import inverse_map, generate_paper_pattern, forward_map, compute_ssim, A4_W, A4_H

def make_text_image(text, width=300, height=100, bg=0, fg=255):
    """生成文字图案（灰度）"""
    img = Image.new('L', (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    # 估算字体大小
    font_size = min(height - 10, width // max(len(text), 1) - 2)
    font_size = max(font_size, 20)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("_utils/NotoSansSC-Regular.ttf", font_size)
        except:
            font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (height - th) // 2
    draw.text((x, y), text, fill=fg, font=font)
    return np.array(img.convert('RGB'))

def make_smiley_image(width=200, height=200):
    """生成笑脸图案"""
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy, r = width//2, height//2, min(width, height)//2 - 10
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(0,0,0), width=4)
    ew = r // 5
    draw.ellipse([cx-r//3-ew, cy-r//4-ew, cx-r//3+ew, cy-r//4+ew], fill=(0,0,0))
    draw.ellipse([cx+r//3-ew, cy-r//4-ew, cx+r//3+ew, cy-r//4+ew], fill=(0,0,0))
    draw.arc([cx-r//2, cy-r//8, cx+r//2, cy+r//2], start=10, end=170, fill=(0,0,0), width=4)
    return np.array(img)

def make_star_image(width=200, height=200, n=5):
    """生成星形图案"""
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = width//2, height//2
    R_out, R_in = min(width, height)//2 - 10, min(width, height)//4
    pts = []
    for i in range(2*n):
        angle = np.pi/2 + i * np.pi / n
        r = R_out if i % 2 == 0 else R_in
        pts.append((cx + r*np.cos(angle), cy - r*np.sin(angle)))
    draw.polygon(pts, fill=(255, 200, 0), outline=(0, 0, 0))
    return np.array(img)

def make_radial_pattern(width=1050, height=1485, n_rays=24):
    """生成放射状花纹（纸面有意义图案）"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 240
    cx, cy = width//2, height//2
    for i in range(n_rays):
        angle = i * 2 * np.pi / n_rays
        x2 = int(cx + max(width, height) * np.cos(angle))
        y2 = int(cy + max(width, height) * np.sin(angle))
        color = (
            int(180 + 60 * np.cos(angle)),
            int(120 + 80 * np.sin(angle + np.pi/3)),
            int(100 + 100 * np.cos(angle + np.pi/6))
        )
        from PIL import ImageDraw as ID
        pil = Image.fromarray(img)
        draw = ID.Draw(pil)
        draw.line([(cx, cy), (x2, y2)], fill=color, width=3)
        img = np.array(pil)
    return img

def case1_art_text():
    """案例1：镜面为文字"ART"，纸面为放射状花纹"""
    print("\n--- 案例1：镜面文字ART + 纸面放射花纹 ---")
    mirror_target = make_text_image("ART", width=300, height=100)
    R, x0, y0, H = 15.0, 105.0, 148.0, 50.0
    phi, alpha = np.pi, np.radians(60.0)

    paper = generate_paper_pattern(mirror_target, R, x0, y0, H, phi, alpha, dpi=5.0)
    # 艺术加工：叠加放射状背景
    radial = make_radial_pattern(paper.shape[1], paper.shape[0])
    # 在圆柱投影区域外填充放射背景
    mask = np.all(paper == 255, axis=2)
    paper_art = paper.copy()
    paper_art[mask] = radial[mask]

    sim = forward_map(paper_art, R, x0, y0, H, phi, alpha, dpi=5.0)
    sim_resized = np.array(Image.fromarray(sim).resize((mirror_target.shape[1], mirror_target.shape[0])))
    ssim = compute_ssim(sim_resized, mirror_target)

    Image.fromarray(mirror_target).save("figures/case1_mirror_target.png")
    Image.fromarray(paper_art).save("figures/case1_paper_art.png")
    Image.fromarray(sim_resized).save("figures/case1_mirror_sim.png")
    print(f"  SSIM={ssim:.4f}")
    return {"case": "案例1_文字ART", "ssim": ssim, "feasible": ssim > 0.5,
            "mirror_target": "figures/case1_mirror_target.png",
            "paper_art": "figures/case1_paper_art.png",
            "mirror_sim": "figures/case1_mirror_sim.png"}

def case2_smiley():
    """案例2：镜面为笑脸，纸面为抽象几何"""
    print("\n--- 案例2：镜面笑脸 + 纸面抽象几何 ---")
    mirror_target = make_smiley_image(200, 200)
    R, x0, y0, H = 20.0, 105.0, 148.0, 60.0
    phi, alpha = np.pi, np.radians(55.0)

    paper = generate_paper_pattern(mirror_target, R, x0, y0, H, phi, alpha, dpi=5.0)
    # 艺术加工：色彩重映射为蓝紫渐变
    paper_art = paper.copy().astype(np.float32)
    paper_art[:,:,0] = np.clip(paper_art[:,:,0] * 0.4, 0, 255)
    paper_art[:,:,2] = np.clip(paper_art[:,:,2] * 1.2 + 30, 0, 255)
    paper_art = paper_art.astype(np.uint8)

    sim = forward_map(paper_art, R, x0, y0, H, phi, alpha, dpi=5.0)
    sim_resized = np.array(Image.fromarray(sim).resize((mirror_target.shape[1], mirror_target.shape[0])))
    ssim = compute_ssim(sim_resized, mirror_target)

    Image.fromarray(mirror_target).save("figures/case2_mirror_target.png")
    Image.fromarray(paper_art).save("figures/case2_paper_art.png")
    Image.fromarray(sim_resized).save("figures/case2_mirror_sim.png")
    print(f"  SSIM={ssim:.4f}")
    return {"case": "案例2_笑脸", "ssim": ssim, "feasible": ssim > 0.5,
            "mirror_target": "figures/case2_mirror_target.png",
            "paper_art": "figures/case2_paper_art.png",
            "mirror_sim": "figures/case2_mirror_sim.png"}

def case3_star_kaleidoscope():
    """案例3：镜面为星形，纸面为万花筒（8重对称）"""
    print("\n--- 案例3：镜面星形 + 纸面万花筒 ---")
    mirror_target = make_star_image(200, 200, n=5)
    R, x0, y0, H = 18.0, 105.0, 148.0, 55.0
    phi, alpha = np.pi, np.radians(58.0)

    paper = generate_paper_pattern(mirror_target, R, x0, y0, H, phi, alpha, dpi=5.0)
    # 艺术加工：8重旋转对称（万花筒效果）
    ph, pw = paper.shape[:2]
    cx, cy = pw//2, ph//2
    paper_art = paper.copy()
    for k in range(1, 8):
        angle = k * np.pi / 4
        pil_rot = Image.fromarray(paper).rotate(np.degrees(angle), center=(cx, cy), expand=False)
        rot_arr = np.array(pil_rot)
        mask = np.all(paper_art == 255, axis=2)
        paper_art[mask] = rot_arr[mask]

    sim = forward_map(paper_art, R, x0, y0, H, phi, alpha, dpi=5.0)
    sim_resized = np.array(Image.fromarray(sim).resize((mirror_target.shape[1], mirror_target.shape[0])))
    ssim = compute_ssim(sim_resized, mirror_target)

    Image.fromarray(mirror_target).save("figures/case3_mirror_target.png")
    Image.fromarray(paper_art).save("figures/case3_paper_art.png")
    Image.fromarray(sim_resized).save("figures/case3_mirror_sim.png")
    print(f"  SSIM={ssim:.4f}")
    return {"case": "案例3_星形万花筒", "ssim": ssim, "feasible": ssim > 0.5,
            "mirror_target": "figures/case3_mirror_target.png",
            "paper_art": "figures/case3_paper_art.png",
            "mirror_sim": "figures/case3_mirror_sim.png"}

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    results = {}
    results["case1"] = case1_art_text()
    results["case2"] = case2_smiley()
    results["case3"] = case3_star_kaleidoscope()

    # 可行性总结
    feasible_count = sum(1 for v in results.values() if v["feasible"])
    results["summary"] = {
        "total_cases": 3,
        "feasible_cases": feasible_count,
        "conclusion_level1": "自由设计场景：可行性高，放射状/对称图案天然适配圆柱反射",
        "conclusion_level2": "指定镜面场景：低频/简单图案可行，高频复杂图案需艺术加工",
        "avg_ssim": float(np.mean([v["ssim"] for v in results.values() if "ssim" in v]))
    }

    with open("figures/problem2_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n问题二结果已保存到 figures/problem2_results.json")
