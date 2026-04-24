#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from numpy.typing import NDArray


ROOT = Path("/home/xianz/huazhongbei")
BASE_SCRIPT_PATH = ROOT / "anamorphosis/generate_patterns.py"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/cylindrical_anamorphosis"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "parameter_report.json"
PLANE_HEIGHT_CANDIDATES = [22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0]

FloatImageArray = NDArray[np.float32]
FloatArray = NDArray[np.float64]
MaskArray = NDArray[np.bool_]


def load_base_module(script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_patterns_base", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GP = load_base_module(BASE_SCRIPT_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate supplementary Huazhong Cup paper figures from the cylindrical anamorphosis model.")
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT_PATH), help="Path to parameter_report.json from generate_patterns.py.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for supplementary figures and validation files.")
    return parser.parse_args()


def load_report(report_path: Path) -> dict[str, Any]:
    return json.loads(report_path.read_text(encoding="utf-8"))


def geometry_from_report(report: dict[str, Any]) -> Any:
    return GP.GeometryCandidate(**report["selected_geometry"])


def load_job_specs(report: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for job in report["jobs"]:
        input_path = Path(job["input_path"]).resolve()
        image_array, image_meta = GP.load_rgb_image(input_path, GP.DEFAULT_MAX_SIDE)
        specs.append(
            {
                "name": job["name"],
                "input_path": input_path,
                "image_array": image_array,
                "image_meta": image_meta,
            }
        )
    return specs


def sample_page_image(page_image: FloatImageArray, Ax: FloatArray, Ay: FloatArray, valid_mask: MaskArray) -> tuple[FloatImageArray, MaskArray]:
    height_px, width_px = page_image.shape[:2]
    px = (Ax - GP.PAGE_X_MIN) / GP.A4_WIDTH_MM * (width_px - 1)
    py = (GP.PAGE_Y_MAX - Ay) / GP.A4_HEIGHT_MM * (height_px - 1)

    sample_mask = valid_mask & (px >= 0.0) & (px <= width_px - 1) & (py >= 0.0) & (py <= height_px - 1)
    safe_px = np.where(sample_mask, px, 0.0)
    safe_py = np.where(sample_mask, py, 0.0)
    x0 = np.floor(safe_px).astype(np.int64)
    y0 = np.floor(safe_py).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width_px - 1)
    y1 = np.clip(y0 + 1, 0, height_px - 1)
    x0 = np.clip(x0, 0, width_px - 1)
    y0 = np.clip(y0, 0, height_px - 1)

    dx = (px - x0).astype(np.float32)
    dy = (py - y0).astype(np.float32)

    sampled = np.ones((*Ax.shape, 3), dtype=np.float32)
    if not np.any(sample_mask):
        return sampled, sample_mask

    c00 = page_image[y0, x0]
    c10 = page_image[y0, x1]
    c01 = page_image[y1, x0]
    c11 = page_image[y1, x1]
    sampled_value = (
        (1.0 - dx)[..., None] * (1.0 - dy)[..., None] * c00
        + dx[..., None] * (1.0 - dy)[..., None] * c10
        + (1.0 - dx)[..., None] * dy[..., None] * c01
        + dx[..., None] * dy[..., None] * c11
    )
    sampled[sample_mask] = sampled_value[sample_mask]
    return sampled, sample_mask


def global_ssim(reference: NDArray[np.float64], estimate: NDArray[np.float64]) -> float:
    scores: list[float] = []
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    for channel in range(reference.shape[1]):
        x = reference[:, channel].astype(np.float64)
        y = estimate[:, channel].astype(np.float64)
        mu_x = float(np.mean(x))
        mu_y = float(np.mean(y))
        var_x = float(np.var(x))
        var_y = float(np.var(y))
        cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
        scores.append(float(numerator / denominator) if denominator > 0.0 else 1.0)
    return float(np.mean(scores))


def compute_reconstruction_metrics(reference: FloatImageArray, estimate: FloatImageArray, valid_mask: MaskArray) -> dict[str, Any]:
    if not np.any(valid_mask):
        return {
            "valid_fraction": 0.0,
            "mae": None,
            "rmse": None,
            "psnr_db": None,
            "max_abs_error": None,
            "mean_channel_correlation": None,
            "global_ssim": None,
        }

    ref = reference[valid_mask].reshape(-1, 3).astype(np.float64)
    est = estimate[valid_mask].reshape(-1, 3).astype(np.float64)
    diff = est - ref
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    rmse = float(np.sqrt(mse))
    psnr_db = float(20.0 * np.log10(1.0 / rmse)) if rmse > 0.0 else float("inf")
    correlations: list[float] = []
    for channel in range(3):
        ref_c = ref[:, channel]
        est_c = est[:, channel]
        if np.std(ref_c) <= 1.0e-12 or np.std(est_c) <= 1.0e-12:
            correlations.append(1.0 if np.allclose(ref_c, est_c) else 0.0)
        else:
            correlations.append(float(np.corrcoef(ref_c, est_c)[0, 1]))

    return {
        "valid_fraction": float(valid_mask.mean()),
        "mae": float(np.mean(abs_diff)),
        "rmse": rmse,
        "psnr_db": psnr_db,
        "max_abs_error": float(np.max(abs_diff)),
        "mean_channel_correlation": float(np.mean(correlations)),
        "global_ssim": global_ssim(ref, est),
    }


def build_job_contexts(report: dict[str, Any], geometry: Any, job_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    dpi = int(report["page"]["dpi"])
    for spec in job_specs:
        source_image = spec["image_array"]
        plane_height_mm = float(report["search_results"][spec["name"]]["plane_height_mm"])
        height_px, width_px = source_image.shape[:2]
        X, Y, Z, plane_width_mm = GP.build_virtual_plane(width_px, height_px, plane_height_mm, geometry)
        mapping = GP.inverse_map_virtual_plane(X, Y, Z, geometry)
        clipped_mask = GP.apply_page_clip(mapping["valid"], mapping["Ax"], mapping["Ay"], geometry, 0.0)
        paper_pattern, occupancy = GP.render_paper_pattern(source_image, mapping["Ax"], mapping["Ay"], clipped_mask, dpi)
        reconstructed, sample_mask = sample_page_image(paper_pattern, mapping["Ax"], mapping["Ay"], clipped_mask)
        error_map = np.mean(np.abs(reconstructed - source_image), axis=2)
        contexts.append(
            {
                "spec": spec,
                "plane_height_mm": plane_height_mm,
                "plane_width_mm": plane_width_mm,
                "mapping": mapping,
                "clipped_mask": clipped_mask,
                "paper_pattern": paper_pattern,
                "occupancy": occupancy,
                "reconstructed": reconstructed,
                "sample_mask": sample_mask,
                "error_map": error_map,
                "metrics": compute_reconstruction_metrics(source_image, reconstructed, sample_mask),
            }
        )
    return contexts


def save_target_figures(contexts: list[dict[str, Any]], output_dir: Path) -> list[str]:
    created: list[str] = []
    for context in contexts:
        spec = context["spec"]
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        ax.imshow(spec["image_array"])
        ax.set_title(
            f"{spec['name']} target image\n"
            f"{spec['image_meta']['working_size_px'][0]} x {spec['image_meta']['working_size_px'][1]} px, "
            f"virtual plane {context['plane_width_mm']:.2f} x {context['plane_height_mm']:.2f} mm"
        )
        ax.axis("off")
        fig.tight_layout()
        output_path = output_dir / f"{spec['name']}_target_image.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        created.append(output_path.name)

    fig, axes = plt.subplots(1, len(contexts), figsize=(6.0 * len(contexts), 5.0), constrained_layout=True)
    axes_list = np.atleast_1d(axes)
    for ax, context in zip(axes_list, contexts, strict=True):
        spec = context["spec"]
        ax.imshow(spec["image_array"])
        ax.set_title(f"{spec['name']} target")
        ax.axis("off")
    gallery_path = output_dir / "target_image_gallery.png"
    fig.savefig(gallery_path, dpi=180)
    plt.close(fig)
    created.append(gallery_path.name)
    return created


def pick_representative_indices(mask: MaskArray, preferred_col: int) -> tuple[int, int]:
    _, width_px = mask.shape
    search_order = sorted(range(width_px), key=lambda col: abs(col - preferred_col))
    for col in search_order:
        valid_rows = np.flatnonzero(mask[:, col])
        if valid_rows.size > 0:
            row = int(valid_rows[(2 * valid_rows.size) // 3])
            return row, col
    raise RuntimeError("No valid representative point could be selected for the geometry schematic.")


def save_geometry_schematic(report: dict[str, Any], geometry: Any, contexts: list[dict[str, Any]], output_dir: Path) -> str:
    context = max(contexts, key=lambda item: item["spec"]["image_array"].shape[0] * item["spec"]["image_array"].shape[1])
    mapping = context["mapping"]
    mask = context["clipped_mask"]

    side_row, side_col = pick_representative_indices(mask, mask.shape[1] // 2)
    top_row, top_col = pick_representative_indices(mask, (3 * mask.shape[1]) // 4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    ax_top = axes[0]
    ax_top.add_patch(Rectangle((GP.PAGE_X_MIN, GP.PAGE_Y_MIN), GP.A4_WIDTH_MM, GP.A4_HEIGHT_MM, fill=False, edgecolor="black", linewidth=1.2))
    ax_top.add_patch(Circle((geometry.x0, geometry.y0), geometry.R, fill=False, edgecolor="firebrick", linewidth=2.0))
    plane_x_min = geometry.x0 - context["plane_width_mm"] / 2.0
    plane_x_max = geometry.x0 + context["plane_width_mm"] / 2.0
    ax_top.plot([plane_x_min, plane_x_max], [geometry.y_img, geometry.y_img], color="royalblue", linewidth=2.0)
    ax_top.scatter([geometry.xv], [geometry.yv], color="darkgreen", s=60)
    ax_top.text(geometry.xv + 3.0, geometry.yv - 10.0, "Viewer V", color="darkgreen")

    x_virtual = float(mapping["X"][top_row, top_col])
    y_hit = float(mapping["Hy"][top_row, top_col])
    x_hit = float(mapping["Hx"][top_row, top_col])
    x_paper = float(mapping["Ax"][top_row, top_col])
    y_paper = float(mapping["Ay"][top_row, top_col])
    ax_top.scatter([x_virtual], [geometry.y_img], color="royalblue", s=40)
    ax_top.scatter([x_hit], [y_hit], color="firebrick", s=40)
    ax_top.scatter([x_paper], [y_paper], color="black", s=40)
    ax_top.plot([geometry.xv, x_hit], [geometry.yv, y_hit], color="royalblue", linestyle="--", linewidth=1.6)
    ax_top.plot([x_hit, x_paper], [y_hit, y_paper], color="black", linewidth=1.8)
    ax_top.text(x_virtual + 2.0, geometry.y_img + 5.0, "I")
    ax_top.text(x_hit + 2.0, y_hit + 5.0, "H")
    ax_top.text(x_paper + 2.0, y_paper + 5.0, "A")
    ax_top.set_title("Top view: paper, cylinder, virtual plane, and ray path")
    ax_top.set_xlabel("x / mm")
    ax_top.set_ylabel("y / mm")
    ax_top.set_aspect("equal")
    ax_top.set_xlim(GP.PAGE_X_MIN - 10.0, GP.PAGE_X_MAX + 10.0)
    ax_top.set_ylim(min(GP.PAGE_Y_MIN, geometry.yv) - 20.0, GP.PAGE_Y_MAX + 15.0)

    ax_side = axes[1]
    ax_side.add_patch(Rectangle((geometry.y0 - geometry.R, 0.0), 2.0 * geometry.R, geometry.H, fill=False, edgecolor="firebrick", linewidth=2.0))
    ax_side.axhline(0.0, color="black", linewidth=1.2)
    ax_side.axvline(geometry.y_img, color="royalblue", linestyle="--", linewidth=1.5)
    z_min = geometry.z_img_center - context["plane_height_mm"] / 2.0
    z_max = geometry.z_img_center + context["plane_height_mm"] / 2.0
    ax_side.plot([geometry.y_img, geometry.y_img], [z_min, z_max], color="royalblue", linewidth=2.0)

    z_virtual = float(mapping["Z"][side_row, side_col])
    y_hit_side = float(mapping["Hy"][side_row, side_col])
    z_hit_side = float(mapping["Hz"][side_row, side_col])
    y_paper_side = float(mapping["Ay"][side_row, side_col])
    ax_side.scatter([geometry.yv], [geometry.zv], color="darkgreen", s=60)
    ax_side.scatter([geometry.y_img], [z_virtual], color="royalblue", s=40)
    ax_side.scatter([y_hit_side], [z_hit_side], color="firebrick", s=40)
    ax_side.scatter([y_paper_side], [0.0], color="black", s=40)
    ax_side.plot([geometry.yv, y_hit_side], [geometry.zv, z_hit_side], color="royalblue", linestyle="--", linewidth=1.6)
    ax_side.plot([y_hit_side, y_paper_side], [z_hit_side, 0.0], color="black", linewidth=1.8)
    ax_side.text(geometry.yv + 4.0, geometry.zv - 8.0, "V", color="darkgreen")
    ax_side.text(geometry.y_img + 3.0, z_virtual + 4.0, "I")
    ax_side.text(y_hit_side + 3.0, z_hit_side + 4.0, "H")
    ax_side.text(y_paper_side + 3.0, 4.0, "A")
    ax_side.annotate(f"R = {geometry.R:.1f} mm", xy=(geometry.y0 + geometry.R, geometry.H * 0.75), xytext=(geometry.y0 + geometry.R + 15.0, geometry.H * 0.82), arrowprops={"arrowstyle": "->", "linewidth": 1.0})
    ax_side.annotate(f"H = {geometry.H:.0f} mm", xy=(geometry.y0 + geometry.R + 2.0, geometry.H), xytext=(geometry.y0 + geometry.R + 18.0, geometry.H - 10.0), arrowprops={"arrowstyle": "->", "linewidth": 1.0})
    ax_side.set_title("Side view: reflection from virtual image to paper")
    ax_side.set_xlabel("y / mm")
    ax_side.set_ylabel("z / mm")
    ax_side.set_xlim(min(geometry.yv, GP.PAGE_Y_MIN) - 20.0, GP.PAGE_Y_MAX + 20.0)
    ax_side.set_ylim(-10.0, max(geometry.H, geometry.zv) + 20.0)

    fig.suptitle(
        "Cylindrical anamorphosis geometry schematic\n"
        f"(x0, y0, R, H)=({geometry.x0:.0f}, {geometry.y0:.0f}, {geometry.R:.0f}, {geometry.H:.0f}) mm, "
        f"V=({geometry.xv:.0f}, {geometry.yv:.0f}, {geometry.zv:.0f}) mm, y_img={geometry.y_img:.0f} mm",
        fontsize=13,
    )

    output_path = output_dir / "geometry_schematic.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path.name


def add_flow_box(ax: plt.Axes, center: tuple[float, float], text: str, box_color: str) -> None:
    width = 0.26
    height = 0.11
    x0 = center[0] - width / 2.0
    y0 = center[1] - height / 2.0
    patch = FancyBboxPatch((x0, y0), width, height, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor=box_color, edgecolor="black", linewidth=1.1)
    ax.add_patch(patch)
    ax.text(center[0], center[1], text, ha="center", va="center", fontsize=10)


def add_flow_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.4, color="black")
    ax.add_patch(arrow)


def save_workflow_flowchart(output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    boxes = {
        "input": ((0.20, 0.84), "Input target images\n(fig3 / fig4)", "#dceeff"),
        "search": ((0.20, 0.60), "Search common geometry\n(score + A4 fit + distortion)", "#e4f5dc"),
        "map": ((0.50, 0.60), "Inverse map I -> H -> A\nusing cylinder reflection law", "#fff2cc"),
        "raster": ((0.80, 0.60), "Rasterize paper pattern\non A4 page", "#fce5cd"),
        "reconstruct": ((0.50, 0.33), "Reconstruct mirror image\nby sampling paper through the same mapping", "#eadcf8"),
        "validate": ((0.80, 0.33), "Output metrics + diagnostics\nRMSE / PSNR / SSIM / sensitivity", "#f4cccc"),
    }
    for center, text, color in boxes.values():
        add_flow_box(ax, center, text, color)

    add_flow_arrow(ax, (0.20, 0.78), (0.20, 0.66))
    add_flow_arrow(ax, (0.33, 0.60), (0.37, 0.60))
    add_flow_arrow(ax, (0.63, 0.60), (0.67, 0.60))
    add_flow_arrow(ax, (0.50, 0.54), (0.50, 0.39))
    add_flow_arrow(ax, (0.63, 0.33), (0.67, 0.33))
    add_flow_arrow(ax, (0.80, 0.54), (0.80, 0.39))

    ax.text(0.50, 0.12, "No change to the mathematical model: only additional visualization, reconstruction, and validation assets.", ha="center", va="center", fontsize=10)
    ax.set_title("Supplementary asset-generation workflow", fontsize=14)

    output_path = output_dir / "algorithm_workflow_flowchart.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path.name


def save_reconstruction_figures(contexts: list[dict[str, Any]], geometry: Any, output_dir: Path) -> list[str]:
    created: list[str] = []
    extent = [GP.PAGE_X_MIN, GP.PAGE_X_MAX, GP.PAGE_Y_MIN, GP.PAGE_Y_MAX]
    for context in contexts:
        spec = context["spec"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

        axes[0, 0].imshow(spec["image_array"])
        axes[0, 0].set_title(f"{spec['name']} target image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(np.flipud(context["paper_pattern"]), extent=extent, origin="lower")
        axes[0, 1].add_patch(Rectangle((GP.PAGE_X_MIN, GP.PAGE_Y_MIN), GP.A4_WIDTH_MM, GP.A4_HEIGHT_MM, fill=False, edgecolor="black", linewidth=1.0))
        axes[0, 1].add_patch(Circle((geometry.x0, geometry.y0), geometry.R, fill=False, edgecolor="firebrick", linewidth=1.2))
        axes[0, 1].set_title("Generated paper pattern")
        axes[0, 1].set_xlabel("x / mm")
        axes[0, 1].set_ylabel("y / mm")
        axes[0, 1].set_aspect("equal")

        axes[1, 0].imshow(np.clip(context["reconstructed"], 0.0, 1.0))
        axes[1, 0].set_title(
            "Model-based mirror reconstruction\n"
            f"RMSE={context['metrics']['rmse']:.4f}, PSNR={context['metrics']['psnr_db']:.2f} dB"
        )
        axes[1, 0].axis("off")

        err = axes[1, 1].imshow(context["error_map"], cmap="magma", vmin=0.0, vmax=max(0.10, float(np.nanpercentile(context["error_map"], 99))))
        axes[1, 1].set_title(
            "Absolute RGB error map\n"
            f"SSIM={context['metrics']['global_ssim']:.4f}, corr={context['metrics']['mean_channel_correlation']:.4f}"
        )
        axes[1, 1].axis("off")
        fig.colorbar(err, ax=axes[1, 1], fraction=0.046, pad=0.04)

        output_path = output_dir / f"{spec['name']}_mirror_reconstruction_comparison.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        created.append(output_path.name)
    return created


def save_candidate_tradeoff_figure(report: dict[str, Any], output_dir: Path) -> str:
    candidates = report["top_geometry_candidates"]
    labels = [f"#{idx + 1}" for idx in range(len(candidates))]
    total_scores = [float(item["total_score"]) for item in candidates]
    mean_conditions = [float(np.mean([job["median_condition"] for job in item["jobs"].values()])) for item in candidates]
    mean_paper_fractions = [float(np.mean([job["paper_fraction"] for job in item["jobs"].values()])) for item in candidates]
    fig3_fractions = [float(item["jobs"]["fig3"]["paper_fraction"]) for item in candidates]
    fig4_fractions = [float(item["jobs"]["fig4"]["paper_fraction"]) for item in candidates]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    bubble_sizes = [900.0 * fraction for fraction in mean_paper_fractions]
    scatter = axes[0].scatter(mean_conditions, total_scores, s=bubble_sizes, c=mean_paper_fractions, cmap="viridis", alpha=0.8, edgecolors="black")
    for idx, label in enumerate(labels):
        geometry = candidates[idx]["geometry"]
        axes[0].annotate(
            f"{label}\nR={geometry['R']:.0f}, yv={geometry['yv']:.0f}, zv={geometry['zv']:.0f}",
            (mean_conditions[idx], total_scores[idx]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    axes[0].set_xlabel("Mean median condition number")
    axes[0].set_ylabel("Total candidate score")
    axes[0].set_title("Search-result tradeoff: score vs. distortion")
    fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04, label="Mean paper-valid fraction")

    x = np.arange(len(candidates))
    width = 0.36
    axes[1].bar(x - width / 2.0, fig3_fractions, width=width, label="fig3", color="#5b9bd5")
    axes[1].bar(x + width / 2.0, fig4_fractions, width=width, label="fig4", color="#ed7d31")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("A4-valid fraction")
    axes[1].set_title("Per-image coverage of top geometry candidates")
    axes[1].legend()

    output_path = output_dir / "candidate_tradeoff_comparison.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path.name


def evaluate_geometry_against_jobs(job_specs: list[dict[str, Any]], geometry: Any) -> dict[str, Any]:
    job_results: dict[str, dict[str, Any]] = {}
    total_score = 0.0
    min_paper_fraction = 1.0
    for spec in job_specs:
        best_result: dict[str, Any] | None = None
        for plane_height_mm in PLANE_HEIGHT_CANDIDATES:
            result = GP.evaluate_candidate(tuple(spec["image_array"].shape[:2]), geometry, plane_height_mm)
            if best_result is None or result["score"] > best_result["score"]:
                best_result = result
        if best_result is None:
            raise RuntimeError(f"Failed to evaluate geometry for job {spec['name']}")
        job_results[spec["name"]] = best_result
        total_score += float(best_result["score"])
        min_paper_fraction = min(min_paper_fraction, float(best_result["paper_fraction"]))
    mean_condition = float(np.mean([result["median_condition"] for result in job_results.values()]))
    return {
        "total_score": float(total_score),
        "min_paper_fraction": float(min_paper_fraction),
        "mean_median_condition": mean_condition,
        "jobs": job_results,
    }


def compute_sensitivity(job_specs: list[dict[str, Any]], base_geometry: Any) -> dict[str, list[dict[str, Any]]]:
    base = asdict(base_geometry)
    sweeps: dict[str, list[float]] = {
        "y0": [34.0, 37.0, 40.0, 43.0, 46.0],
        "R": [18.0, 20.0, 22.0, 24.0, 26.0],
        "yv": [-240.0, -220.0, -200.0, -180.0, -160.0],
        "zv": [120.0, 130.0, 140.0, 150.0, 160.0],
        "y_img": [65.0, 70.0, 75.0, 80.0, 85.0],
        "z_img_center": [4.0, 8.0, 12.0, 16.0, 20.0],
    }
    sensitivity: dict[str, list[dict[str, Any]]] = {}
    for parameter, values in sweeps.items():
        rows: list[dict[str, Any]] = []
        for value in values:
            params = dict(base)
            params[parameter] = value
            geometry = GP.GeometryCandidate(**params)
            result = evaluate_geometry_against_jobs(job_specs, geometry)
            rows.append(
                {
                    "value": float(value),
                    "total_score": result["total_score"],
                    "min_paper_fraction": result["min_paper_fraction"],
                    "mean_median_condition": result["mean_median_condition"],
                }
            )
        sensitivity[parameter] = rows
    return sensitivity


def save_sensitivity_figures(base_geometry: Any, sensitivity: dict[str, list[dict[str, Any]]], output_dir: Path) -> list[str]:
    created: list[str] = []
    labels = {
        "y0": "Cylinder center y0 / mm",
        "R": "Radius R / mm",
        "yv": "Viewer yv / mm",
        "zv": "Viewer zv / mm",
        "y_img": "Virtual plane y_img / mm",
        "z_img_center": "Virtual plane z center / mm",
    }

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    for ax, parameter in zip(axes.flat, labels, strict=True):
        rows = sensitivity[parameter]
        x_vals = [row["value"] for row in rows]
        scores = [row["total_score"] for row in rows]
        coverage = [row["min_paper_fraction"] for row in rows]
        ax.plot(x_vals, scores, marker="o", color="#1f4e79", linewidth=1.8)
        ax.axvline(float(getattr(base_geometry, parameter)), color="gray", linestyle="--", linewidth=1.0)
        ax.set_xlabel(labels[parameter])
        ax.set_ylabel("Total score", color="#1f4e79")
        ax.tick_params(axis="y", labelcolor="#1f4e79")
        ax.set_title(parameter)
        ax2 = ax.twinx()
        ax2.plot(x_vals, coverage, marker="s", color="#c55a11", linewidth=1.4)
        ax2.set_ylabel("Minimum paper-valid fraction", color="#c55a11")
        ax2.tick_params(axis="y", labelcolor="#c55a11")

    output_path = output_dir / "sensitivity_score_and_coverage.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    created.append(output_path.name)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    for ax, parameter in zip(axes.flat, labels, strict=True):
        rows = sensitivity[parameter]
        x_vals = [row["value"] for row in rows]
        conditions = [row["mean_median_condition"] for row in rows]
        ax.plot(x_vals, conditions, marker="o", color="#548235", linewidth=1.8)
        ax.axvline(float(getattr(base_geometry, parameter)), color="gray", linestyle="--", linewidth=1.0)
        ax.set_xlabel(labels[parameter])
        ax.set_ylabel("Mean median condition number")
        ax.set_title(parameter)

    output_path = output_dir / "sensitivity_condition_numbers.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    created.append(output_path.name)

    return created


def save_validation_metrics(report_path: Path, report: dict[str, Any], geometry: Any, contexts: list[dict[str, Any]], sensitivity: dict[str, list[dict[str, Any]]], created_files: list[str], output_dir: Path) -> str:
    job_metrics: dict[str, Any] = {}
    for context in contexts:
        spec = context["spec"]
        occupancy = context["occupancy"]
        job_metrics[spec["name"]] = {
            "input_path": str(spec["input_path"]),
            "working_size_px": spec["image_meta"]["working_size_px"],
            "virtual_plane_mm": {
                "width": float(context["plane_width_mm"]),
                "height": float(context["plane_height_mm"]),
            },
            "reconstruction": context["metrics"],
            "paper_pattern": {
                "ink_coverage_fraction": float(np.mean(np.any(np.abs(context["paper_pattern"] - 1.0) > 1.0e-3, axis=2))),
                "occupancy_nonzero_fraction": float(np.mean(occupancy > 0.0)),
                "occupancy_mean": float(np.mean(occupancy)),
                "occupancy_max": float(np.max(occupancy)),
            },
        }

    output_path = output_dir / "validation_metrics.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_report": str(report_path),
        "selected_geometry": asdict(geometry),
        "candidate_tradeoff": report["top_geometry_candidates"],
        "jobs": job_metrics,
        "sensitivity_analysis": sensitivity,
        "generated_files": created_files,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path.name


def main() -> None:
    args = parse_args()
    report_path = Path(args.report).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = load_report(report_path)
    geometry = geometry_from_report(report)
    job_specs = load_job_specs(report)
    contexts = build_job_contexts(report, geometry, job_specs)

    created_files: list[str] = []
    created_files.extend(save_target_figures(contexts, output_dir))
    created_files.append(save_geometry_schematic(report, geometry, contexts, output_dir))
    created_files.append(save_workflow_flowchart(output_dir))
    created_files.extend(save_reconstruction_figures(contexts, geometry, output_dir))
    created_files.append(save_candidate_tradeoff_figure(report, output_dir))
    sensitivity = compute_sensitivity(job_specs, geometry)
    created_files.extend(save_sensitivity_figures(geometry, sensitivity, output_dir))
    created_files.append(save_validation_metrics(report_path, report, geometry, contexts, sensitivity, created_files.copy(), output_dir))

    for filename in created_files:
        print(output_dir / filename)


if __name__ == "__main__":
    main()
