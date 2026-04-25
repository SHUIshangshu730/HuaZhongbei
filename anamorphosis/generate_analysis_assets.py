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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from numpy.typing import NDArray


ROOT = Path("/home/xianz/huazhongbei")
BASE_SCRIPT_PATH = ROOT / "anamorphosis/generate_patterns.py"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/cylindrical_anamorphosis"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "parameter_report.json"
PLANE_HEIGHT_CANDIDATES = [22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0]
FREE_DESIGN_CASES = ["flower", "compass", "fish", "arrow"]
SPECIFIED_MIRROR_CASES = ["compass", "fish", "letter_h", "dense"]
PROBLEM3_PAPER_KEYS = ["flower", "compass", "fish", "arrow", "checker", "dense"]
PROBLEM3_MIRROR_KEYS = ["compass", "fish", "flower", "letter_h", "checker", "dense"]
COMPATIBILITY_WEIGHTS = {
    "mirror_match": 0.25,
    "low_freq_alignment": 0.25,
    "symmetry_alignment": 0.25,
    "complexity_alignment": 0.25,
}
HIGH_OVERLAP_THRESHOLD = 0.74
MIXED_OVERLAP_THRESHOLD = 0.68

FloatImageArray = NDArray[np.float32]
FloatArray = NDArray[np.float64]
MaskArray = NDArray[np.bool_]


DESIGN_SYSTEM = {
    "figure_face": "#ffffff",
    "panel_face": "#ffffff",
    "panel_edge": "#c7b8a1",
    "grid": "#d7cab6",
    "ink": "#1c2733",
    "muted": "#5d6772",
    "page_fill": "#ffffff",
    "page_edge": "#705c47",
    "paper_shadow": "#d4c5ae",
    "cylinder": "#b4543a",
    "cylinder_soft": "#ebc9bf",
    "viewer": "#1d6f86",
    "viewer_soft": "#d9eef2",
    "virtual_plane": "#305d8f",
    "virtual_plane_soft": "#dce7f5",
    "ray": "#a9852d",
    "ray_soft": "#f0e3b2",
    "success": "#2e6a4f",
    "warning": "#9b6a11",
    "danger": "#9a3f38",
    "neutral_fill": "#e7ddd0",
}
FLOW_BOX_COLORS = {
    "input": "#d9e7f7",
    "search": "#ddebd8",
    "map": "#f6ebc8",
    "raster": "#f3dec5",
    "reconstruct": "#e7dbf2",
    "validate": "#f2d7d7",
}
CLASS_STYLES = {
    "Low overlap": {"marker": "^", "color": DESIGN_SYSTEM["danger"]},
    "Mixed overlap": {"marker": "s", "color": DESIGN_SYSTEM["warning"]},
    "High overlap": {"marker": "o", "color": DESIGN_SYSTEM["success"]},
}
GEOMETRY_PARAMETER_KEYS = ("x0", "y0", "R", "H", "xv", "yv", "zv", "y_img", "z_img_center")
DEFAULT_IPD_MM = 64.0
BINOCULAR_ZONE_THRESHOLD = 0.90
BINOCULAR_ZONE_SIZE_PX = 56


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 10.2,
        "axes.titlesize": 12.2,
        "axes.labelsize": 10.4,
        "axes.facecolor": DESIGN_SYSTEM["panel_face"],
        "figure.facecolor": DESIGN_SYSTEM["figure_face"],
        "savefig.facecolor": DESIGN_SYSTEM["figure_face"],
        "axes.edgecolor": DESIGN_SYSTEM["panel_edge"],
        "axes.labelcolor": DESIGN_SYSTEM["ink"],
        "axes.titlecolor": DESIGN_SYSTEM["ink"],
        "xtick.color": DESIGN_SYSTEM["muted"],
        "ytick.color": DESIGN_SYSTEM["muted"],
        "grid.color": DESIGN_SYSTEM["grid"],
        "grid.linestyle": ":",
        "grid.linewidth": 0.75,
        "legend.framealpha": 0.96,
        "legend.edgecolor": DESIGN_SYSTEM["panel_edge"],
        "mathtext.fontset": "dejavuserif",
    }
)


def style_panel(
    ax: plt.Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    equal: bool = False,
    grid: bool = False,
    hide_ticks: bool = False,
) -> None:
    ax.set_facecolor(DESIGN_SYSTEM["panel_face"])
    for spine in ax.spines.values():
        spine.set_color(DESIGN_SYSTEM["panel_edge"])
        spine.set_linewidth(1.0)
    if title is not None:
        ax.set_title(title, loc="left", pad=9.0, fontweight="semibold")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if equal:
        ax.set_aspect("equal")
    if grid:
        ax.grid(True, alpha=0.45)
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def add_panel_badge(ax: plt.Axes, text: str, *, facecolor: str, textcolor: str = "white", anchor: tuple[float, float] = (0.02, 0.98)) -> None:
    ax.text(
        anchor[0],
        anchor[1],
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="semibold",
        color=textcolor,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": facecolor, "edgecolor": "none", "alpha": 0.98},
    )


def add_figure_note(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=9.3, color=DESIGN_SYSTEM["muted"])


def save_figure(fig: plt.Figure, output_path: Path, *, dpi: int = 220) -> str:
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=DESIGN_SYSTEM["figure_face"])
    plt.close(fig)
    return output_path.name


def geometry_with_overrides(geometry: Any, **overrides: float) -> Any:
    params = asdict(geometry)
    params.update(overrides)
    return GP.GeometryCandidate(**params)


def draw_page_outline(ax: plt.Axes, geometry: Any, *, fill: bool = False, alpha: float = 1.0) -> None:
    ax.add_patch(
        Rectangle(
            (GP.PAGE_X_MIN, GP.PAGE_Y_MIN),
            GP.A4_WIDTH_MM,
            GP.A4_HEIGHT_MM,
            fill=fill,
            facecolor=DESIGN_SYSTEM["page_fill"] if fill else "none",
            edgecolor=DESIGN_SYSTEM["page_edge"],
            linewidth=1.25,
            alpha=alpha,
        )
    )
    ax.add_patch(Circle((geometry.x0, geometry.y0), geometry.R, fill=False, edgecolor=DESIGN_SYSTEM["cylinder"], linewidth=1.7, alpha=alpha))


def representative_context(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    return max(contexts, key=lambda item: item["plane_width_mm"] * item["plane_height_mm"])


def pick_dense_valid_row(mask: MaskArray) -> int:
    valid_counts = np.sum(mask, axis=1)
    return int(np.argmax(valid_counts))


def select_row_samples(mask: MaskArray, row: int, sample_count: int) -> list[int]:
    valid_cols = np.flatnonzero(mask[row])
    if valid_cols.size == 0:
        raise RuntimeError("No valid virtual-plane samples found for binocular plotting.")
    count = min(sample_count, int(valid_cols.size))
    raw = np.linspace(0, valid_cols.size - 1, count)
    indices = np.unique(np.round(raw).astype(np.int64))
    return [int(valid_cols[idx]) for idx in indices]


def map_virtual_point_to_paper(geometry: Any, x_virtual: float, z_virtual: float) -> dict[str, float] | None:
    X = np.array([[x_virtual]], dtype=np.float64)
    Y = np.full_like(X, geometry.y_img, dtype=np.float64)
    Z = np.array([[z_virtual]], dtype=np.float64)
    mapping = GP.inverse_map_virtual_plane(X, Y, Z, geometry)
    clipped = GP.apply_page_clip(mapping["valid"], mapping["Ax"], mapping["Ay"], geometry, 0.0)
    if not bool(clipped[0, 0]):
        return None
    return {
        "Hx": float(mapping["Hx"][0, 0]),
        "Hy": float(mapping["Hy"][0, 0]),
        "Hz": float(mapping["Hz"][0, 0]),
        "Ax": float(mapping["Ax"][0, 0]),
        "Ay": float(mapping["Ay"][0, 0]),
        "X": float(x_virtual),
        "Z": float(z_virtual),
    }


def build_binocular_view_data(geometry: Any, contexts: list[dict[str, Any]], *, ipd_mm: float = DEFAULT_IPD_MM) -> dict[str, Any]:
    context = representative_context(contexts)
    row = pick_dense_valid_row(context["clipped_mask"])
    cols = select_row_samples(context["clipped_mask"], row, sample_count=9)
    left_geometry = geometry_with_overrides(geometry, xv=geometry.xv - ipd_mm / 2.0)
    right_geometry = geometry_with_overrides(geometry, xv=geometry.xv + ipd_mm / 2.0)
    samples: list[dict[str, Any]] = []
    for idx, col in enumerate(cols, start=1):
        x_virtual = float(context["mapping"]["X"][row, col])
        z_virtual = float(context["mapping"]["Z"][row, col])
        center = map_virtual_point_to_paper(geometry, x_virtual, z_virtual)
        left = map_virtual_point_to_paper(left_geometry, x_virtual, z_virtual)
        right = map_virtual_point_to_paper(right_geometry, x_virtual, z_virtual)
        if center is None or left is None or right is None:
            continue
        separation = float(np.hypot(left["Ax"] - right["Ax"], left["Ay"] - right["Ay"]))
        samples.append({"index": idx, "virtual": (x_virtual, z_virtual), "center": center, "left": left, "right": right, "paper_separation_mm": separation})
    if len(samples) < 3:
        raise RuntimeError("Failed to construct enough binocular samples for the selected geometry.")
    plane_x_min = geometry.x0 - context["plane_width_mm"] / 2.0
    plane_x_max = geometry.x0 + context["plane_width_mm"] / 2.0
    return {
        "context_name": context["spec"]["name"],
        "plane_height_mm": float(context["plane_height_mm"]),
        "plane_width_mm": float(context["plane_width_mm"]),
        "row": row,
        "samples": samples,
        "ipd_mm": float(ipd_mm),
        "plane_x_min": float(plane_x_min),
        "plane_x_max": float(plane_x_max),
    }


def compute_binocular_view_zone(geometry: Any, context: dict[str, Any], *, ipd_mm: float = DEFAULT_IPD_MM, size_px: int = BINOCULAR_ZONE_SIZE_PX) -> dict[str, Any]:
    x_vals = np.linspace(geometry.xv - 80.0, geometry.xv + 80.0, 33, dtype=np.float64)
    y_vals = np.linspace(geometry.yv - 70.0, geometry.yv + 55.0, 37, dtype=np.float64)
    reference = build_virtual_plane_mapping(geometry, size_px=size_px, plane_height_mm=context["plane_height_mm"])
    reference_fraction = float(np.mean(reference["clipped_mask"]))
    if reference_fraction <= 1.0e-8:
        raise RuntimeError("Reference viewing-zone mask is empty for the selected context.")

    joint_ratio = np.zeros((len(y_vals), len(x_vals)), dtype=np.float64)
    single_ratio = np.zeros_like(joint_ratio)
    for row_idx, y_center in enumerate(y_vals):
        for col_idx, x_center in enumerate(x_vals):
            left_geometry = geometry_with_overrides(geometry, xv=float(x_center - ipd_mm / 2.0), yv=float(y_center))
            right_geometry = geometry_with_overrides(geometry, xv=float(x_center + ipd_mm / 2.0), yv=float(y_center))
            left_mask = build_virtual_plane_mapping(left_geometry, size_px=size_px, plane_height_mm=context["plane_height_mm"])["clipped_mask"]
            right_mask = build_virtual_plane_mapping(right_geometry, size_px=size_px, plane_height_mm=context["plane_height_mm"])["clipped_mask"]
            joint_ratio[row_idx, col_idx] = float(np.mean(left_mask & right_mask) / reference_fraction)
            single_ratio[row_idx, col_idx] = float(0.5 * (np.mean(left_mask) + np.mean(right_mask)) / reference_fraction)

    current_col = int(np.argmin(np.abs(x_vals - geometry.xv)))
    current_row = int(np.argmin(np.abs(y_vals - geometry.yv)))
    recommended_mask = joint_ratio >= BINOCULAR_ZONE_THRESHOLD
    if np.any(recommended_mask):
        bbox = {
            "x_min": float(np.min(x_vals[np.any(recommended_mask, axis=0)])),
            "x_max": float(np.max(x_vals[np.any(recommended_mask, axis=0)])),
            "y_min": float(np.min(y_vals[np.any(recommended_mask, axis=1)])),
            "y_max": float(np.max(y_vals[np.any(recommended_mask, axis=1)])),
        }
    else:
        bbox = None
    peak_idx = np.unravel_index(int(np.argmax(joint_ratio)), joint_ratio.shape)
    return {
        "x_vals": x_vals,
        "y_vals": y_vals,
        "joint_ratio": joint_ratio,
        "single_ratio": single_ratio,
        "reference_fraction": reference_fraction,
        "threshold": BINOCULAR_ZONE_THRESHOLD,
        "selected_joint_ratio": float(joint_ratio[current_row, current_col]),
        "selected_single_ratio": float(single_ratio[current_row, current_col]),
        "max_joint_ratio": float(np.max(joint_ratio)),
        "best_center": {"xv": float(x_vals[peak_idx[1]]), "yv": float(y_vals[peak_idx[0]])},
        "recommended_bbox": bbox,
        "recommended_fraction_of_grid": float(np.mean(recommended_mask)),
    }


def build_spatial_mockup_example(geometry: Any) -> dict[str, Any]:
    library = build_symbol_library(size=320)
    target_key = "compass"
    plane_height_mm = 32.0
    target = library[target_key]["image"]
    inverse_result = inverse_design_from_mirror_target(target, geometry, dpi=200, plane_height_mm=plane_height_mm)
    return {
        "target_key": target_key,
        "target_label": library[target_key]["label"],
        "target_image": target,
        "plane_height_mm": plane_height_mm,
        "plane_width_mm": plane_height_mm * target.shape[1] / float(target.shape[0]),
        "paper_pattern": inverse_result["paper_pattern"],
        "reconstructed": inverse_result["reconstructed"],
        "metrics": inverse_result["metrics"],
        "honesty_label": "Synthetic scene visualization — not a photograph",
    }


def compute_q3_dof_summary(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not pair_results:
        raise RuntimeError("Problem-3 pair results are required for the DOF comparison figure.")
    paper_shape = list(pair_results[0]["paper_target"].shape[:2])
    mirror_shape = list(pair_results[0]["desired_mirror"].shape[:2])
    paper_scalar_dof = int(np.prod(paper_shape))
    mirror_scalar_dof = int(np.prod(mirror_shape))
    class_counts = {class_name: sum(1 for item in pair_results if item["compatibility_class"] == class_name) for class_name in CLASS_STYLES}
    return {
        "geometry_parameter_keys": list(GEOMETRY_PARAMETER_KEYS),
        "geometry_parameter_count": len(GEOMETRY_PARAMETER_KEYS),
        "paper_target_shape_px": paper_shape,
        "mirror_target_shape_px": mirror_shape,
        "paper_scalar_dof": paper_scalar_dof,
        "mirror_scalar_dof": mirror_scalar_dof,
        "single_image_to_geometry_ratio": float(mirror_scalar_dof / len(GEOMETRY_PARAMETER_KEYS)),
        "dual_image_to_geometry_ratio": float((paper_scalar_dof + mirror_scalar_dof) / len(GEOMETRY_PARAMETER_KEYS)),
        "pair_class_counts": class_counts,
        "pair_count": len(pair_results),
    }


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


def grayscale_image(image: FloatImageArray) -> FloatArray:
    if image.ndim == 2:
        return image.astype(np.float64)
    gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    return gray.astype(np.float64)


def gray_to_rgb(gray: FloatArray) -> FloatImageArray:
    clipped = np.clip(gray, 0.0, 1.0).astype(np.float32)
    return np.repeat(clipped[..., None], 3, axis=2)


def normalize01(values: FloatArray) -> FloatArray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin <= 1.0e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def grid_square(size: int = 320) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    coords = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return x, y, r, theta


def polygon_mask(x: FloatArray, y: FloatArray, vertices: list[tuple[float, float]]) -> MaskArray:
    points = np.column_stack([x.ravel(), y.ravel()])
    path = MplPath(vertices)
    return path.contains_points(points).reshape(x.shape)


def make_flower_icon(size: int = 320) -> FloatImageArray:
    x, y, r, theta = grid_square(size)
    gray = np.ones_like(x)
    center = r < 0.18
    petals = ((r > 0.18) & (r < 0.72) & (np.cos(6.0 * theta) > 0.18))
    ring = np.abs(r - 0.76) < 0.035
    accent = ((r > 0.40) & (r < 0.55) & (np.cos(12.0 * theta) > 0.75))
    gray[center | petals | ring | accent] = 0.10
    return gray_to_rgb(gray)


def make_compass_icon(size: int = 320) -> FloatImageArray:
    x, y, r, theta = grid_square(size)
    gray = np.ones_like(x)
    star = ((r > 0.16) & (r < 0.78) & (np.cos(4.0 * theta) ** 2 > 0.78))
    inner_ring = np.abs(r - 0.28) < 0.04
    outer_ring = np.abs(r - 0.80) < 0.03
    center = r < 0.12
    gray[star | inner_ring | outer_ring | center] = 0.10
    return gray_to_rgb(gray)


def make_fish_icon(size: int = 320) -> FloatImageArray:
    x, y, r, _ = grid_square(size)
    gray = np.ones_like(x)
    body = ((x + 0.08) / 0.56) ** 2 + (y / 0.34) ** 2 <= 1.0
    tail = polygon_mask(x, y, [(0.18, 0.0), (0.72, 0.34), (0.72, -0.34)])
    fin = polygon_mask(x, y, [(-0.02, 0.0), (0.22, 0.26), (0.10, 0.02)])
    eye = (x + 0.34) ** 2 + (y + 0.08) ** 2 <= 0.03 ** 2
    mouth = ((x + 0.60) ** 2 + (y * 1.4) ** 2 < 0.04 ** 2) & (x < -0.52)
    gray[body | tail | fin] = 0.12
    gray[eye | mouth] = 1.0
    return gray_to_rgb(gray)


def make_arrow_icon(size: int = 320) -> FloatImageArray:
    x, y, _, _ = grid_square(size)
    gray = np.ones_like(x)
    shaft = (np.abs(y) < 0.13) & (x > -0.72) & (x < 0.25)
    head = polygon_mask(x, y, [(0.20, 0.34), (0.78, 0.0), (0.20, -0.34)])
    tail = ((x + 0.62) ** 2 + (y / 1.2) ** 2 < 0.10 ** 2)
    gray[shaft | head | tail] = 0.10
    return gray_to_rgb(gray)


def make_letter_h_icon(size: int = 320) -> FloatImageArray:
    x, y, _, _ = grid_square(size)
    gray = np.ones_like(x)
    left_bar = (x > -0.68) & (x < -0.40) & (np.abs(y) < 0.78)
    right_bar = (x > 0.40) & (x < 0.68) & (np.abs(y) < 0.78)
    middle_bar = (np.abs(y) < 0.14) & (np.abs(x) < 0.68)
    border = (np.abs(x) < 0.85) & (np.abs(y) < 0.85) & ((np.abs(np.abs(x) - 0.85) < 0.03) | (np.abs(np.abs(y) - 0.85) < 0.03))
    gray[left_bar | right_bar | middle_bar | border] = 0.10
    return gray_to_rgb(gray)


def make_checker_icon(size: int = 320) -> FloatImageArray:
    x, y, r, _ = grid_square(size)
    gray = np.ones_like(x)
    checker = ((np.floor((x + 1.0) * 6.0) + np.floor((y + 1.0) * 6.0)) % 2 == 0) & (r < 0.82)
    ring = np.abs(r - 0.84) < 0.03
    gray[checker | ring] = 0.12
    return gray_to_rgb(gray)


def make_dense_stripe_icon(size: int = 320) -> FloatImageArray:
    x, y, r, _ = grid_square(size)
    gray = np.ones_like(x)
    stripes = np.sin(20.0 * np.pi * (0.55 * x + 0.15 * y)) > 0.0
    grid = np.sin(16.0 * np.pi * y) > 0.25
    texture = (stripes ^ grid) & (r < 0.86)
    gray[texture] = 0.12
    gray[np.abs(r - 0.86) < 0.03] = 0.12
    return gray_to_rgb(gray)


def build_symbol_library(size: int = 320) -> dict[str, dict[str, Any]]:
    return {
        "flower": {"label": "Floral pattern", "image": make_flower_icon(size), "class_hint": "radial"},
        "compass": {"label": "Compass emblem", "image": make_compass_icon(size), "class_hint": "radial"},
        "fish": {"label": "Fish silhouette", "image": make_fish_icon(size), "class_hint": "silhouette"},
        "arrow": {"label": "Arrow sign", "image": make_arrow_icon(size), "class_hint": "directional"},
        "letter_h": {"label": "Letter-like contour", "image": make_letter_h_icon(size), "class_hint": "block"},
        "checker": {"label": "Checker motif", "image": make_checker_icon(size), "class_hint": "checker"},
        "dense": {"label": "Dense texture", "image": make_dense_stripe_icon(size), "class_hint": "dense"},
    }


def page_coordinate_grid(height_px: int = 900) -> tuple[FloatArray, FloatArray]:
    width_px = int(round(height_px * GP.A4_WIDTH_MM / GP.A4_HEIGHT_MM))
    x = np.linspace(GP.PAGE_X_MIN, GP.PAGE_X_MAX, width_px, dtype=np.float64)
    y = np.linspace(GP.PAGE_Y_MAX, GP.PAGE_Y_MIN, height_px, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x, y)
    return grid_x, grid_y


def embed_square_image_on_page(
    square_image: FloatImageArray,
    geometry: Any,
    *,
    center_x_mm: float = 0.0,
    center_y_mm: float = -22.0,
    width_mm: float = 176.0,
    height_mm: float = 176.0,
    height_px: int = 900,
) -> FloatImageArray:
    X, Y = page_coordinate_grid(height_px)
    height_img, width_img = square_image.shape[:2]
    u = (X - center_x_mm) / (width_mm / 2.0)
    v = -(Y - center_y_mm) / (height_mm / 2.0)
    inside = (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0)

    x_idx = np.clip(np.round((u + 1.0) * 0.5 * (width_img - 1)).astype(np.int64), 0, width_img - 1)
    y_idx = np.clip(np.round((v + 1.0) * 0.5 * (height_img - 1)).astype(np.int64), 0, height_img - 1)

    page = np.ones((X.shape[0], X.shape[1], 3), dtype=np.float32)
    page[inside] = square_image[y_idx[inside], x_idx[inside]]

    blocked = (X - geometry.x0) ** 2 + (Y - geometry.y0) ** 2 <= (geometry.R + geometry.cylinder_clearance_mm) ** 2
    page[blocked] = 1.0
    return page


def build_virtual_plane_mapping(geometry: Any, *, size_px: int = 320, plane_height_mm: float = 32.0) -> dict[str, Any]:
    X, Y, Z, plane_width_mm = GP.build_virtual_plane(size_px, size_px, plane_height_mm, geometry)
    mapping = GP.inverse_map_virtual_plane(X, Y, Z, geometry)
    clipped_mask = GP.apply_page_clip(mapping["valid"], mapping["Ax"], mapping["Ay"], geometry, 0.0)
    return {
        "plane_width_mm": plane_width_mm,
        "plane_height_mm": plane_height_mm,
        "mapping": mapping,
        "clipped_mask": clipped_mask,
    }


def simulate_mirror_from_page(page_image: FloatImageArray, geometry: Any, *, size_px: int = 320, plane_height_mm: float = 32.0) -> tuple[FloatImageArray, MaskArray]:
    context = build_virtual_plane_mapping(geometry, size_px=size_px, plane_height_mm=plane_height_mm)
    mirror, mask = sample_page_image(page_image, context["mapping"]["Ax"], context["mapping"]["Ay"], context["clipped_mask"])
    return mirror, mask


def inverse_design_from_mirror_target(target_image: FloatImageArray, geometry: Any, *, dpi: int = 200, plane_height_mm: float = 32.0) -> dict[str, Any]:
    size_px = int(target_image.shape[0])
    context = build_virtual_plane_mapping(geometry, size_px=size_px, plane_height_mm=plane_height_mm)
    paper_pattern, occupancy = GP.render_paper_pattern(target_image, context["mapping"]["Ax"], context["mapping"]["Ay"], context["clipped_mask"], dpi)
    reconstructed, sample_mask = sample_page_image(paper_pattern, context["mapping"]["Ax"], context["mapping"]["Ay"], context["clipped_mask"])
    metrics = compute_reconstruction_metrics(target_image, reconstructed, sample_mask)
    return {
        "paper_pattern": paper_pattern,
        "occupancy": occupancy,
        "reconstructed": reconstructed,
        "sample_mask": sample_mask,
        "metrics": metrics,
    }


def global_ssim_gray(reference: FloatArray, estimate: FloatArray) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    x = reference.astype(np.float64)
    y = estimate.astype(np.float64)
    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    var_x = float(np.var(x))
    var_y = float(np.var(y))
    cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return float(numerator / denominator) if denominator > 0.0 else 1.0


def low_frequency_ratio(image: FloatImageArray) -> float:
    gray = grayscale_image(image)
    gray = gray - float(np.mean(gray))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    if float(np.sum(spectrum)) <= 1.0e-12:
        return 1.0
    yy, xx = np.indices(gray.shape)
    cy = (gray.shape[0] - 1) / 2.0
    cx = (gray.shape[1] - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = radius <= 0.16 * min(gray.shape)
    return float(np.sum(spectrum[mask]) / np.sum(spectrum))


def rotational_symmetry_score(image: FloatImageArray) -> float:
    gray = grayscale_image(image)
    denom = float(np.mean(np.abs(gray - np.mean(gray))) + 1.0e-6)
    rot180 = np.rot90(gray, 2)
    return float(np.clip(1.0 - np.mean(np.abs(gray - rot180)) / denom, 0.0, 1.0))


def axial_symmetry_score(image: FloatImageArray, axis: str = "vertical") -> float:
    gray = grayscale_image(image)
    denom = float(np.mean(np.abs(gray - np.mean(gray))) + 1.0e-6)
    if axis == "vertical":
        mirrored = np.fliplr(gray)
    elif axis == "horizontal":
        mirrored = np.flipud(gray)
    else:
        raise ValueError(f"Unsupported symmetry axis: {axis}")
    return float(np.clip(1.0 - np.mean(np.abs(gray - mirrored)) / denom, 0.0, 1.0))


def edge_density(image: FloatImageArray) -> float:
    gray = grayscale_image(image)
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx * gx + gy * gy)
    threshold = float(np.quantile(magnitude, 0.75))
    return float(np.mean(magnitude > threshold))


def foreground_fraction(image: FloatImageArray, threshold: float = 0.92) -> float:
    gray = grayscale_image(image)
    return float(np.mean(gray < threshold))


def tight_crop_nonwhite(image: FloatImageArray, *, threshold: float = 0.985, pad_px: int = 18) -> FloatImageArray:
    gray = grayscale_image(image)
    rows, cols = np.where(gray < threshold)
    if rows.size == 0 or cols.size == 0:
        return image
    row0 = max(int(rows.min()) - pad_px, 0)
    row1 = min(int(rows.max()) + pad_px + 1, image.shape[0])
    col0 = max(int(cols.min()) - pad_px, 0)
    col1 = min(int(cols.max()) + pad_px + 1, image.shape[1])
    return image[row0:row1, col0:col1]


def image_complexity(image: FloatImageArray) -> float:
    low_freq = low_frequency_ratio(image)
    edges = edge_density(image)
    return float(np.clip(0.55 * edges + 0.45 * (1.0 - low_freq), 0.0, 1.0))


def class_hint_label(class_hint: str) -> str:
    return {
        "radial": "radial family",
        "silhouette": "silhouette family",
        "directional": "directional family",
        "block": "block family",
        "checker": "checker family",
        "dense": "dense-texture family",
    }.get(class_hint, class_hint)


def free_design_assessment(key: str) -> str:
    return {
        "flower": "Mirror cue: bilateral floral emblem remains readable",
        "compass": "Mirror cue: star-like compass cue remains readable",
        "fish": "Mirror cue: coarse silhouette survives while fine detail softens",
        "arrow": "Limitation: the stroke remains, but the arrow direction weakens",
    }.get(key, "Mirror cue under forward reflection")


def compatibility_metrics(
    paper_target: FloatImageArray,
    desired_mirror: FloatImageArray,
    actual_mirror: FloatImageArray,
) -> dict[str, float]:
    paper_complexity = image_complexity(paper_target)
    mirror_complexity = image_complexity(desired_mirror)
    mirror_match = global_ssim_gray(grayscale_image(desired_mirror), grayscale_image(actual_mirror))
    low_freq_alignment = 1.0 - abs(low_frequency_ratio(paper_target) - low_frequency_ratio(desired_mirror))
    symmetry_alignment = np.sqrt(rotational_symmetry_score(paper_target) * rotational_symmetry_score(desired_mirror))
    complexity_alignment = 1.0 - abs(paper_complexity - mirror_complexity)
    overall = float(
        np.clip(
            COMPATIBILITY_WEIGHTS["mirror_match"] * mirror_match
            + COMPATIBILITY_WEIGHTS["low_freq_alignment"] * low_freq_alignment
            + COMPATIBILITY_WEIGHTS["symmetry_alignment"] * symmetry_alignment
            + COMPATIBILITY_WEIGHTS["complexity_alignment"] * complexity_alignment,
            0.0,
            1.0,
        )
    )
    return {
        "paper_complexity": float(paper_complexity),
        "mirror_complexity": float(mirror_complexity),
        "mirror_match": float(mirror_match),
        "low_freq_alignment": float(low_freq_alignment),
        "symmetry_alignment": float(symmetry_alignment),
        "complexity_alignment": float(complexity_alignment),
        "overall": overall,
    }


def compatibility_class(score: float) -> str:
    if score >= HIGH_OVERLAP_THRESHOLD:
        return "High overlap"
    if score >= MIXED_OVERLAP_THRESHOLD:
        return "Mixed overlap"
    return "Low overlap"


def show_page_panel(ax: plt.Axes, page_image: FloatImageArray, geometry: Any, title: str) -> None:
    extent = (GP.PAGE_X_MIN, GP.PAGE_X_MAX, GP.PAGE_Y_MIN, GP.PAGE_Y_MAX)
    ax.imshow(np.flipud(page_image), extent=extent, origin="lower")
    draw_page_outline(ax, geometry)
    style_panel(ax, title=title, equal=True, hide_ticks=True)


def save_problem2_free_design_gallery(geometry: Any, output_dir: Path) -> str:
    library = build_symbol_library(size=320)
    selected = FREE_DESIGN_CASES

    fig, axes = plt.subplots(2, len(selected), figsize=(4.9 * len(selected), 8.6), constrained_layout=True)
    for col, key in enumerate(selected):
        meta = library[key]
        square = meta["image"]
        page = embed_square_image_on_page(square, geometry)
        mirror, _ = simulate_mirror_from_page(page, geometry, size_px=square.shape[0], plane_height_mm=32.0)
        show_page_panel(
            axes[0, col],
            page,
            geometry,
            f"Paper-side source\n{meta['label']} ({class_hint_label(meta['class_hint'])})",
        )
        axes[1, col].imshow(np.clip(mirror, 0.0, 1.0))
        axes[1, col].set_title(
            f"{free_design_assessment(key)}\nBilateral sym.={axial_symmetry_score(mirror, axis='vertical'):.2f}, edge dens.={edge_density(mirror):.2f}",
            fontsize=10.5,
        )
        axes[1, col].axis("off")

    fig.suptitle(
        "Problem 2(a): free paper motifs can generate readable mirror cues, but directional symbols remain a limitation",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.01,
        "The previous 180° symmetry metric was replaced with bilateral symmetry, which better matches cylindrical mirror appearance; the arrow case is now labeled as a limitation rather than a success.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    output_path = output_dir / "q2_free_design_gallery.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path.name


def save_problem2_specified_gallery(geometry: Any, output_dir: Path) -> str:
    library = build_symbol_library(size=320)
    selected = SPECIFIED_MIRROR_CASES

    fig, axes = plt.subplots(4, len(selected), figsize=(4.8 * len(selected), 14.6), constrained_layout=True)
    for col, key in enumerate(selected):
        meta = library[key]
        target = meta["image"]
        inverse_result = inverse_design_from_mirror_target(target, geometry, dpi=200, plane_height_mm=32.0)
        paper_crop = tight_crop_nonwhite(inverse_result["paper_pattern"])

        axes[0, col].imshow(target)
        axes[0, col].set_title(
            f"Specified mirror target\n{meta['label']} ({class_hint_label(meta['class_hint'])})",
            fontsize=10.5,
        )
        axes[0, col].axis("off")

        show_page_panel(axes[1, col], inverse_result["paper_pattern"], geometry, "Inverse-designed paper print")

        axes[2, col].imshow(np.clip(paper_crop, 0.0, 1.0))
        axes[2, col].set_title(
            "Paper-side cue zoom\n"
            f"Ink frac.={foreground_fraction(paper_crop):.2f}, edge dens.={edge_density(paper_crop):.2f}",
            fontsize=10.5,
        )
        axes[2, col].axis("off")

        axes[3, col].imshow(np.clip(inverse_result["reconstructed"], 0.0, 1.0))
        axes[3, col].set_title(
            "Secondary sanity check: forward mirror reconstruction\n"
            f"SSIM={inverse_result['metrics']['global_ssim']:.3f}, PSNR={inverse_result['metrics']['psnr_db']:.2f} dB",
            fontsize=10.5,
        )
        axes[3, col].axis("off")

    fig.suptitle(
        "Problem 2(b): specified mirror targets lead to structured paper prints; round-trip metrics are shown only as a secondary model check",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.01,
        "The cue zoom highlights paper-side structure directly, so the figure no longer relies on same-pipeline SSIM/PSNR as its main evidence.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    output_path = output_dir / "q2_specified_mirror_gallery.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path.name


def build_problem3_pair_results(geometry: Any) -> list[dict[str, Any]]:
    library = build_symbol_library(size=320)
    results: list[dict[str, Any]] = []

    for paper_key in PROBLEM3_PAPER_KEYS:
        paper_square = library[paper_key]["image"]
        paper_page = embed_square_image_on_page(paper_square, geometry)
        simulated_mirror, _ = simulate_mirror_from_page(paper_page, geometry, size_px=paper_square.shape[0], plane_height_mm=32.0)
        for mirror_key in PROBLEM3_MIRROR_KEYS:
            desired_mirror = library[mirror_key]["image"]
            metrics = compatibility_metrics(paper_square, desired_mirror, simulated_mirror)
            results.append(
                {
                    "paper_key": paper_key,
                    "paper_label": library[paper_key]["label"],
                    "paper_target": paper_square,
                    "paper_page": paper_page,
                    "mirror_key": mirror_key,
                    "mirror_label": library[mirror_key]["label"],
                    "desired_mirror": desired_mirror,
                    "actual_mirror": simulated_mirror,
                    "metrics": metrics,
                    "compatibility_class": compatibility_class(metrics["overall"]),
                }
            )
    return results


def save_problem3_phase_diagram(pair_results: list[dict[str, Any]], output_dir: Path) -> str:
    x = np.array([item["metrics"]["paper_complexity"] for item in pair_results], dtype=np.float64)
    y = np.array([item["metrics"]["mirror_complexity"] for item in pair_results], dtype=np.float64)
    z = np.array([item["metrics"]["overall"] for item in pair_results], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 7.6), constrained_layout=True)
    scatter = None
    for class_name in ["Low overlap", "Mixed overlap", "High overlap"]:
        subset = [item for item in pair_results if item["compatibility_class"] == class_name]
        if not subset:
            continue
        scatter = ax.scatter(
            [item["metrics"]["paper_complexity"] for item in subset],
            [item["metrics"]["mirror_complexity"] for item in subset],
            c=[item["metrics"]["overall"] for item in subset],
            cmap="viridis",
            vmin=float(np.min(z)),
            vmax=float(np.max(z)),
            s=120,
            marker=CLASS_STYLES[class_name]["marker"],
            edgecolors=CLASS_STYLES[class_name]["color"],
            linewidths=1.1,
            alpha=0.95,
            label=f"{class_name} (n={len(subset)})",
        )

    diag_min = float(min(np.min(x), np.min(y)) - 0.01)
    diag_max = float(max(np.max(x), np.max(y)) + 0.01)
    ax.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", linewidth=1.0, color=DESIGN_SYSTEM["muted"], alpha=0.85)
    ax.text(diag_min + 0.005, diag_max - 0.01, "equal-complexity line", fontsize=9, color=DESIGN_SYSTEM["muted"])

    style_panel(
        ax,
        title="Problem 3: empirical compatibility map for the 6×6 paper–mirror sweep",
        xlabel="Paper-target complexity",
        ylabel="Mirror-target complexity",
        grid=True,
    )
    ax.legend(loc="lower left", frameon=True)
    if scatter is None:
        raise RuntimeError("No problem-3 compatibility points were generated.")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Compatibility index C")
    add_panel_badge(ax, "shared Q3 class palette", facecolor=DESIGN_SYSTEM["ink"])

    representatives = sorted(pair_results, key=lambda item: item["metrics"]["overall"])
    representative_indices = [0, len(representatives) // 2, len(representatives) - 1]
    for idx in representative_indices:
        item = representatives[idx]
        ax.annotate(
            f"{item['paper_key']}->{item['mirror_key']}\nC={item['metrics']['overall']:.2f}",
            (item["metrics"]["paper_complexity"], item["metrics"]["mirror_complexity"]),
            textcoords="offset points",
            xytext=(5, 6),
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )

    fig.text(
        0.5,
        0.01,
        "C = 0.25×mirror match + 0.25×low-frequency alignment + 0.25×symmetry alignment + 0.25×complexity alignment; classes use honest overlap labels: High ≥ 0.74, Mixed 0.68–0.74, Low < 0.68.",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    output_path = output_dir / "q3_compatibility_phase_diagram.png"
    return save_figure(fig, output_path)


def save_problem3_pairwise_heatmap(pair_results: list[dict[str, Any]], output_dir: Path) -> str:
    library = build_symbol_library(size=320)
    lookup = {(item["paper_key"], item["mirror_key"]): item for item in pair_results}
    scores = np.zeros((len(PROBLEM3_PAPER_KEYS), len(PROBLEM3_MIRROR_KEYS)), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(11, 6.8), constrained_layout=True)
    for row, paper_key in enumerate(PROBLEM3_PAPER_KEYS):
        for col, mirror_key in enumerate(PROBLEM3_MIRROR_KEYS):
            item = lookup[(paper_key, mirror_key)]
            scores[row, col] = item["metrics"]["overall"]

    heatmap = ax.imshow(scores, cmap="viridis", vmin=float(np.min(scores)), vmax=float(np.max(scores)))
    for row, paper_key in enumerate(PROBLEM3_PAPER_KEYS):
        for col, mirror_key in enumerate(PROBLEM3_MIRROR_KEYS):
            item = lookup[(paper_key, mirror_key)]
            score = item["metrics"]["overall"]
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor=CLASS_STYLES[item["compatibility_class"]]["color"],
                    linewidth=2.0,
                )
            )
            text_color = "white" if score >= float(np.mean(scores)) else "black"
            ax.text(col, row, f"{score:.2f}", ha="center", va="center", fontsize=10, color=text_color)

    ax.set_xticks(range(len(PROBLEM3_MIRROR_KEYS)), [library[key]["label"] for key in PROBLEM3_MIRROR_KEYS], rotation=24, ha="right")
    ax.set_yticks(range(len(PROBLEM3_PAPER_KEYS)), [library[key]["label"] for key in PROBLEM3_PAPER_KEYS])
    style_panel(
        ax,
        title="Problem 3: pairwise compatibility matrix for the sampled symbol library",
        xlabel="Specified mirror target",
        ylabel="Specified paper target",
    )
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04, label="Compatibility index C")
    legend_handles = [Rectangle((0, 0), 1, 1, fill=False, edgecolor=CLASS_STYLES[name]["color"], linewidth=2.0) for name in CLASS_STYLES]
    ax.legend(legend_handles, list(CLASS_STYLES.keys()), loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    add_panel_badge(ax, "same overlap thresholds", facecolor=DESIGN_SYSTEM["ink"])
    fig.text(
        0.5,
        0.01,
        "Cell colors show the numeric score; border colors show the overlap class under the same 0.74 / 0.68 thresholds used in the phase map.",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    output_path = output_dir / "q3_pairwise_compatibility_heatmap.png"
    return save_figure(fig, output_path)


def save_problem3_example_gallery(pair_results: list[dict[str, Any]], geometry: Any, output_dir: Path) -> str:
    _ = geometry
    selected: list[dict[str, Any]] = []
    for class_name in ["High overlap", "Mixed overlap", "Low overlap"]:
        subset = sorted(
            (item for item in pair_results if item["compatibility_class"] == class_name),
            key=lambda item: item["metrics"]["overall"],
        )
        if subset:
            selected.append(subset[len(subset) // 2])

    fig, axes = plt.subplots(len(selected), 4, figsize=(16, 4.4 * len(selected)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, item in enumerate(selected):
        axes[row, 0].imshow(item["paper_target"])
        axes[row, 0].set_title(
            f"{item['compatibility_class']} pair\nPaper target: {item['paper_label']}",
            fontsize=10.5,
        )
        axes[row, 0].axis("off")

        axes[row, 1].imshow(item["desired_mirror"])
        axes[row, 1].set_title(f"Desired mirror target: {item['mirror_label']}", fontsize=10.5)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(np.clip(item["actual_mirror"], 0.0, 1.0))
        axes[row, 2].set_title(
            "Actual mirror result from the paper target\n"
            f"{item['paper_key']}→{item['mirror_key']}, C={item['metrics']['overall']:.3f}",
            fontsize=10.5,
        )
        axes[row, 2].axis("off")

        names = ["Mirror match", "Low-freq.", "Symmetry", "Complexity"]
        values = [
            item["metrics"]["mirror_match"],
            item["metrics"]["low_freq_alignment"],
            item["metrics"]["symmetry_alignment"],
            item["metrics"]["complexity_alignment"],
        ]
        bars = axes[row, 3].barh(names, values, color=["#5b9bd5", "#70ad47", "#ffc000", "#c55a11"])
        axes[row, 3].set_xlim(0.0, 1.0)
        axes[row, 3].set_title("Compatibility components (equal 25% weights)", fontsize=10.5)
        axes[row, 3].grid(axis="x", linestyle="--", alpha=0.35)
        for bar, value in zip(bars, values, strict=True):
            axes[row, 3].text(min(value + 0.02, 0.98), bar.get_y() + bar.get_height() / 2.0, f"{value:.2f}", va="center", fontsize=9)
        axes[row, 3].text(
            0.02,
            0.04,
            f"class={item['compatibility_class']}",
            transform=axes[row, 3].transAxes,
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )

    fig.suptitle("Problem 3: representative pairs now use computed overlap classes instead of hardcoded labels", fontsize=15)

    output_path = output_dir / "q3_compatibility_examples.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path.name


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

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.2), constrained_layout=True)

    ax_top = axes[0]
    draw_page_outline(ax_top, geometry)
    plane_x_min = geometry.x0 - context["plane_width_mm"] / 2.0
    plane_x_max = geometry.x0 + context["plane_width_mm"] / 2.0
    ax_top.plot([plane_x_min, plane_x_max], [geometry.y_img, geometry.y_img], color=DESIGN_SYSTEM["virtual_plane"], linewidth=2.4)
    ax_top.scatter([geometry.xv], [geometry.yv], color=DESIGN_SYSTEM["viewer"], s=70)
    ax_top.text(geometry.xv + 3.0, geometry.yv - 10.0, "Observer V", color=DESIGN_SYSTEM["viewer"], fontsize=10)

    x_virtual = float(mapping["X"][top_row, top_col])
    y_hit = float(mapping["Hy"][top_row, top_col])
    x_hit = float(mapping["Hx"][top_row, top_col])
    x_paper = float(mapping["Ax"][top_row, top_col])
    y_paper = float(mapping["Ay"][top_row, top_col])
    ax_top.scatter([x_virtual], [geometry.y_img], color=DESIGN_SYSTEM["virtual_plane"], s=42)
    ax_top.scatter([x_hit], [y_hit], color=DESIGN_SYSTEM["cylinder"], s=42)
    ax_top.scatter([x_paper], [y_paper], color=DESIGN_SYSTEM["ink"], s=42)
    ax_top.plot([geometry.xv, x_hit], [geometry.yv, y_hit], color=DESIGN_SYSTEM["virtual_plane"], linestyle="--", linewidth=1.6)
    ax_top.plot([x_hit, x_paper], [y_hit, y_paper], color=DESIGN_SYSTEM["ray"], linewidth=1.8)
    ax_top.text(x_virtual + 2.0, geometry.y_img + 5.0, "I")
    ax_top.text(x_hit + 2.0, y_hit + 5.0, "H")
    ax_top.text(x_paper + 2.0, y_paper + 5.0, "A")
    style_panel(ax_top, title="Top view: page footprint, cylinder, virtual plane, and one reflection chain", xlabel="x / mm", ylabel="y / mm", equal=True, grid=True)
    add_panel_badge(ax_top, "A", facecolor=DESIGN_SYSTEM["ink"])
    ax_top.set_xlim(GP.PAGE_X_MIN - 10.0, GP.PAGE_X_MAX + 10.0)
    ax_top.set_ylim(min(GP.PAGE_Y_MIN, geometry.yv) - 20.0, GP.PAGE_Y_MAX + 15.0)

    ax_side = axes[1]
    ax_side.add_patch(Rectangle((geometry.y0 - geometry.R, 0.0), 2.0 * geometry.R, geometry.H, fill=True, facecolor=DESIGN_SYSTEM["cylinder_soft"], edgecolor=DESIGN_SYSTEM["cylinder"], linewidth=1.8, alpha=0.85))
    ax_side.axhline(0.0, color=DESIGN_SYSTEM["ink"], linewidth=1.2)
    ax_side.axvline(geometry.y_img, color=DESIGN_SYSTEM["virtual_plane"], linestyle="--", linewidth=1.5)
    z_min = geometry.z_img_center - context["plane_height_mm"] / 2.0
    z_max = geometry.z_img_center + context["plane_height_mm"] / 2.0
    ax_side.plot([geometry.y_img, geometry.y_img], [z_min, z_max], color=DESIGN_SYSTEM["virtual_plane"], linewidth=2.4)

    z_virtual = float(mapping["Z"][side_row, side_col])
    y_hit_side = float(mapping["Hy"][side_row, side_col])
    z_hit_side = float(mapping["Hz"][side_row, side_col])
    y_paper_side = float(mapping["Ay"][side_row, side_col])
    ax_side.scatter([geometry.yv], [geometry.zv], color=DESIGN_SYSTEM["viewer"], s=70)
    ax_side.scatter([geometry.y_img], [z_virtual], color=DESIGN_SYSTEM["virtual_plane"], s=42)
    ax_side.scatter([y_hit_side], [z_hit_side], color=DESIGN_SYSTEM["cylinder"], s=42)
    ax_side.scatter([y_paper_side], [0.0], color=DESIGN_SYSTEM["ink"], s=42)
    ax_side.plot([geometry.yv, y_hit_side], [geometry.zv, z_hit_side], color=DESIGN_SYSTEM["virtual_plane"], linestyle="--", linewidth=1.6)
    ax_side.plot([y_hit_side, y_paper_side], [z_hit_side, 0.0], color=DESIGN_SYSTEM["ray"], linewidth=1.8)
    ax_side.text(geometry.yv + 4.0, geometry.zv - 8.0, "V", color=DESIGN_SYSTEM["viewer"])
    ax_side.text(geometry.y_img + 3.0, z_virtual + 4.0, "I")
    ax_side.text(y_hit_side + 3.0, z_hit_side + 4.0, "H")
    ax_side.text(y_paper_side + 3.0, 4.0, "A")
    ax_side.annotate(f"R = {geometry.R:.1f} mm", xy=(geometry.y0 + geometry.R, geometry.H * 0.75), xytext=(geometry.y0 + geometry.R + 15.0, geometry.H * 0.82), arrowprops={"arrowstyle": "->", "linewidth": 1.0})
    ax_side.annotate(f"H = {geometry.H:.0f} mm", xy=(geometry.y0 + geometry.R + 2.0, geometry.H), xytext=(geometry.y0 + geometry.R + 18.0, geometry.H - 10.0), arrowprops={"arrowstyle": "->", "linewidth": 1.0})
    style_panel(ax_side, title="Side view: reflection from virtual image plane to printed paper", xlabel="y / mm", ylabel="z / mm", grid=True)
    add_panel_badge(ax_side, "B", facecolor=DESIGN_SYSTEM["ink"])
    ax_side.set_xlim(min(geometry.yv, GP.PAGE_Y_MIN) - 20.0, GP.PAGE_Y_MAX + 20.0)
    ax_side.set_ylim(-10.0, max(geometry.H, geometry.zv) + 20.0)

    fig.suptitle(
        "Cylindrical anamorphosis geometry with a unified ray-language legend",
        fontsize=14.5,
        fontweight="semibold",
    )
    add_figure_note(
        fig,
        f"Selected geometry from {report['base_report'] if 'base_report' in report else 'parameter_report'}: (x0, y0, R, H)=({geometry.x0:.0f}, {geometry.y0:.0f}, {geometry.R:.0f}, {geometry.H:.0f}) mm, observer V=({geometry.xv:.0f}, {geometry.yv:.0f}, {geometry.zv:.0f}) mm, virtual plane y_img={geometry.y_img:.0f} mm.",
    )

    output_path = output_dir / "geometry_schematic.png"
    return save_figure(fig, output_path)


def add_flow_box(ax: plt.Axes, center: tuple[float, float], text: str, box_color: str) -> None:
    width = 0.26
    height = 0.11
    x0 = center[0] - width / 2.0
    y0 = center[1] - height / 2.0
    patch = FancyBboxPatch((x0, y0), width, height, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor=box_color, edgecolor=DESIGN_SYSTEM["panel_edge"], linewidth=1.15)
    ax.add_patch(patch)
    ax.text(center[0], center[1], text, ha="center", va="center", fontsize=10, color=DESIGN_SYSTEM["ink"])


def add_flow_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.4, color=DESIGN_SYSTEM["ink"])
    ax.add_patch(arrow)


def save_workflow_flowchart(output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_facecolor(DESIGN_SYSTEM["figure_face"])

    boxes = {
        "input": ((0.20, 0.84), "Input target images\n+ grayscale / clustering / edge cues", FLOW_BOX_COLORS["input"]),
        "search": ((0.20, 0.60), "Search common geometry\n(score + A4 fit + manufacturability)", FLOW_BOX_COLORS["search"]),
        "map": ((0.50, 0.60), "Inverse map I -> H -> A\nusing cylinder reflection law", FLOW_BOX_COLORS["map"]),
        "raster": ((0.80, 0.60), "Rasterize paper pattern\nand enforce page / cylinder constraints", FLOW_BOX_COLORS["raster"]),
        "reconstruct": ((0.50, 0.33), "Forward simulate mirror image\nfor closed-loop verification", FLOW_BOX_COLORS["reconstruct"]),
        "validate": ((0.80, 0.33), "Output metrics + case studies\nRMSE / PSNR / SSIM / compatibility", FLOW_BOX_COLORS["validate"]),
    }
    for center, text, color in boxes.values():
        add_flow_box(ax, center, text, color)

    add_flow_arrow(ax, (0.20, 0.78), (0.20, 0.66))
    add_flow_arrow(ax, (0.33, 0.60), (0.37, 0.60))
    add_flow_arrow(ax, (0.63, 0.60), (0.67, 0.60))
    add_flow_arrow(ax, (0.50, 0.54), (0.50, 0.39))
    add_flow_arrow(ax, (0.63, 0.33), (0.67, 0.33))
    add_flow_arrow(ax, (0.80, 0.54), (0.80, 0.39))

    ax.text(
        0.50,
        0.12,
        "One inverse-reflection engine supports problem 1 reconstruction, problem 2 dual-meaning synthesis, and problem 3 compatibility evidence.",
        ha="center",
        va="center",
        fontsize=10,
        color=DESIGN_SYSTEM["muted"],
        bbox={"boxstyle": "round,pad=0.32", "facecolor": DESIGN_SYSTEM["panel_face"], "edgecolor": DESIGN_SYSTEM["panel_edge"]},
    )
    ax.set_title("Overall modeling and algorithm workflow", fontsize=14.5, fontweight="semibold", color=DESIGN_SYSTEM["ink"])
    add_panel_badge(ax, "shared editorial palette", facecolor=DESIGN_SYSTEM["ink"])

    output_path = output_dir / "algorithm_workflow_flowchart.png"
    return save_figure(fig, output_path)


def save_binocular_viewing_geometry(geometry: Any, binocular_data: dict[str, Any], output_dir: Path) -> str:
    samples = binocular_data["samples"]
    ipd_mm = binocular_data["ipd_mm"]
    left_eye = (geometry.xv - ipd_mm / 2.0, geometry.yv)
    right_eye = (geometry.xv + ipd_mm / 2.0, geometry.yv)
    mean_sep = float(np.mean([item["paper_separation_mm"] for item in samples]))
    max_sep = float(np.max([item["paper_separation_mm"] for item in samples]))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.1), constrained_layout=True)

    ax_top = axes[0]
    draw_page_outline(ax_top, geometry, fill=True, alpha=0.82)
    ax_top.plot(
        [binocular_data["plane_x_min"], binocular_data["plane_x_max"]],
        [geometry.y_img, geometry.y_img],
        color=DESIGN_SYSTEM["virtual_plane"],
        linewidth=2.5,
    )
    ax_top.add_patch(
        Ellipse(
            (geometry.xv, geometry.yv),
            ipd_mm + 18.0,
            19.0,
            facecolor=DESIGN_SYSTEM["viewer_soft"],
            edgecolor=DESIGN_SYSTEM["viewer"],
            linewidth=1.4,
            alpha=0.95,
        )
    )
    ax_top.scatter([left_eye[0], right_eye[0]], [left_eye[1], right_eye[1]], color=DESIGN_SYSTEM["viewer"], s=58, zorder=4)
    ax_top.text(left_eye[0] - 8.0, left_eye[1] - 12.0, "L eye", color=DESIGN_SYSTEM["viewer"], fontsize=9.2)
    ax_top.text(right_eye[0] - 5.0, right_eye[1] - 12.0, "R eye", color=DESIGN_SYSTEM["viewer"], fontsize=9.2)
    ax_top.annotate(
        f"typical binocular baseline ≈ {ipd_mm:.0f} mm",
        xy=((left_eye[0] + right_eye[0]) / 2.0, geometry.yv + 8.0),
        xytext=(geometry.xv + 28.0, geometry.yv - 28.0),
        fontsize=9.2,
        color=DESIGN_SYSTEM["muted"],
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": DESIGN_SYSTEM["muted"]},
    )
    for item in samples:
        ax_top.scatter(item["virtual"][0], geometry.y_img, s=22, facecolor=DESIGN_SYSTEM["virtual_plane_soft"], edgecolor=DESIGN_SYSTEM["virtual_plane"], linewidth=0.8)
    for item in [samples[0], samples[len(samples) // 2], samples[-1]]:
        for eye_name, eye_point, eye_color in [
            ("left", left_eye, DESIGN_SYSTEM["viewer"]),
            ("right", right_eye, DESIGN_SYSTEM["ray"]),
        ]:
            hit = item[eye_name]
            ax_top.plot([eye_point[0], hit["Hx"]], [eye_point[1], hit["Hy"]], linestyle="--", linewidth=1.35, color=eye_color, alpha=0.95)
            ax_top.plot([hit["Hx"], hit["Ax"]], [hit["Hy"], hit["Ay"]], linewidth=1.5, color=DESIGN_SYSTEM["ray"], alpha=0.88)
            ax_top.scatter([hit["Hx"]], [hit["Hy"]], s=25, color=DESIGN_SYSTEM["cylinder"], zorder=4)
    style_panel(
        ax_top,
        title="Top view: two-eye viewing geometry over the selected observer position",
        xlabel="x / mm",
        ylabel="y / mm",
        equal=True,
        grid=True,
    )
    add_panel_badge(ax_top, "A", facecolor=DESIGN_SYSTEM["ink"])
    add_panel_badge(ax_top, "schematic binocular extension", facecolor=DESIGN_SYSTEM["viewer"], anchor=(0.02, 0.88))
    ax_top.set_xlim(GP.PAGE_X_MIN - 16.0, GP.PAGE_X_MAX + 16.0)
    ax_top.set_ylim(min(GP.PAGE_Y_MIN, geometry.yv) - 32.0, GP.PAGE_Y_MAX + 16.0)

    ax_page = axes[1]
    draw_page_outline(ax_page, geometry, fill=True, alpha=0.82)
    for item in samples:
        ax_page.plot(
            [item["left"]["Ax"], item["right"]["Ax"]],
            [item["left"]["Ay"], item["right"]["Ay"]],
            color=DESIGN_SYSTEM["grid"],
            linewidth=1.0,
            alpha=0.9,
        )
        ax_page.scatter(item["left"]["Ax"], item["left"]["Ay"], s=52, color=DESIGN_SYSTEM["viewer"], edgecolors="white", linewidths=0.9, zorder=4)
        ax_page.scatter(item["right"]["Ax"], item["right"]["Ay"], s=52, color=DESIGN_SYSTEM["ray"], edgecolors="white", linewidths=0.9, zorder=4)
        ax_page.scatter(item["center"]["Ax"], item["center"]["Ay"], s=22, color=DESIGN_SYSTEM["ink"], zorder=5)
        ax_page.text(item["center"]["Ax"] + 2.0, item["center"]["Ay"] + 3.0, str(item["index"]), fontsize=8.2, color=DESIGN_SYSTEM["muted"])
    style_panel(
        ax_page,
        title="Paper-side sampling footprint for nine representative virtual-plane points",
        xlabel="x / mm",
        ylabel="y / mm",
        equal=True,
        grid=True,
    )
    add_panel_badge(ax_page, "B", facecolor=DESIGN_SYSTEM["ink"])
    ax_page.legend(
        [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=DESIGN_SYSTEM["viewer"], markeredgecolor="white", markersize=8, label="Left eye hit"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=DESIGN_SYSTEM["ray"], markeredgecolor="white", markersize=8, label="Right eye hit"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=DESIGN_SYSTEM["ink"], markersize=5, label="Center-eye reference"),
        ],
        ["Left eye hit", "Right eye hit", "Center-eye reference"],
        loc="lower left",
    )
    ax_page.text(
        0.02,
        0.05,
        f"mean eye-to-eye paper shift = {mean_sep:.1f} mm\nmax sampled shift = {max_sep:.1f} mm",
        transform=ax_page.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        color=DESIGN_SYSTEM["ink"],
        bbox={"boxstyle": "round,pad=0.28", "facecolor": DESIGN_SYSTEM["panel_face"], "edgecolor": DESIGN_SYSTEM["panel_edge"]},
    )
    ax_page.set_xlim(GP.PAGE_X_MIN - 14.0, GP.PAGE_X_MAX + 14.0)
    ax_page.set_ylim(GP.PAGE_Y_MIN - 16.0, GP.PAGE_Y_MAX + 16.0)

    fig.suptitle("Binocular viewing geometry around the selected observer", fontsize=14.6, fontweight="semibold")
    add_figure_note(
        fig,
        f"Two-eye overlay uses the selected geometry plus a standard {ipd_mm:.0f} mm interpupillary distance; the main inverse-design solver remains a single-viewpoint model.",
    )
    output_path = output_dir / "binocular_viewing_geometry.png"
    return save_figure(fig, output_path)


def save_binocular_viewing_zone(
    geometry: Any,
    binocular_data: dict[str, Any],
    zone_data: dict[str, Any],
    output_dir: Path,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.6, 1.0]})

    ax_zone = axes[0]
    x_vals = zone_data["x_vals"]
    y_vals = zone_data["y_vals"]
    joint_ratio = zone_data["joint_ratio"]
    heat = ax_zone.imshow(
        joint_ratio,
        extent=(float(x_vals[0]), float(x_vals[-1]), float(y_vals[0]), float(y_vals[-1])),
        origin="lower",
        aspect="auto",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=max(1.0, float(np.max(joint_ratio))),
    )
    contour_levels = [level for level in [0.75, zone_data["threshold"]] if float(np.max(joint_ratio)) >= level]
    if contour_levels:
        contours = ax_zone.contour(x_vals, y_vals, joint_ratio, levels=contour_levels, colors=[DESIGN_SYSTEM["ray"], DESIGN_SYSTEM["success"]][: len(contour_levels)], linewidths=1.8)
        ax_zone.clabel(contours, fmt={level: f"{level:.2f}" for level in contour_levels}, fontsize=8.5)
    ax_zone.scatter([geometry.xv], [geometry.yv], marker="x", s=95, linewidths=2.0, color=DESIGN_SYSTEM["ink"])
    ax_zone.scatter([zone_data["best_center"]["xv"]], [zone_data["best_center"]["yv"]], marker="*", s=120, color=DESIGN_SYSTEM["ray"], edgecolors="white", linewidths=0.7)
    style_panel(
        ax_zone,
        title="Binocular viewing zone from shared valid-ray coverage",
        xlabel="Head-center x_v / mm",
        ylabel="Head-center y_v / mm",
        grid=True,
    )
    add_panel_badge(ax_zone, "geometric overlap only", facecolor=DESIGN_SYSTEM["ink"])
    fig.colorbar(heat, ax=ax_zone, fraction=0.046, pad=0.04, label="Joint visible fraction / selected-view reference")

    ax_stats = axes[1]
    labels = ["Selected binocular overlap", "Selected mean single-eye visibility", "Best binocular overlap"]
    values = [zone_data["selected_joint_ratio"], zone_data["selected_single_ratio"], zone_data["max_joint_ratio"]]
    colors = [DESIGN_SYSTEM["viewer"], DESIGN_SYSTEM["virtual_plane"], DESIGN_SYSTEM["success"]]
    y_pos = np.arange(len(labels), dtype=np.float64)
    bars = ax_stats.barh(y_pos, values, color=colors, edgecolor=DESIGN_SYSTEM["panel_edge"], linewidth=1.0)
    ax_stats.axvline(zone_data["threshold"], color=DESIGN_SYSTEM["ray"], linestyle="--", linewidth=1.4)
    ax_stats.set_yticks(y_pos, labels)
    ax_stats.invert_yaxis()
    ax_stats.set_xlim(0.0, max(1.05, float(max(values)) + 0.08))
    style_panel(ax_stats, title="Selected-view summary", xlabel="Normalized visible fraction", grid=True)
    add_panel_badge(ax_stats, "B", facecolor=DESIGN_SYSTEM["ink"])
    for bar, value in zip(bars, values, strict=True):
        ax_stats.text(min(value + 0.02, ax_stats.get_xlim()[1] - 0.04), bar.get_y() + bar.get_height() / 2.0, f"{value:.2f}", va="center", fontsize=9.2, color=DESIGN_SYSTEM["ink"])
    recommended_text = "No contiguous zone above threshold"
    if zone_data["recommended_bbox"] is not None:
        bbox = zone_data["recommended_bbox"]
        recommended_text = (
            f"Recommended center range (joint ≥ {zone_data['threshold']:.2f})\n"
            f"x_v ≈ [{bbox['x_min']:.0f}, {bbox['x_max']:.0f}] mm\n"
            f"y_v ≈ [{bbox['y_min']:.0f}, {bbox['y_max']:.0f}] mm"
        )
    ax_stats.text(
        0.02,
        0.08,
        f"Reference context: {binocular_data['context_name']}\nRepresentative plane: {binocular_data['plane_width_mm']:.1f} × {binocular_data['plane_height_mm']:.1f} mm\n{recommended_text}",
        transform=ax_stats.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.1,
        color=DESIGN_SYSTEM["ink"],
        bbox={"boxstyle": "round,pad=0.32", "facecolor": DESIGN_SYSTEM["panel_face"], "edgecolor": DESIGN_SYSTEM["panel_edge"]},
    )

    fig.suptitle("Binocular viewing zone around the selected observer", fontsize=14.6, fontweight="semibold")
    add_figure_note(
        fig,
        f"The heatmap reports pure geometric overlap for two eyes separated by {binocular_data['ipd_mm']:.0f} mm; it is an honest visibility proxy rather than a psychophysical comfort measurement.",
    )
    output_path = output_dir / "binocular_viewing_zone.png"
    return save_figure(fig, output_path)


def scene_project(points: NDArray[np.float64]) -> FloatArray:
    pts = np.asarray(points, dtype=np.float64)
    u = pts[..., 0] + 0.58 * pts[..., 1]
    v = pts[..., 2] + 0.22 * pts[..., 1]
    return np.stack([u, v], axis=-1)


def add_projected_image(
    ax: plt.Axes,
    image: FloatImageArray,
    lower_left: tuple[float, float, float],
    lower_right: tuple[float, float, float],
    upper_left: tuple[float, float, float],
    *,
    zorder: float,
    alpha: float = 1.0,
) -> None:
    p00 = scene_project(np.array(lower_left, dtype=np.float64))
    p10 = scene_project(np.array(lower_right, dtype=np.float64))
    p01 = scene_project(np.array(upper_left, dtype=np.float64))
    vec_u = p10 - p00
    vec_v = p01 - p00
    transform = Affine2D.from_values(vec_u[0], vec_u[1], vec_v[0], vec_v[1], p00[0], p00[1])
    ax.imshow(
        np.flipud(image),
        extent=(0.0, 1.0, 0.0, 1.0),
        origin="lower",
        interpolation="bilinear",
        transform=transform + ax.transData,
        zorder=zorder,
        alpha=alpha,
    )


def draw_scene_cylinder(ax: plt.Axes, geometry: Any) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 240, dtype=np.float64)
    theta_front = np.linspace(-np.pi, 0.0, 140, dtype=np.float64)
    theta_back = np.linspace(0.0, np.pi, 140, dtype=np.float64)

    def ring_points(angle_values: FloatArray, z_value: float) -> FloatArray:
        return np.column_stack(
            [
                geometry.x0 + geometry.R * np.cos(angle_values),
                geometry.y0 + geometry.R * np.sin(angle_values),
                np.full_like(angle_values, z_value),
            ]
        )

    front_base = ring_points(theta_front, 0.0)
    front_top = ring_points(theta_front, geometry.H)
    back_base = ring_points(theta_back, 0.0)
    back_top = ring_points(theta_back, geometry.H)
    top_full = ring_points(theta, geometry.H)

    side_wall = np.vstack([scene_project(front_base), scene_project(front_top[::-1])])
    ax.add_patch(Polygon(side_wall, closed=True, facecolor=DESIGN_SYSTEM["cylinder_soft"], edgecolor=DESIGN_SYSTEM["cylinder"], linewidth=1.4, alpha=0.95, zorder=2.6))
    ax.add_patch(Polygon(scene_project(top_full), closed=True, facecolor="#f5ddd6", edgecolor=DESIGN_SYSTEM["cylinder"], linewidth=1.4, alpha=0.96, zorder=3.2))
    ax.plot(*scene_project(back_base).T, linestyle="--", linewidth=1.0, color=DESIGN_SYSTEM["cylinder"], alpha=0.35, zorder=2.4)
    ax.plot(*scene_project(front_base).T, linewidth=1.2, color=DESIGN_SYSTEM["cylinder"], alpha=0.8, zorder=2.7)
    ax.plot(*scene_project(back_top).T, linestyle="--", linewidth=1.0, color=DESIGN_SYSTEM["cylinder"], alpha=0.45, zorder=3.0)


def draw_scene_observer(ax: plt.Axes, geometry: Any, *, ipd_mm: float = DEFAULT_IPD_MM) -> None:
    head_center = scene_project(np.array([geometry.xv, geometry.yv, geometry.zv + 10.0], dtype=np.float64))
    left_eye = scene_project(np.array([geometry.xv - ipd_mm / 2.0, geometry.yv, geometry.zv], dtype=np.float64))
    right_eye = scene_project(np.array([geometry.xv + ipd_mm / 2.0, geometry.yv, geometry.zv], dtype=np.float64))
    shoulder_poly = scene_project(
        np.array(
            [
                [geometry.xv - 22.0, geometry.yv + 6.0, geometry.zv - 48.0],
                [geometry.xv + 22.0, geometry.yv + 6.0, geometry.zv - 48.0],
                [geometry.xv + 12.0, geometry.yv - 4.0, geometry.zv - 26.0],
                [geometry.xv - 12.0, geometry.yv - 4.0, geometry.zv - 26.0],
            ],
            dtype=np.float64,
        )
    )
    ax.add_patch(Polygon(shoulder_poly, closed=True, facecolor=DESIGN_SYSTEM["viewer_soft"], edgecolor=DESIGN_SYSTEM["viewer"], linewidth=1.1, alpha=0.96, zorder=4.0))
    ax.add_patch(
        Ellipse(
            (float(head_center[0]), float(head_center[1])),
            width=22.0,
            height=28.0,
            facecolor=DESIGN_SYSTEM["viewer_soft"],
            edgecolor=DESIGN_SYSTEM["viewer"],
            linewidth=1.3,
            alpha=0.98,
            zorder=4.2,
        )
    )
    ax.scatter([left_eye[0], right_eye[0]], [left_eye[1], right_eye[1]], s=16, color=DESIGN_SYSTEM["viewer"], zorder=4.3)


def save_spatial_experiment_mockup(geometry: Any, mockup_example: dict[str, Any], output_dir: Path) -> str:
    virtual_context = build_virtual_plane_mapping(
        geometry,
        size_px=mockup_example["target_image"].shape[0],
        plane_height_mm=mockup_example["plane_height_mm"],
    )
    samples = build_binocular_view_data(
        geometry,
        [
            {
                "spec": {"name": mockup_example["target_key"]},
                "plane_height_mm": mockup_example["plane_height_mm"],
                "plane_width_mm": mockup_example["plane_width_mm"],
                "mapping": virtual_context["mapping"],
                "clipped_mask": virtual_context["clipped_mask"],
            }
        ],
        ipd_mm=DEFAULT_IPD_MM,
    )["samples"]

    fig, ax = plt.subplots(figsize=(13.8, 7.8))
    ax.set_facecolor(DESIGN_SYSTEM["figure_face"])

    x_grid = np.arange(-120.0, 121.0, 30.0)
    y_grid = np.arange(-240.0, 161.0, 35.0)
    for x_value in x_grid:
        line = scene_project(np.array([[x_value, float(y_grid[0]), 0.0], [x_value, float(y_grid[-1]), 0.0]], dtype=np.float64))
        ax.plot(line[:, 0], line[:, 1], color=DESIGN_SYSTEM["grid"], linewidth=0.85, alpha=0.5, zorder=0.8)
    for y_value in y_grid:
        line = scene_project(np.array([[float(x_grid[0]), y_value, 0.0], [float(x_grid[-1]), y_value, 0.0]], dtype=np.float64))
        ax.plot(line[:, 0], line[:, 1], color=DESIGN_SYSTEM["grid"], linewidth=0.85, alpha=0.5, zorder=0.8)

    page_lower_left = (GP.PAGE_X_MIN, GP.PAGE_Y_MIN, 0.0)
    page_lower_right = (GP.PAGE_X_MAX, GP.PAGE_Y_MIN, 0.0)
    page_upper_left = (GP.PAGE_X_MIN, GP.PAGE_Y_MAX, 0.0)
    add_projected_image(ax, mockup_example["paper_pattern"], page_lower_left, page_lower_right, page_upper_left, zorder=1.2)
    page_outline = scene_project(
        np.array(
            [
                [GP.PAGE_X_MIN, GP.PAGE_Y_MIN, 0.0],
                [GP.PAGE_X_MAX, GP.PAGE_Y_MIN, 0.0],
                [GP.PAGE_X_MAX, GP.PAGE_Y_MAX, 0.0],
                [GP.PAGE_X_MIN, GP.PAGE_Y_MAX, 0.0],
            ],
            dtype=np.float64,
        )
    )
    ax.add_patch(Polygon(page_outline, closed=True, fill=False, edgecolor=DESIGN_SYSTEM["page_edge"], linewidth=1.4, zorder=1.6))

    plane_half_width = mockup_example["plane_width_mm"] / 2.0
    plane_half_height = mockup_example["plane_height_mm"] / 2.0
    plane_lower_left = (geometry.x0 - plane_half_width, geometry.y_img, geometry.z_img_center - plane_half_height)
    plane_lower_right = (geometry.x0 + plane_half_width, geometry.y_img, geometry.z_img_center - plane_half_height)
    plane_upper_left = (geometry.x0 - plane_half_width, geometry.y_img, geometry.z_img_center + plane_half_height)
    add_projected_image(ax, mockup_example["reconstructed"], plane_lower_left, plane_lower_right, plane_upper_left, zorder=3.4, alpha=0.98)
    plane_outline = scene_project(
        np.array(
            [
                [geometry.x0 - plane_half_width, geometry.y_img, geometry.z_img_center - plane_half_height],
                [geometry.x0 + plane_half_width, geometry.y_img, geometry.z_img_center - plane_half_height],
                [geometry.x0 + plane_half_width, geometry.y_img, geometry.z_img_center + plane_half_height],
                [geometry.x0 - plane_half_width, geometry.y_img, geometry.z_img_center + plane_half_height],
            ],
            dtype=np.float64,
        )
    )
    ax.add_patch(Polygon(plane_outline, closed=True, fill=False, edgecolor=DESIGN_SYSTEM["virtual_plane"], linewidth=1.6, zorder=3.6))
    ax.add_patch(Polygon(plane_outline, closed=True, facecolor=DESIGN_SYSTEM["virtual_plane_soft"], edgecolor="none", alpha=0.18, zorder=3.1))

    draw_scene_cylinder(ax, geometry)
    draw_scene_observer(ax, geometry)

    for item in [samples[1], samples[len(samples) // 2], samples[-2]]:
        virtual_pt = scene_project(np.array([item["virtual"][0], geometry.y_img, item["virtual"][1]], dtype=np.float64))
        hit_pt = scene_project(np.array([item["center"]["Hx"], item["center"]["Hy"], item["center"]["Hz"]], dtype=np.float64))
        paper_pt = scene_project(np.array([item["center"]["Ax"], item["center"]["Ay"], 0.0], dtype=np.float64))
        viewer_pt = scene_project(np.array([geometry.xv, geometry.yv, geometry.zv], dtype=np.float64))
        ax.plot([viewer_pt[0], hit_pt[0]], [viewer_pt[1], hit_pt[1]], linestyle="--", linewidth=1.3, color=DESIGN_SYSTEM["virtual_plane"], alpha=0.88, zorder=4.4)
        ax.plot([hit_pt[0], paper_pt[0]], [hit_pt[1], paper_pt[1]], linewidth=1.5, color=DESIGN_SYSTEM["ray"], alpha=0.92, zorder=4.4)
        ax.plot([hit_pt[0], virtual_pt[0]], [hit_pt[1], virtual_pt[1]], linewidth=1.0, color=DESIGN_SYSTEM["virtual_plane"], alpha=0.34, zorder=3.8)
        ax.scatter([hit_pt[0]], [hit_pt[1]], s=20, color=DESIGN_SYSTEM["cylinder"], zorder=4.6)

    plane_anchor = scene_project(np.array([geometry.x0 + plane_half_width, geometry.y_img, geometry.z_img_center + plane_half_height], dtype=np.float64))
    paper_anchor = scene_project(np.array([GP.PAGE_X_MAX, GP.PAGE_Y_MIN + 45.0, 0.0], dtype=np.float64))
    cylinder_anchor = scene_project(np.array([geometry.x0 + geometry.R, geometry.y0, geometry.H * 0.78], dtype=np.float64))
    observer_anchor = scene_project(np.array([geometry.xv, geometry.yv, geometry.zv + 20.0], dtype=np.float64))
    ax.annotate("Perceived mirror result\n(virtual image plane)", xy=plane_anchor, xytext=(plane_anchor[0] + 68.0, plane_anchor[1] + 42.0), fontsize=10.0, color=DESIGN_SYSTEM["virtual_plane"], arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": DESIGN_SYSTEM["virtual_plane"]}, zorder=5.2)
    ax.annotate("Printed paper pattern", xy=paper_anchor, xytext=(paper_anchor[0] + 50.0, paper_anchor[1] - 24.0), fontsize=10.0, color=DESIGN_SYSTEM["page_edge"], arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": DESIGN_SYSTEM["page_edge"]}, zorder=5.2)
    ax.annotate("Cylinder mirror", xy=cylinder_anchor, xytext=(cylinder_anchor[0] + 44.0, cylinder_anchor[1] + 10.0), fontsize=10.0, color=DESIGN_SYSTEM["cylinder"], arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": DESIGN_SYSTEM["cylinder"]}, zorder=5.2)
    ax.annotate("Observer", xy=observer_anchor, xytext=(observer_anchor[0] - 88.0, observer_anchor[1] + 26.0), fontsize=10.0, color=DESIGN_SYSTEM["viewer"], arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": DESIGN_SYSTEM["viewer"]}, zorder=5.2)

    add_panel_badge(ax, mockup_example["honesty_label"], facecolor=DESIGN_SYSTEM["danger"])
    ax.text(
        0.02,
        0.90,
        f"Example target: {mockup_example['target_label']}\nRound-trip SSIM={mockup_example['metrics']['global_ssim']:.3f}, PSNR={mockup_example['metrics']['psnr_db']:.1f} dB",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color=DESIGN_SYSTEM["ink"],
        bbox={"boxstyle": "round,pad=0.32", "facecolor": DESIGN_SYSTEM["panel_face"], "edgecolor": DESIGN_SYSTEM["panel_edge"]},
    )
    ax.set_title("Synthetic spatial experiment mockup for the cylindrical-anamorphosis setup", loc="left", fontsize=14.4, fontweight="semibold", color=DESIGN_SYSTEM["ink"])
    ax.axis("off")

    bound_points = np.vstack(
        [
            page_outline,
            plane_outline,
            scene_project(np.array([[geometry.xv, geometry.yv, geometry.zv + 20.0]], dtype=np.float64)),
            scene_project(np.array([[geometry.x0 + geometry.R, geometry.y0, geometry.H]], dtype=np.float64)),
        ]
    )
    pad_x = 55.0
    pad_y = 42.0
    ax.set_xlim(float(np.min(bound_points[:, 0]) - pad_x), float(np.max(bound_points[:, 0]) + pad_x))
    ax.set_ylim(float(np.min(bound_points[:, 1]) - pad_y), float(np.max(bound_points[:, 1]) + pad_y))

    output_path = output_dir / "spatial_experiment_mockup.png"
    return save_figure(fig, output_path)


def save_q3_dof_constraint_comparison(q3_dof_summary: dict[str, Any], output_dir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax_scale = axes[0]
    labels = ["Geometry\nparameters θ", "One specified\nimage proxy", "Dual specified\nimage proxy"]
    values = [
        q3_dof_summary["geometry_parameter_count"],
        q3_dof_summary["mirror_scalar_dof"],
        q3_dof_summary["paper_scalar_dof"] + q3_dof_summary["mirror_scalar_dof"],
    ]
    colors = [DESIGN_SYSTEM["neutral_fill"], DESIGN_SYSTEM["virtual_plane"], DESIGN_SYSTEM["ray"]]
    bars = ax_scale.bar(labels, values, color=colors, edgecolor=DESIGN_SYSTEM["panel_edge"], linewidth=1.0, width=0.64)
    ax_scale.set_yscale("log")
    ax_scale.set_ylim(1.0, max(values) * 1.8)
    style_panel(ax_scale, title="Panel A — scale proxy for geometry DOF versus image constraints", ylabel="Nominal scalar DOF / constraint count (log scale)", grid=True)
    add_panel_badge(ax_scale, "A", facecolor=DESIGN_SYSTEM["ink"])
    for bar, value in zip(bars, values, strict=True):
        ax_scale.text(bar.get_x() + bar.get_width() / 2.0, value * 1.1, f"{value:,}", ha="center", va="bottom", fontsize=9.3, color=DESIGN_SYSTEM["ink"])
    ax_scale.text(
        0.02,
        0.06,
        f"one-image / geometry ≈ {q3_dof_summary['single_image_to_geometry_ratio']:.0f}×\ndual-image / geometry ≈ {q3_dof_summary['dual_image_to_geometry_ratio']:.0f}×",
        transform=ax_scale.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.3,
        color=DESIGN_SYSTEM["ink"],
        bbox={"boxstyle": "round,pad=0.30", "facecolor": DESIGN_SYSTEM["panel_face"], "edgecolor": DESIGN_SYSTEM["panel_edge"]},
    )
    ax_scale.text(
        0.98,
        0.06,
        f"sampled symbol targets: {q3_dof_summary['paper_target_shape_px'][1]}×{q3_dof_summary['paper_target_shape_px'][0]} px",
        transform=ax_scale.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.0,
        color=DESIGN_SYSTEM["muted"],
    )

    ax_pairs = axes[1]
    counts = q3_dof_summary["pair_class_counts"]
    total = q3_dof_summary["pair_count"]
    left = 0.0
    for class_name in ["Low overlap", "Mixed overlap", "High overlap"]:
        count = counts[class_name]
        ax_pairs.barh([0.58], [count], left=left, height=0.26, color=CLASS_STYLES[class_name]["color"], edgecolor=DESIGN_SYSTEM["panel_edge"], linewidth=0.9)
        if count > 0:
            ax_pairs.text(left + count / 2.0, 0.58, f"{count}", ha="center", va="center", fontsize=10.0, color="white", fontweight="semibold")
        left += count
    style_panel(ax_pairs, title="Panel B — observed overlap classes in the current 6×6 sweep", xlabel="Sampled paper–mirror pairs (count)", grid=False)
    add_panel_badge(ax_pairs, "B", facecolor=DESIGN_SYSTEM["ink"])
    ax_pairs.set_xlim(0.0, float(total))
    ax_pairs.set_ylim(0.0, 1.0)
    ax_pairs.set_yticks([])
    legend_handles = [Rectangle((0, 0), 1, 1, facecolor=CLASS_STYLES[name]["color"], edgecolor="none") for name in ["Low overlap", "Mixed overlap", "High overlap"]]
    ax_pairs.legend(legend_handles, ["Low overlap", "Mixed overlap", "High overlap"], loc="upper left", bbox_to_anchor=(0.0, 0.92), ncol=3, frameon=False)
    ax_pairs.text(
        0.02,
        0.34,
        f"Total sampled pairs: {total}\nLow overlap: {counts['Low overlap']}\nMixed overlap: {counts['Mixed overlap']}\nHigh overlap: {counts['High overlap']}",
        transform=ax_pairs.transAxes,
        ha="left",
        va="top",
        fontsize=9.25,
        color=DESIGN_SYSTEM["ink"],
        bbox={"boxstyle": "round,pad=0.32", "facecolor": DESIGN_SYSTEM["panel_face"], "edgecolor": DESIGN_SYSTEM["panel_edge"]},
    )
    ax_pairs.text(
        0.02,
        0.10,
        "Finite geometry only reaches a narrow feasible subset of the independently specified pair space.",
        transform=ax_pairs.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.05,
        color=DESIGN_SYSTEM["muted"],
    )

    fig.suptitle("Problem 3: finite geometry DOF versus dual-image constraint load", fontsize=14.8, fontweight="semibold")
    add_figure_note(fig, "Pixel counts are used only as an honest scale proxy for image freedom. The Q3 compatibility metric remains unchanged: C = 0.25·mirror match + 0.25·low-frequency alignment + 0.25·symmetry alignment + 0.25·complexity alignment.")
    output_path = output_dir / "q3_dof_constraint_comparison.png"
    return save_figure(fig, output_path)


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


def save_validation_metrics(
    report_path: Path,
    report: dict[str, Any],
    geometry: Any,
    contexts: list[dict[str, Any]],
    sensitivity: dict[str, list[dict[str, Any]]],
    pair_results: list[dict[str, Any]],
    binocular_summary: dict[str, Any],
    q3_dof_summary: dict[str, Any],
    mockup_example: dict[str, Any],
    created_files: list[str],
    output_dir: Path,
) -> str:
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
        "compatibility_scoring": {
            "weights": COMPATIBILITY_WEIGHTS,
            "thresholds": {
                "high_overlap_min": HIGH_OVERLAP_THRESHOLD,
                "mixed_overlap_min": MIXED_OVERLAP_THRESHOLD,
                "low_overlap_max": MIXED_OVERLAP_THRESHOLD,
            },
        },
        "jobs": job_metrics,
        "sensitivity_analysis": sensitivity,
        "compatibility_pairs": [
            {
                "paper_key": item["paper_key"],
                "paper_label": item["paper_label"],
                "mirror_key": item["mirror_key"],
                "mirror_label": item["mirror_label"],
                "compatibility_class": item["compatibility_class"],
                "metrics": item["metrics"],
            }
            for item in pair_results
        ],
        "binocular_view_analysis": binocular_summary,
        "q3_dof_comparison": q3_dof_summary,
        "synthetic_spatial_mockup": {
            "target_key": mockup_example["target_key"],
            "target_label": mockup_example["target_label"],
            "plane_height_mm": float(mockup_example["plane_height_mm"]),
            "plane_width_mm": float(mockup_example["plane_width_mm"]),
            "round_trip_metrics": mockup_example["metrics"],
            "honesty_label": mockup_example["honesty_label"],
        },
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
    binocular_data = build_binocular_view_data(geometry, contexts)
    binocular_zone = compute_binocular_view_zone(geometry, representative_context(contexts))
    created_files.append(save_binocular_viewing_geometry(geometry, binocular_data, output_dir))
    created_files.append(save_binocular_viewing_zone(geometry, binocular_data, binocular_zone, output_dir))
    mockup_example = build_spatial_mockup_example(geometry)
    created_files.append(save_spatial_experiment_mockup(geometry, mockup_example, output_dir))
    created_files.extend(save_reconstruction_figures(contexts, geometry, output_dir))
    created_files.append(save_candidate_tradeoff_figure(report, output_dir))
    sensitivity = compute_sensitivity(job_specs, geometry)
    created_files.extend(save_sensitivity_figures(geometry, sensitivity, output_dir))
    created_files.append(save_problem2_free_design_gallery(geometry, output_dir))
    created_files.append(save_problem2_specified_gallery(geometry, output_dir))
    pair_results = build_problem3_pair_results(geometry)
    created_files.append(save_problem3_phase_diagram(pair_results, output_dir))
    created_files.append(save_problem3_pairwise_heatmap(pair_results, output_dir))
    created_files.append(save_problem3_example_gallery(pair_results, geometry, output_dir))
    q3_dof_summary = compute_q3_dof_summary(pair_results)
    created_files.append(save_q3_dof_constraint_comparison(q3_dof_summary, output_dir))
    created_files.append(
        save_validation_metrics(
            report_path,
            report,
            geometry,
            contexts,
            sensitivity,
            pair_results,
            {
                "reference_context": binocular_data["context_name"],
                "ipd_mm": binocular_data["ipd_mm"],
                "plane_width_mm": binocular_data["plane_width_mm"],
                "plane_height_mm": binocular_data["plane_height_mm"],
                "sample_count": len(binocular_data["samples"]),
                "mean_sampled_paper_separation_mm": float(np.mean([item["paper_separation_mm"] for item in binocular_data["samples"]])),
                "max_sampled_paper_separation_mm": float(np.max([item["paper_separation_mm"] for item in binocular_data["samples"]])),
                "zone": {
                    "reference_fraction": binocular_zone["reference_fraction"],
                    "threshold": binocular_zone["threshold"],
                    "selected_joint_ratio": binocular_zone["selected_joint_ratio"],
                    "selected_single_ratio": binocular_zone["selected_single_ratio"],
                    "max_joint_ratio": binocular_zone["max_joint_ratio"],
                    "best_center": binocular_zone["best_center"],
                    "recommended_bbox": binocular_zone["recommended_bbox"],
                    "recommended_fraction_of_grid": binocular_zone["recommended_fraction_of_grid"],
                },
            },
            q3_dof_summary,
            mockup_example,
            created_files.copy(),
            output_dir,
        )
    )

    for filename in created_files:
        print(output_dir / filename)


if __name__ == "__main__":
    main()
