#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from matplotlib.patches import Circle, Rectangle
from PIL import Image


A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
PAGE_X_MIN = -A4_WIDTH_MM / 2.0
PAGE_X_MAX = A4_WIDTH_MM / 2.0
PAGE_Y_MIN = -A4_HEIGHT_MM / 2.0
PAGE_Y_MAX = A4_HEIGHT_MM / 2.0
EPS = 1.0e-9
SECOND_HIT_EPS = 1.0e-4
DEFAULT_DPI = 200
DEFAULT_MAX_SIDE = 900

FloatArray = NDArray[np.float64]
ImageArray = NDArray[np.float32]
MaskArray = NDArray[np.bool_]
MappingArray = FloatArray | MaskArray


class MappingResult(TypedDict):
    Ax: FloatArray
    Ay: FloatArray
    Hx: FloatArray
    Hy: FloatArray
    Hz: FloatArray
    X: FloatArray
    Z: FloatArray
    valid: MaskArray


@dataclass(frozen=True)
class GeometryCandidate:
    x0: float
    y0: float
    R: float
    H: float
    xv: float
    yv: float
    zv: float
    y_img: float
    z_img_center: float
    page_margin_mm: float = 5.0
    cylinder_clearance_mm: float = 3.0


DEFAULT_JOBS = [
    {"name": "fig3", "input_path": "/home/xianz/huazhongbei/附件/图3.png"},
    {"name": "fig4", "input_path": "/home/xianz/huazhongbei/附件/图4.png"},
]


def load_rgb_image(path: Path, max_side: int) -> tuple[ImageArray, dict[str, Any]]:
    image = Image.open(path).convert("RGB")
    original_size = image.size
    longest_side = max(original_size)
    scale = max_side / float(longest_side)
    if abs(scale - 1.0) > 1.0e-6:
        new_size = tuple(max(1, int(round(v * scale))) for v in original_size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    working_size = image.size
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array, {
        "original_size_px": list(original_size),
        "working_size_px": list(working_size),
        "resample_scale": float(scale),
    }


def page_bounds(margin_mm: float) -> tuple[float, float, float, float]:
    return (
        PAGE_X_MIN + margin_mm,
        PAGE_X_MAX - margin_mm,
        PAGE_Y_MIN + margin_mm,
        PAGE_Y_MAX - margin_mm,
    )


def build_virtual_plane(width_px: int, height_px: int, plane_height_mm: float, geometry: GeometryCandidate) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    plane_width_mm = plane_height_mm * width_px / float(height_px)
    x_coords = geometry.x0 + plane_width_mm * (((np.arange(width_px) + 0.5) / width_px) - 0.5)
    z_coords = geometry.z_img_center + plane_height_mm * (0.5 - ((np.arange(height_px) + 0.5) / height_px))
    X, Z = np.meshgrid(x_coords, z_coords)
    Y = np.full_like(X, geometry.y_img, dtype=np.float64)
    return X.astype(np.float64), Y, Z.astype(np.float64), float(plane_width_mm)


def inverse_map_virtual_plane(X: FloatArray, Y: FloatArray, Z: FloatArray, geometry: GeometryCandidate) -> MappingResult:
    Vx, Vy, Vz = geometry.xv, geometry.yv, geometry.zv
    dx = X - Vx
    dy = Y - Vy
    dz = Z - Vz

    a = dx * dx + dy * dy
    b = 2.0 * ((Vx - geometry.x0) * dx + (Vy - geometry.y0) * dy)
    c = (Vx - geometry.x0) ** 2 + (Vy - geometry.y0) ** 2 - geometry.R ** 2
    disc = b * b - 4.0 * a * c

    valid = (a > EPS) & (disc > EPS)
    sqrt_disc = np.zeros_like(X)
    sqrt_disc[valid] = np.sqrt(disc[valid])

    t1 = np.full_like(X, np.nan)
    t2 = np.full_like(X, np.nan)
    denom = 2.0 * a
    t1[valid] = (-b[valid] - sqrt_disc[valid]) / denom[valid]
    t2[valid] = (-b[valid] + sqrt_disc[valid]) / denom[valid]

    t1 = np.where((t1 > EPS) & (t1 < 1.0 - EPS), t1, np.inf)
    t2 = np.where((t2 > EPS) & (t2 < 1.0 - EPS), t2, np.inf)
    t = np.minimum(t1, t2)

    valid &= np.isfinite(t)
    Hx = Vx + t * dx
    Hy = Vy + t * dy
    Hz = Vz + t * dz
    valid &= (Hz >= -EPS) & (Hz <= geometry.H + EPS)

    nx = (Hx - geometry.x0) / geometry.R
    ny = (Hy - geometry.y0) / geometry.R

    dix = X - Hx
    diy = Y - Hy
    diz = Z - Hz
    norm = np.sqrt(dix * dix + diy * diy + diz * diz)
    valid &= norm > EPS
    safe_norm = np.where(norm > EPS, norm, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dix = np.divide(dix, safe_norm, out=np.zeros_like(dix), where=safe_norm > EPS)
        diy = np.divide(diy, safe_norm, out=np.zeros_like(diy), where=safe_norm > EPS)
        diz = np.divide(diz, safe_norm, out=np.zeros_like(diz), where=safe_norm > EPS)

    dot = dix * nx + diy * ny
    rox = dix - 2.0 * dot * nx
    roy = diy - 2.0 * dot * ny
    roz = diz

    valid &= roz < -EPS
    t_plane = -Hz / np.where(np.abs(roz) > EPS, roz, -1.0)
    valid &= t_plane > EPS

    Ax = Hx + t_plane * rox
    Ay = Hy + t_plane * roy

    Ox = Hx + SECOND_HIT_EPS * rox
    Oy = Hy + SECOND_HIT_EPS * roy
    Oz = Hz + SECOND_HIT_EPS * roz
    a2 = rox * rox + roy * roy
    b2 = 2.0 * ((Ox - geometry.x0) * rox + (Oy - geometry.y0) * roy)
    c2 = (Ox - geometry.x0) ** 2 + (Oy - geometry.y0) ** 2 - geometry.R ** 2
    disc2 = b2 * b2 - 4.0 * a2 * c2
    second_mask = valid & (a2 > EPS) & (disc2 > EPS)
    sqrt_disc2 = np.zeros_like(X)
    sqrt_disc2[second_mask] = np.sqrt(disc2[second_mask])
    s1 = np.full_like(X, np.nan)
    s2 = np.full_like(X, np.nan)
    denom2 = 2.0 * a2
    s1[second_mask] = (-b2[second_mask] - sqrt_disc2[second_mask]) / denom2[second_mask]
    s2[second_mask] = (-b2[second_mask] + sqrt_disc2[second_mask]) / denom2[second_mask]
    s1 = np.where(s1 > SECOND_HIT_EPS, s1, np.inf)
    s2 = np.where(s2 > SECOND_HIT_EPS, s2, np.inf)
    t_second = np.minimum(s1, s2)
    z_second = Oz + t_second * roz
    self_blocked = (
        np.isfinite(t_second)
        & (t_second < t_plane - SECOND_HIT_EPS)
        & (z_second >= -EPS)
        & (z_second <= geometry.H + EPS)
    )
    valid &= ~self_blocked

    return {
        "Ax": Ax,
        "Ay": Ay,
        "Hx": Hx,
        "Hy": Hy,
        "Hz": Hz,
        "X": X,
        "Z": Z,
        "valid": valid,
    }


def apply_page_clip(valid: MaskArray, Ax: FloatArray, Ay: FloatArray, geometry: GeometryCandidate, margin_mm: float) -> MaskArray:
    x_min, x_max, y_min, y_max = page_bounds(margin_mm)
    outside_cylinder = (Ax - geometry.x0) ** 2 + (Ay - geometry.y0) ** 2 >= (geometry.R + geometry.cylinder_clearance_mm) ** 2
    return valid & (Ax >= x_min) & (Ax <= x_max) & (Ay >= y_min) & (Ay <= y_max) & outside_cylinder


def compute_distortion(Ax: FloatArray, Ay: FloatArray, valid: MaskArray, plane_width_mm: float, plane_height_mm: float) -> dict[str, Any]:
    if valid.sum() < 16:
        empty = np.full((max(1, Ax.shape[0] - 1), max(1, Ax.shape[1] - 1)), np.nan, dtype=np.float64)
        return {
            "log10_area_scale": empty,
            "log10_condition": empty.copy(),
            "median_condition": float("inf"),
            "std_log_area_scale": float("inf"),
        }

    cell_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1]
    dux = Ax[:-1, 1:] - Ax[:-1, :-1]
    duy = Ay[:-1, 1:] - Ay[:-1, :-1]
    dvx = Ax[1:, :-1] - Ax[:-1, :-1]
    dvy = Ay[1:, :-1] - Ay[:-1, :-1]

    step_u = plane_width_mm / float(Ax.shape[1])
    step_v = plane_height_mm / float(Ax.shape[0])
    a = dux / step_u
    b = dvx / step_v
    c = duy / step_u
    d = dvy / step_v

    det = a * d - b * c
    area_scale = np.abs(det)
    frob_sq = a * a + b * b + c * c + d * d
    delta = np.sqrt(np.maximum(frob_sq * frob_sq - 4.0 * det * det, 0.0))
    sigma_max = np.sqrt(np.maximum((frob_sq + delta) * 0.5, EPS))
    sigma_min = np.sqrt(np.maximum((frob_sq - delta) * 0.5, EPS))
    condition = sigma_max / sigma_min

    log_area = np.full_like(area_scale, np.nan)
    log_condition = np.full_like(area_scale, np.nan)
    positive_mask = cell_valid & (area_scale > EPS)
    log_area[positive_mask] = np.log10(area_scale[positive_mask])
    log_condition[cell_valid] = np.log10(np.maximum(condition[cell_valid], 1.0))

    if np.any(positive_mask):
        median_condition = float(np.nanmedian(condition[cell_valid]))
        std_log_area_scale = float(np.nanstd(np.log(np.maximum(area_scale[positive_mask], EPS))))
    else:
        median_condition = float("inf")
        std_log_area_scale = float("inf")

    return {
        "log10_area_scale": log_area,
        "log10_condition": log_condition,
        "median_condition": median_condition,
        "std_log_area_scale": std_log_area_scale,
    }


def evaluate_candidate(image_shape: tuple[int, int], geometry: GeometryCandidate, plane_height_mm: float, coarse_side: int = 48) -> dict[str, Any]:
    height_px, width_px = image_shape
    aspect = width_px / float(height_px)
    if width_px >= height_px:
        coarse_width = coarse_side
        coarse_height = max(8, int(round(coarse_side / aspect)))
    else:
        coarse_height = coarse_side
        coarse_width = max(8, int(round(coarse_side * aspect)))

    X, Y, Z, plane_width_mm = build_virtual_plane(coarse_width, coarse_height, plane_height_mm, geometry)
    mapping = inverse_map_virtual_plane(X, Y, Z, geometry)
    ray_mask = mapping["valid"]
    page_mask = apply_page_clip(ray_mask, mapping["Ax"], mapping["Ay"], geometry, geometry.page_margin_mm)
    if int(page_mask.sum()) < max(24, coarse_width):
        return {
            "score": -1.0e9,
            "ray_fraction": float(ray_mask.mean()),
            "paper_fraction": float(page_mask.mean()),
            "plane_height_mm": plane_height_mm,
            "plane_width_mm": plane_width_mm,
        }

    x_vals = mapping["Ax"][page_mask]
    y_vals = mapping["Ay"][page_mask]
    bbox_width = float(x_vals.max() - x_vals.min())
    bbox_height = float(y_vals.max() - y_vals.min())
    fill_ratio = (bbox_width / A4_WIDTH_MM) * (bbox_height / A4_HEIGHT_MM)
    distortion = compute_distortion(mapping["Ax"], mapping["Ay"], page_mask, plane_width_mm, plane_height_mm)

    if not np.isfinite(distortion["median_condition"]):
        score = -1.0e8
    else:
        score = (
            float(page_mask.mean())
            + 0.30 * fill_ratio
            + 0.10 * float(ray_mask.mean())
            - 0.08 * np.log(max(distortion["median_condition"], 1.0))
            - 0.08 * distortion["std_log_area_scale"]
        )

    return {
        "score": float(score),
        "ray_fraction": float(ray_mask.mean()),
        "paper_fraction": float(page_mask.mean()),
        "plane_height_mm": float(plane_height_mm),
        "plane_width_mm": float(plane_width_mm),
        "bbox_width_mm": bbox_width,
        "bbox_height_mm": bbox_height,
        "bbox_x_min_mm": float(x_vals.min()),
        "bbox_x_max_mm": float(x_vals.max()),
        "bbox_y_min_mm": float(y_vals.min()),
        "bbox_y_max_mm": float(y_vals.max()),
        "median_condition": float(distortion["median_condition"]),
        "std_log_area_scale": float(distortion["std_log_area_scale"]),
    }


def build_geometry_candidates() -> list[GeometryCandidate]:
    candidates: list[GeometryCandidate] = []
    for y0 in (40.0, 50.0):
        for radius in (20.0, 22.0):
            for yv in (-200.0, -220.0):
                for zv in (130.0, 140.0):
                    for y_offset in (35.0, 45.0):
                        for z_img_center in (12.0, 16.0):
                            candidates.append(
                                GeometryCandidate(
                                    x0=0.0,
                                    y0=y0,
                                    R=radius,
                                    H=180.0,
                                    xv=0.0,
                                    yv=yv,
                                    zv=zv,
                                    y_img=y0 + y_offset,
                                    z_img_center=z_img_center,
                                )
                            )
    return candidates


def search_geometry(job_specs: list[dict[str, Any]]) -> tuple[GeometryCandidate, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    plane_height_candidates = [22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0]
    candidate_summaries: list[dict[str, Any]] = []
    best_total = -1.0e18
    best_geometry: GeometryCandidate | None = None
    best_job_results: dict[str, dict[str, Any]] = {}

    for geometry in build_geometry_candidates():
        total_score = 0.0
        per_job: dict[str, dict[str, Any]] = {}
        acceptable = True
        for spec in job_specs:
            best_result: dict[str, Any] | None = None
            for plane_height_mm in plane_height_candidates:
                result = evaluate_candidate(tuple(spec["image_array"].shape[:2]), geometry, plane_height_mm)
                if best_result is None or result["score"] > best_result["score"]:
                    best_result = result
            assert best_result is not None
            if best_result["paper_fraction"] < 0.55:
                acceptable = False
                break
            per_job[spec["name"]] = best_result
            total_score += best_result["score"]
        candidate_summaries.append(
            {
                "geometry": asdict(geometry),
                "total_score": float(total_score if acceptable else -1.0e18),
                "accepted": acceptable,
                "jobs": per_job,
            }
        )
        if acceptable and total_score > best_total:
            best_total = total_score
            best_geometry = geometry
            best_job_results = per_job

    if best_geometry is None:
        raise RuntimeError("No geometry candidate produced an A4-fit solution for both images.")

    candidate_summaries.sort(key=lambda item: item["total_score"], reverse=True)
    return best_geometry, best_job_results, candidate_summaries[:5]


def render_paper_pattern(image_array: ImageArray, Ax: FloatArray, Ay: FloatArray, valid: MaskArray, dpi: int) -> tuple[ImageArray, ImageArray]:
    page_width_px = int(round(A4_WIDTH_MM / 25.4 * dpi))
    page_height_px = int(round(A4_HEIGHT_MM / 25.4 * dpi))

    x = Ax[valid]
    y = Ay[valid]
    colors = image_array[valid]
    px = (x - PAGE_X_MIN) / A4_WIDTH_MM * (page_width_px - 1)
    py = (PAGE_Y_MAX - y) / A4_HEIGHT_MM * (page_height_px - 1)

    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    dx = px - x0
    dy = py - y0

    accum = np.zeros((page_width_px * page_height_px, 3), dtype=np.float32)
    weights = np.zeros(page_width_px * page_height_px, dtype=np.float32)

    contributions = (
        (0, 0, (1.0 - dx) * (1.0 - dy)),
        (1, 0, dx * (1.0 - dy)),
        (0, 1, (1.0 - dx) * dy),
        (1, 1, dx * dy),
    )
    for ox, oy, w in contributions:
        xi = x0 + ox
        yi = y0 + oy
        in_bounds = (xi >= 0) & (xi < page_width_px) & (yi >= 0) & (yi < page_height_px) & (w > 0.0)
        if not np.any(in_bounds):
            continue
        flat_idx = yi[in_bounds] * page_width_px + xi[in_bounds]
        ww = w[in_bounds].astype(np.float32)
        np.add.at(weights, flat_idx, ww)
        np.add.at(accum[:, 0], flat_idx, ww * colors[in_bounds, 0])
        np.add.at(accum[:, 1], flat_idx, ww * colors[in_bounds, 1])
        np.add.at(accum[:, 2], flat_idx, ww * colors[in_bounds, 2])

    paper = np.ones_like(accum)
    nonzero = weights > EPS
    paper[nonzero] = accum[nonzero] / weights[nonzero, None]
    paper = paper.reshape((page_height_px, page_width_px, 3))
    occupancy = weights.reshape((page_height_px, page_width_px))
    return paper, occupancy


def save_rgb_image(image_array: ImageArray, output_path: Path) -> None:
    clipped = np.clip(image_array, 0.0, 1.0)
    image = Image.fromarray(np.round(clipped * 255.0).astype(np.uint8), mode="RGB")
    image.save(output_path)


def save_diagnostic_figure(
    source_image: ImageArray,
    paper_pattern: ImageArray,
    mapping: MappingResult,
    clipped_mask: MaskArray,
    plane_width_mm: float,
    plane_height_mm: float,
    geometry: GeometryCandidate,
    output_path: Path,
) -> dict[str, float]:
    distortion = compute_distortion(mapping["Ax"], mapping["Ay"], clipped_mask, plane_width_mm, plane_height_mm)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    axes[0, 0].imshow(source_image)
    axes[0, 0].set_title("Target mirror image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.flipud(paper_pattern), extent=[PAGE_X_MIN, PAGE_X_MAX, PAGE_Y_MIN, PAGE_Y_MAX], origin="lower")
    axes[0, 1].add_patch(Rectangle((PAGE_X_MIN, PAGE_Y_MIN), A4_WIDTH_MM, A4_HEIGHT_MM, fill=False, edgecolor="black", linewidth=1.0))
    axes[0, 1].add_patch(Circle((geometry.x0, geometry.y0), geometry.R, fill=False, edgecolor="red", linewidth=1.2))
    axes[0, 1].set_title("Generated paper pattern on A4")
    axes[0, 1].set_xlabel("x / mm")
    axes[0, 1].set_ylabel("y / mm")
    axes[0, 1].set_aspect("equal")

    log_area = distortion["log10_area_scale"]
    extent = [geometry.x0 - plane_width_mm / 2.0, geometry.x0 + plane_width_mm / 2.0, geometry.z_img_center - plane_height_mm / 2.0, geometry.z_img_center + plane_height_mm / 2.0]
    im = axes[1, 0].imshow(log_area, origin="lower", extent=extent, cmap="viridis")
    axes[1, 0].set_title("log10(local area scale)")
    axes[1, 0].set_xlabel("virtual image x / mm")
    axes[1, 0].set_ylabel("virtual image z / mm")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    step = max(1, mapping["Ax"].shape[0] // 50)
    sample_mask = clipped_mask[::step, ::step]
    sample_x = mapping["Ax"][::step, ::step][sample_mask]
    sample_y = mapping["Ay"][::step, ::step][sample_mask]
    sample_z = mapping["Z"][::step, ::step][sample_mask]
    axes[1, 1].scatter(sample_x, sample_y, c=sample_z, s=6, cmap="plasma", alpha=0.75, linewidths=0.0)
    axes[1, 1].add_patch(Rectangle((PAGE_X_MIN, PAGE_Y_MIN), A4_WIDTH_MM, A4_HEIGHT_MM, fill=False, edgecolor="black", linewidth=1.0))
    axes[1, 1].add_patch(Circle((geometry.x0, geometry.y0), geometry.R, fill=False, edgecolor="red", linewidth=1.2))
    axes[1, 1].set_title("Mapped valid samples on paper")
    axes[1, 1].set_xlabel("x / mm")
    axes[1, 1].set_ylabel("y / mm")
    axes[1, 1].set_aspect("equal")

    fig.suptitle("Cylindrical anamorphosis diagnostics", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "median_condition": float(distortion["median_condition"]),
        "std_log_area_scale": float(distortion["std_log_area_scale"]),
    }


def a4_anchor_mm(x_mm: float, y_mm: float) -> dict[str, float]:
    return {
        "from_left_mm": float(x_mm - PAGE_X_MIN),
        "from_front_edge_mm": float(y_mm - PAGE_Y_MIN),
    }


def run_job(spec: dict[str, Any], geometry: GeometryCandidate, plane_height_mm: float, dpi: int, output_dir: Path) -> dict[str, Any]:
    source_image = spec["image_array"]
    height_px, width_px = source_image.shape[:2]
    X, Y, Z, plane_width_mm = build_virtual_plane(width_px, height_px, plane_height_mm, geometry)
    mapping = inverse_map_virtual_plane(X, Y, Z, geometry)
    ray_mask = mapping["valid"]
    clipped_mask = apply_page_clip(ray_mask, mapping["Ax"], mapping["Ay"], geometry, 0.0)
    paper_pattern, occupancy = render_paper_pattern(source_image, mapping["Ax"], mapping["Ay"], clipped_mask, dpi)

    pattern_path = output_dir / f"{spec['name']}_paper_pattern.png"
    diagnostic_path = output_dir / f"{spec['name']}_diagnostic.png"
    occupancy_path = output_dir / f"{spec['name']}_occupancy.png"
    save_rgb_image(paper_pattern, pattern_path)

    occupancy_norm = occupancy / max(float(occupancy.max()), 1.0)
    save_rgb_image(np.repeat(occupancy_norm[:, :, None], 3, axis=2), occupancy_path)

    distortion_stats = save_diagnostic_figure(
        source_image=source_image,
        paper_pattern=paper_pattern,
        mapping=mapping,
        clipped_mask=clipped_mask,
        plane_width_mm=plane_width_mm,
        plane_height_mm=plane_height_mm,
        geometry=geometry,
        output_path=diagnostic_path,
    )

    if clipped_mask.sum() == 0:
        raise RuntimeError(f"No valid A4-clipped samples for {spec['name']}.")

    x_vals = mapping["Ax"][clipped_mask]
    y_vals = mapping["Ay"][clipped_mask]
    bbox = {
        "x_min_mm": float(x_vals.min()),
        "x_max_mm": float(x_vals.max()),
        "y_min_mm": float(y_vals.min()),
        "y_max_mm": float(y_vals.max()),
        "width_mm": float(x_vals.max() - x_vals.min()),
        "height_mm": float(y_vals.max() - y_vals.min()),
    }

    return {
        "name": spec["name"],
        "input_path": str(spec["input_path"]),
        "original_size_px": spec["image_meta"]["original_size_px"],
        "working_size_px": spec["image_meta"]["working_size_px"],
        "resample_scale": spec["image_meta"]["resample_scale"],
        "virtual_image_plane": {
            "width_mm": float(plane_width_mm),
            "height_mm": float(plane_height_mm),
            "center_mm": {
                "x": float(geometry.x0),
                "y": float(geometry.y_img),
                "z": float(geometry.z_img_center),
            },
            "z_range_mm": [float(geometry.z_img_center - plane_height_mm / 2.0), float(geometry.z_img_center + plane_height_mm / 2.0)],
        },
        "ray_valid_fraction": float(ray_mask.mean()),
        "a4_valid_fraction": float(clipped_mask.mean()),
        "paper_bbox_mm": bbox,
        "paper_bbox_a4_anchor_mm": {
            "lower_left": a4_anchor_mm(bbox["x_min_mm"], bbox["y_min_mm"]),
            "upper_right": a4_anchor_mm(bbox["x_max_mm"], bbox["y_max_mm"]),
        },
        "distortion": distortion_stats,
        "outputs": {
            "paper_pattern_png": str(pattern_path),
            "diagnostic_png": str(diagnostic_path),
            "occupancy_png": str(occupancy_path),
        },
    }


def write_summary(report: dict[str, Any], output_path: Path) -> None:
    lines = []
    lines.append("# Cylindrical-mirror anamorphosis batch")
    lines.append("")
    lines.append("This batch uses the inverse chain `I -> H -> A`: line-cylinder intersection on `VI`, cylinder normal `n`, specular reflection `d_out = d_in - 2(d_in·n)n`, then line-plane intersection with `z=0`.")
    lines.append("")
    geometry = report["selected_geometry"]
    lines.append("## Selected common geometry")
    lines.append("")
    lines.append(
        f"- Cylinder center `(x0, y0)=({geometry['x0']:.1f}, {geometry['y0']:.1f}) mm`, radius `R={geometry['R']:.1f} mm`, height `H={geometry['H']:.1f} mm`."
    )
    lines.append(
        f"- Viewpoint `V=({geometry['xv']:.1f}, {geometry['yv']:.1f}, {geometry['zv']:.1f}) mm`. Virtual image plane `y={geometry['y_img']:.1f} mm`, `z` center `{geometry['z_img_center']:.1f} mm`."
    )
    lines.append(
        f"- A4 paper is modeled as `210 x 297 mm`; the cylinder footprint is clipped with an extra `{geometry['cylinder_clearance_mm']:.1f} mm` clearance."
    )
    lines.append("")
    lines.append("## Per-image outputs")
    lines.append("")
    for job in report["jobs"]:
        plane = job["virtual_image_plane"]
        bbox = job["paper_bbox_mm"]
        outputs = job["outputs"]
        lines.append(
            f"- **{job['name']}**: virtual plane `{plane['width_mm']:.2f} x {plane['height_mm']:.2f} mm`, A4-valid fraction `{job['a4_valid_fraction']:.3f}`, paper bbox `{bbox['width_mm']:.1f} x {bbox['height_mm']:.1f} mm`."
        )
        lines.append(f"  - pattern: `{outputs['paper_pattern_png']}`")
        lines.append(f"  - diagnostic: `{outputs['diagnostic_png']}`")
        lines.append(f"  - occupancy: `{outputs['occupancy_png']}`")
    lines.append("")
    lines.append("## Rerun")
    lines.append("")
    lines.append("```bash")
    lines.append("python /home/xianz/huazhongbei/anamorphosis/generate_patterns.py")
    lines.append("```")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_job_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input is not None:
        name = args.name or Path(args.input).stem
        job_defs = [{"name": name, "input_path": str(Path(args.input).resolve())}]
    else:
        job_defs = DEFAULT_JOBS

    specs: list[dict[str, Any]] = []
    for item in job_defs:
        input_path = Path(item["input_path"]).resolve()
        image_array, image_meta = load_rgb_image(input_path, args.max_side)
        specs.append(
            {
                "name": item["name"],
                "input_path": input_path,
                "image_array": image_array,
                "image_meta": image_meta,
            }
        )
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate A4 paper patterns for cylindrical-mirror anamorphosis.")
    parser.add_argument("--input", type=str, help="Optional single input image path.")
    parser.add_argument("--name", type=str, help="Optional job name when --input is used.")
    parser.add_argument("--output-dir", type=str, default="/home/xianz/huazhongbei/outputs/cylindrical_anamorphosis", help="Directory for generated outputs.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Output paper pattern DPI.")
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE, help="Resize long image side to this many pixels before mapping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    job_specs = build_job_specs(args)
    geometry, search_results, top_candidates = search_geometry(job_specs)

    jobs = []
    for spec in job_specs:
        best_height = search_results[spec["name"]]["plane_height_mm"]
        jobs.append(run_job(spec, geometry, best_height, args.dpi, output_dir))

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script_path": str(Path(__file__).resolve()),
        "rerun_command": "python /home/xianz/huazhongbei/anamorphosis/generate_patterns.py",
        "page": {
            "size_mm": [A4_WIDTH_MM, A4_HEIGHT_MM],
            "dpi": int(args.dpi),
            "bounds_mm": {
                "x_min": PAGE_X_MIN,
                "x_max": PAGE_X_MAX,
                "y_min": PAGE_Y_MIN,
                "y_max": PAGE_Y_MAX,
            },
        },
        "selected_geometry": asdict(geometry),
        "search_results": {name: result for name, result in search_results.items()},
        "top_geometry_candidates": top_candidates,
        "jobs": jobs,
    }

    report_path = output_dir / "parameter_report.json"
    summary_path = output_dir / "summary.md"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(report, summary_path)

    print(f"Report written to: {report_path}")
    print(f"Summary written to: {summary_path}")
    for job in jobs:
        print(f"{job['name']}: {job['outputs']['paper_pattern_png']}")


if __name__ == "__main__":
    main()
