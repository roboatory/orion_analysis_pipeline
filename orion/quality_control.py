from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import ListedColormap
from skimage import color

from orion.data_models import RegionOfInterestBox
from orion.input_output import ensure_directory

matplotlib.use("Agg")


def save_region_of_interest_preview(
    histology_patch: np.ndarray,
    region_of_interest: RegionOfInterestBox,
    path: Path,
) -> Path:
    ensure_directory(path.parent)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(histology_patch)
    axis.set_title(
        "Region of interest "
        f"x={region_of_interest.x_pixels} "
        f"y={region_of_interest.y_pixels} "
        f"width={region_of_interest.width_pixels} "
        f"height={region_of_interest.height_pixels}"
    )
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_marker_panels(
    marker_images: dict[str, np.ndarray],
    path: Path,
    marker_names: list[str],
) -> Path:
    ensure_directory(path.parent)
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, marker_name in zip(axes.ravel(), marker_names, strict=False):
        axis.imshow(marker_images[marker_name], cmap="magma")
        axis.set_title(marker_name)
        axis.axis("off")
    for axis in axes.ravel()[len(marker_names) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_segmentation_overlay(
    histology_patch: np.ndarray,
    label_image: np.ndarray,
    path: Path,
) -> Path:
    ensure_directory(path.parent)
    overlay_image = color.label2rgb(
        label_image,
        image=histology_patch,
        bg_label=0,
        alpha=0.35,
    )
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(overlay_image)
    axis.set_title("Segmentation overlay")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_threshold_histograms(
    normalized_cell_measurements: pl.DataFrame,
    threshold_summary: pl.DataFrame,
    marker_names: list[str],
    path: Path,
) -> Path:
    ensure_directory(path.parent)
    figure, axes = plt.subplots(3, 5, figsize=(15, 9))
    flat_axes = axes.ravel()
    threshold_by_marker = (
        {
            row["marker_name"]: row["threshold_value"]
            for row in threshold_summary.to_dicts()
        }
        if threshold_summary.height
        else {}
    )
    for axis, marker_name in zip(flat_axes, marker_names, strict=False):
        normalized_column_name = f"{marker_name}_arcsinh"
        if normalized_column_name not in normalized_cell_measurements.columns:
            axis.axis("off")
            continue
        normalized_values = normalized_cell_measurements[
            normalized_column_name
        ].to_numpy()
        axis.hist(normalized_values, bins=50, color="#375a7f", alpha=0.9)
        if marker_name in threshold_by_marker:
            axis.axvline(
                threshold_by_marker[marker_name],
                color="#d62728",
                linewidth=1.5,
            )
        axis.set_title(marker_name)
    for axis in flat_axes[len(marker_names) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_cell_type_map(annotated_cell_measurements: pl.DataFrame, path: Path) -> Path:
    ensure_directory(path.parent)
    if annotated_cell_measurements.is_empty():
        return path
    cell_types = annotated_cell_measurements["cell_type"].to_list()
    unique_cell_types = sorted(set(cell_types))
    color_map = ListedColormap(
        plt.cm.tab20(np.linspace(0, 1, max(len(unique_cell_types), 1)))
    )
    color_index_by_cell_type = {
        cell_type: index for index, cell_type in enumerate(unique_cell_types)
    }
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(
        annotated_cell_measurements["x_pixels"].to_numpy(),
        annotated_cell_measurements["y_pixels"].to_numpy(),
        c=[color_index_by_cell_type[cell_type] for cell_type in cell_types],
        cmap=color_map,
        s=6,
        alpha=0.8,
    )
    axis.set_title("Cell type map")
    axis.set_xlabel("x (pixels)")
    axis.set_ylabel("y (pixels)")
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_neighborhood_map(
    neighborhood_clusters: pl.DataFrame,
    annotated_cell_measurements: pl.DataFrame,
    path: Path,
) -> Path:
    ensure_directory(path.parent)
    if neighborhood_clusters.is_empty():
        return path
    merged_data_frame = neighborhood_clusters.select(
        ["cell_identifier", "neighborhood_cluster"]
    ).join(
        annotated_cell_measurements.select(["cell_identifier", "x_pixels", "y_pixels"]),
        on="cell_identifier",
        how="left",
    )
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(
        merged_data_frame["x_pixels"].to_numpy(),
        merged_data_frame["y_pixels"].to_numpy(),
        c=merged_data_frame["neighborhood_cluster"].to_numpy(),
        cmap="tab20",
        s=6,
        alpha=0.8,
    )
    axis.set_title("Neighborhood clusters")
    axis.set_xlabel("x (pixels)")
    axis.set_ylabel("y (pixels)")
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
