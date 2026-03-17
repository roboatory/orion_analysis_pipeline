from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import ListedColormap
from skimage import color

from src.io import ensure_directory, write_rgb_image

matplotlib.use("Agg")


def save_preprocessing_comparison(
    original_image: np.ndarray,
    corrected_image: np.ndarray,
    marker_name: str,
    path: Path,
) -> Path:
    ensure_directory(path.parent)
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image, cmap="magma")
    axes[0].set_title(f"{marker_name} original")
    axes[0].axis("off")
    axes[1].imshow(corrected_image, cmap="magma")
    axes[1].set_title(f"{marker_name} corrected")
    axes[1].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_segmentation_overlay_image(
    background_image: np.ndarray,
    label_image: np.ndarray,
    path: Path,
) -> Path:
    background_rgb = normalize_background_for_overlay(background_image)
    overlay_image = color.label2rgb(
        label_image,
        image=background_rgb,
        bg_label=0,
        alpha=0.35,
    )
    overlay_uint8 = np.clip(overlay_image * 255.0, 0, 255).astype(np.uint8)
    return write_rgb_image(path, overlay_uint8)


def save_cell_assignment_map(
    cell_annotations: pl.DataFrame,
    label_column_name: str,
    title: str,
    path: Path,
) -> Path:
    ensure_directory(path.parent)
    if cell_annotations.is_empty():
        return path
    labels = cell_annotations[label_column_name].to_list()
    unique_labels = sorted(set(labels))
    color_map = ListedColormap(
        plt.cm.tab20(np.linspace(0, 1, max(len(unique_labels), 1)))
    )
    color_index_by_label = {label: index for index, label in enumerate(unique_labels)}
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(
        cell_annotations["x_pixels"].to_numpy(),
        cell_annotations["y_pixels"].to_numpy(),
        c=[color_index_by_label[label] for label in labels],
        cmap=color_map,
        s=6,
        alpha=0.8,
    )
    axis.set_title(title)
    axis.set_xlabel("x (pixels)")
    axis.set_ylabel("y (pixels)")
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def normalize_background_for_overlay(background_image: np.ndarray) -> np.ndarray:
    image = background_image.astype(np.float32)
    if image.ndim == 2:
        minimum_value = float(np.min(image))
        maximum_value = float(np.max(image))
        if maximum_value <= minimum_value:
            normalized_image = np.zeros_like(image, dtype=np.float32)
        else:
            normalized_image = (image - minimum_value) / (maximum_value - minimum_value)
        return np.repeat(normalized_image[..., None], 3, axis=2)
    if image.ndim == 3:
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        minimum_value = float(np.min(image))
        maximum_value = float(np.max(image))
        if maximum_value <= minimum_value:
            return np.zeros_like(image, dtype=np.float32)
        return (image - minimum_value) / (maximum_value - minimum_value)
    raise ValueError("Unsupported background image shape for overlay generation.")
