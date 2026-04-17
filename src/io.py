from __future__ import annotations

import xml.etree.ElementTree as xml_element_tree
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tifffile
import yaml
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from skimage import color

from src.data_models import RegionOfInterestBox, SlideMetadata

if TYPE_CHECKING:
    from src.configuration import ApplicationConfiguration

matplotlib.use("Agg")


def ensure_directory(path: Path) -> Path:
    """Create the directory and any missing parents, returning the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_marker_names(path: Path) -> list[str]:
    """Read one marker name per line from a plain-text markers file."""
    with path.open("r", encoding="utf-8") as file_handle:
        return [line.strip() for line in file_handle if line.strip()]


def build_marker_name_to_index(marker_names: list[str]) -> dict[str, int]:
    """Map each marker name to its zero-based channel index."""
    return {
        marker_name: marker_index
        for marker_index, marker_name in enumerate(marker_names)
    }


def percentile_normalize_image(
    image: np.ndarray,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> np.ndarray:
    """Rescale pixel intensities to [0, 1] using the given percentile bounds."""
    image = image.astype(np.float32, copy=False)
    lower_value = float(np.quantile(image, lower_quantile))
    upper_value = float(np.quantile(image, upper_quantile))
    if upper_value <= lower_value:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip(
        (image - lower_value) / (upper_value - lower_value),
        0.0,
        1.0,
    )


def load_slide_metadata(configuration: ApplicationConfiguration) -> SlideMetadata:
    """Read TIFF dimensions, OME pixel sizes, and marker names into a SlideMetadata."""
    marker_names = read_marker_names(configuration.input_paths.markers)
    with tifffile.TiffFile(configuration.input_paths.readouts) as tiff_file:
        primary_series = tiff_file.series[0]
        (
            physical_size_x_micrometers,
            physical_size_y_micrometers,
        ) = parse_open_microscopy_environment_pixel_size(tiff_file.ome_metadata)
        channel_count, height_pixels, width_pixels = primary_series.shape
    if physical_size_x_micrometers is None or physical_size_y_micrometers is None:
        raise ValueError(
            "Physical pixel size is missing from the open microscopy environment metadata. "
            "Add it to the source data before running the pipeline."
        )
    if len(marker_names) != channel_count:
        raise ValueError(
            "Marker count mismatch: markers.csv has "
            f"{len(marker_names)} rows but TIFF has {channel_count} channels."
        )
    return SlideMetadata(
        readouts_path=configuration.input_paths.readouts,
        histology_path=configuration.input_paths.histology,
        width_pixels=width_pixels,
        height_pixels=height_pixels,
        pixel_size_x_micrometers=physical_size_x_micrometers,
        pixel_size_y_micrometers=physical_size_y_micrometers,
        marker_names=marker_names,
    )


def parse_open_microscopy_environment_pixel_size(
    open_microscopy_environment_xml: str | None,
) -> tuple[float | None, float | None]:
    """Extract physical pixel size in micrometers from OME-XML metadata."""
    if not open_microscopy_environment_xml:
        return None, None
    xml_root = xml_element_tree.fromstring(open_microscopy_environment_xml)
    namespace = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    pixel_element = xml_root.find(
        ".//ome:Pixels",
        namespace,
    )
    if pixel_element is None:
        return None, None
    physical_size_x = pixel_element.attrib.get("PhysicalSizeX")
    physical_size_y = pixel_element.attrib.get("PhysicalSizeY")
    return (
        float(physical_size_x) if physical_size_x is not None else None,
        float(physical_size_y) if physical_size_y is not None else None,
    )


def read_readouts_region_of_interest(
    path: Path,
    region_of_interest: RegionOfInterestBox,
) -> Any:
    """Read a (channels, height, width) patch from a multiplex readouts TIFF."""
    return tifffile.imread(
        path,
        selection=(
            slice(None),
            slice(region_of_interest.y_pixels, region_of_interest.y_end_pixels),
            slice(region_of_interest.x_pixels, region_of_interest.x_end_pixels),
        ),
    )


def read_histology_region_of_interest(
    path: Path,
    region_of_interest: RegionOfInterestBox,
) -> Any:
    """Read a (height, width, 3) RGB patch from a registered histology TIFF."""
    return tifffile.imread(
        path,
        selection=(
            slice(region_of_interest.y_pixels, region_of_interest.y_end_pixels),
            slice(region_of_interest.x_pixels, region_of_interest.x_end_pixels),
            slice(None),
        ),
    )


def write_csv(
    data_frame: pl.DataFrame,
    path: Path,
) -> Path:
    """Write a Polars DataFrame to a CSV file, creating parent directories as needed."""
    ensure_directory(path.parent)
    data_frame.write_csv(path)
    return path


def write_yaml_file(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Serialize a dictionary to a YAML file."""
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(
            payload,
            file_handle,
            sort_keys=False,
        )


def write_image_stack(
    path: Path,
    image_stack: Any,
    marker_names: list[str],
) -> Path:
    """Write a multichannel image stack to a TIFF with CYX axes metadata."""
    ensure_directory(path.parent)
    tifffile.imwrite(
        path,
        image_stack,
        metadata={"axes": "CYX", "markers": marker_names},
    )
    return path


def write_label_array(
    path: Path,
    label_image: Any,
) -> Path:
    """Write a segmentation label array to a NumPy .npy file."""
    ensure_directory(path.parent)
    np.save(path, label_image)
    return path


def save_preprocessing_comparison(
    original_image: np.ndarray,
    corrected_image: np.ndarray,
    marker_name: str,
    path: Path,
) -> Path:
    """Save a side-by-side comparison of original and AF-corrected images as a PNG."""
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
    """Overlay colored segmentation labels onto a background image and save as TIFF."""
    background_rgb = normalize_background_for_overlay(background_image)
    overlay_image = color.label2rgb(
        label_image,
        image=background_rgb,
        bg_label=0,
        alpha=0.35,
    )
    overlay_uint8 = np.clip(overlay_image * 255.0, 0, 255).astype(np.uint8)
    ensure_directory(path.parent)
    tifffile.imwrite(path, overlay_uint8)
    return path


def save_cell_assignment_map(
    cell_annotations: pl.DataFrame,
    label_column_name: str,
    title: str,
    path: Path,
) -> Path:
    """Plot cell centroids colored by a categorical label column and save as PNG."""
    ensure_directory(path.parent)
    if cell_annotations.is_empty():
        return path
    labels = cell_annotations[label_column_name].to_list()
    unique_labels = sorted(set(labels))
    color_values = plt.cm.tab20(np.linspace(0, 1, max(len(unique_labels), 1)))
    color_map = ListedColormap(color_values)
    color_index_by_label = {label: index for index, label in enumerate(unique_labels)}
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(
        cell_annotations["x_micrometers"].to_numpy(),
        cell_annotations["y_micrometers"].to_numpy(),
        c=[color_index_by_label[label] for label in labels],
        cmap=color_map,
        s=6,
        alpha=0.8,
    )
    for label in unique_labels:
        label_rows = cell_annotations.filter(pl.col(label_column_name) == label)
        if label_rows.is_empty():
            continue
        x_position = float(np.median(label_rows["x_micrometers"].to_numpy()))
        y_position = float(np.median(label_rows["y_micrometers"].to_numpy()))
        axis.text(
            x_position,
            y_position,
            str(label),
            fontsize=8,
            ha="center",
            va="center",
            color="black",
            bbox={
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "none",
                "pad": 1.5,
            },
        )
    axis.set_title(title)
    axis.set_xlabel("x (μm)")
    axis.set_ylabel("y (μm)")
    axis.invert_yaxis()
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=color_values[color_index_by_label[label]],
            markeredgecolor="none",
            label=str(label),
        )
        for label in unique_labels
    ]
    axis.legend(
        handles=legend_handles,
        title=label_column_name.replace("_", " ").title(),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def normalize_background_for_overlay(background_image: np.ndarray) -> np.ndarray:
    """Normalize a grayscale or RGB image to float32 [0, 1] for label overlay compositing."""
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
