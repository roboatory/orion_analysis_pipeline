from __future__ import annotations

import numpy as np
import polars as pl
from scipy import ndimage as scipy_ndimage
from skimage import measure

from src.data_models import RegionOfInterestBox


def quantify_cells_in_region_of_interest(
    label_image: np.ndarray,
    intensity_image_by_marker: dict[str, np.ndarray],
    marker_names: list[str],
    region_of_interest: RegionOfInterestBox,
) -> pl.DataFrame:
    """Measure per-cell morphology and mean marker intensities from a segmentation mask."""
    region_properties = measure.regionprops(label_image)
    columns: dict[str, list[float | int]] = {
        "cell_identifier": [],
        "x_pixels": [],
        "y_pixels": [],
        "area_square_pixels": [],
        "major_axis_length_pixels": [],
        "minor_axis_length_pixels": [],
        "eccentricity": [],
        "solidity": [],
        "extent": [],
        "orientation_degrees": [],
    }
    for marker_name in marker_names:
        columns[marker_name] = []

    if not region_properties:
        return pl.DataFrame(columns)

    label_identifiers = np.arange(
        1,
        label_image.max() + 1,
        dtype=np.int32,
    )
    mean_intensity_by_marker = {
        marker_name: scipy_ndimage.mean(
            image,
            labels=label_image,
            index=label_identifiers,
        )
        for marker_name, image in intensity_image_by_marker.items()
    }

    for region_properties_entry in region_properties:
        label_identifier = int(region_properties_entry.label)
        centroid_y_pixels, centroid_x_pixels = region_properties_entry.centroid
        columns["cell_identifier"].append(label_identifier)
        columns["x_pixels"].append(
            float(centroid_x_pixels + region_of_interest.x_pixels)
        )
        columns["y_pixels"].append(
            float(centroid_y_pixels + region_of_interest.y_pixels)
        )
        columns["area_square_pixels"].append(float(region_properties_entry.area))
        columns["major_axis_length_pixels"].append(
            float(region_properties_entry.axis_major_length)
        )
        columns["minor_axis_length_pixels"].append(
            float(region_properties_entry.axis_minor_length)
        )
        columns["eccentricity"].append(float(region_properties_entry.eccentricity))
        columns["solidity"].append(float(region_properties_entry.solidity))
        columns["extent"].append(float(region_properties_entry.extent))
        columns["orientation_degrees"].append(
            float(np.degrees(region_properties_entry.orientation))
        )
        for marker_name in marker_names:
            columns[marker_name].append(
                float(mean_intensity_by_marker[marker_name][label_identifier - 1])
            )

    return pl.DataFrame(columns)
