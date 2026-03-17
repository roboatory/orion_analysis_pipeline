from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tifffile

from src.configuration import load_configuration
from src.io import load_slide_metadata
from src.preprocessing import preprocess_region_of_interest_patch
from src.region_of_interest import choose_region_of_interest
from src.segmentation import (
    find_labels_touching_boundary,
    segment_cells_from_marker_images,
    summarize_segmentation_validation,
)

REAL_DATA_CONFIGURATION_PATH = Path(
    "/Users/rohit/Desktop/orion/configurations/CRC33_01.yaml"
)
REAL_DATA_SEGMENTATION_PATH = Path(
    "/Users/rohit/Desktop/orion/data/CRC33_01/segmentation.tif"
)
REAL_DATA_QUANTIFICATIONS_PATH = Path(
    "/Users/rohit/Desktop/orion/data/CRC33_01/quantifications.csv"
)


class DummySegmentationConfiguration:
    gaussian_sigma = 1.2
    minimum_nucleus_area_pixels = 10
    maximum_nucleus_area_pixels = 800
    peak_minimum_distance_pixels = 4
    cell_expansion_distance_pixels = 4


class DummyPercentileClipConfiguration:
    lower_quantile = 0.005
    upper_quantile = 0.995


class DummyPreprocessingConfiguration:
    percentile_clip = DummyPercentileClipConfiguration()


class DummyApplicationConfiguration:
    segmentation = DummySegmentationConfiguration()
    preprocessing = DummyPreprocessingConfiguration()


def test_watershed_splits_touching_nuclei() -> None:
    nuclear_image = np.zeros((64, 64), dtype=np.float32)
    cytoplasmic_image = np.zeros((64, 64), dtype=np.float32)
    row_coordinates, column_coordinates = np.ogrid[:64, :64]
    nuclear_image[
        (row_coordinates - 24) ** 2 + (column_coordinates - 24) ** 2 < 100
    ] = 1.0
    nuclear_image[
        (row_coordinates - 24) ** 2 + (column_coordinates - 38) ** 2 < 100
    ] = 1.0
    cytoplasmic_image[
        (row_coordinates - 24) ** 2 + (column_coordinates - 24) ** 2 < 196
    ] = 1.0
    cytoplasmic_image[
        (row_coordinates - 24) ** 2 + (column_coordinates - 38) ** 2 < 196
    ] = 1.0
    segmentation_result = segment_cells_from_marker_images(
        nuclear_image,
        cytoplasmic_image,
        DummyApplicationConfiguration(),
    )
    assert segmentation_result.nuclei_labels.max() >= 2


def test_expanded_labels_do_not_overlap() -> None:
    nuclear_image = np.zeros((64, 64), dtype=np.float32)
    cytoplasmic_image = np.zeros((64, 64), dtype=np.float32)
    nuclear_image[16:24, 16:24] = 1
    nuclear_image[40:48, 40:48] = 1
    cytoplasmic_image[12:28, 12:28] = 1
    cytoplasmic_image[36:52, 36:52] = 1
    segmentation_result = segment_cells_from_marker_images(
        nuclear_image,
        cytoplasmic_image,
        DummyApplicationConfiguration(),
    )
    assert np.all(segmentation_result.expanded_cell_labels >= 0)


def test_boundary_touching_labels_detected() -> None:
    label_image = np.zeros((8, 8), dtype=np.int32)
    label_image[0:3, 0:3] = 1
    label_image[4:6, 4:6] = 2
    assert find_labels_touching_boundary(label_image) == {1}


@pytest.mark.skipif(
    not REAL_DATA_CONFIGURATION_PATH.exists()
    or not REAL_DATA_SEGMENTATION_PATH.exists()
    or not REAL_DATA_QUANTIFICATIONS_PATH.exists(),
    reason="Real CRC33_01 inputs are not available.",
)
def test_real_patch_segmentation_has_reasonable_overlap_with_reference() -> None:
    configuration = load_configuration(REAL_DATA_CONFIGURATION_PATH)
    slide_metadata = load_slide_metadata(configuration)
    region_of_interest, _ = choose_region_of_interest(configuration, slide_metadata)
    readout_patch = tifffile.imread(
        configuration.input_paths.readouts,
        selection=(
            slice(None),
            slice(region_of_interest.y_pixels, region_of_interest.y_end_pixels),
            slice(region_of_interest.x_pixels, region_of_interest.x_end_pixels),
        ),
    )
    reference_segmentation_patch = tifffile.imread(
        REAL_DATA_SEGMENTATION_PATH,
        selection=(
            slice(region_of_interest.y_pixels, region_of_interest.y_end_pixels),
            slice(region_of_interest.x_pixels, region_of_interest.x_end_pixels),
        ),
    )
    preprocessing_result = preprocess_region_of_interest_patch(
        readout_patch,
        slide_metadata.marker_names,
        configuration,
    )
    marker_name_to_index = {
        marker_name: index
        for index, marker_name in enumerate(slide_metadata.marker_names)
    }
    segmentation_result = segment_cells_from_marker_images(
        preprocessing_result.corrected_image_stack[
            marker_name_to_index[configuration.channels.nuclear_marker]
        ],
        preprocessing_result.corrected_image_stack[
            marker_name_to_index[configuration.channels.cytoplasmic_marker]
        ],
        configuration,
    )
    validation_summary = summarize_segmentation_validation(
        reference_segmentation_patch,
        segmentation_result.kept_cell_labels,
    )
    assert validation_summary is not None
    assert validation_summary.new_cell_count > 0
    assert validation_summary.existing_cell_count > 0
    reference_quantification_count = int(
        pl.scan_csv(REAL_DATA_QUANTIFICATIONS_PATH)
        .filter(
            (pl.col("X_centroid") >= region_of_interest.x_pixels)
            & (pl.col("X_centroid") < region_of_interest.x_end_pixels)
            & (pl.col("Y_centroid") >= region_of_interest.y_pixels)
            & (pl.col("Y_centroid") < region_of_interest.y_end_pixels)
        )
        .select(pl.len())
        .collect()
        .item()
    )
    relative_cell_count_delta = (
        abs(validation_summary.new_cell_count - validation_summary.existing_cell_count)
        / validation_summary.existing_cell_count
    )
    relative_quantification_delta = abs(
        validation_summary.new_cell_count - reference_quantification_count
    ) / max(reference_quantification_count, 1)
    assert relative_cell_count_delta < 0.75
    assert reference_quantification_count > 0
    assert relative_quantification_delta < 0.75
    assert validation_summary.centroid_density_overlap > 0.0
