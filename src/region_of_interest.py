from __future__ import annotations

import zlib
from dataclasses import asdict

import numpy as np
import polars as pl
from skimage import measure, morphology
from skimage.filters import threshold_otsu

from src.configuration import ApplicationConfiguration
from src.data_models import RegionOfInterestBox, SlideMetadata
from src.io import (
    build_marker_name_to_index,
    percentile_normalize_image,
    read_readouts_region_of_interest,
)


def choose_region_of_interest(
    configuration: ApplicationConfiguration,
    slide_metadata: SlideMetadata,
) -> tuple[list[RegionOfInterestBox], pl.DataFrame]:
    """Select the top non-overlapping analysis patches from the scored candidate pool."""
    validate_region_of_interest_bounds(
        configuration,
        slide_metadata,
    )
    random_seed = zlib.crc32(configuration.sample_identifier.encode("utf-8"))
    candidate_regions_of_interest = generate_candidate_regions_of_interest(
        configuration,
        slide_metadata,
        random_seed,
    )
    candidate_rows = [
        build_candidate_row(
            configuration,
            slide_metadata,
            region_of_interest,
            random_seed,
        )
        for region_of_interest in candidate_regions_of_interest
    ]
    sorted_candidate_data_frame = sort_candidate_data_frame(candidate_rows)
    selected_row_indices = select_non_overlapping_top_rows(
        sorted_candidate_data_frame,
        configuration.region_of_interest.analysis_patch_count,
    )
    candidate_data_frame = mark_selected_rows(
        sorted_candidate_data_frame,
        selected_row_indices,
    )
    selected_regions_of_interest = [
        build_region_of_interest_box_from_row(
            candidate_data_frame.row(row_index, named=True)
        )
        for row_index in selected_row_indices
    ]
    return selected_regions_of_interest, candidate_data_frame


def build_region_of_interest_box_from_row(
    row: dict[str, object],
) -> RegionOfInterestBox:
    """Build a RegionOfInterestBox from a candidate dataframe row."""
    return RegionOfInterestBox(
        x_pixels=int(row["x_pixels"]),
        y_pixels=int(row["y_pixels"]),
        width_pixels=int(row["width_pixels"]),
        height_pixels=int(row["height_pixels"]),
    )


def build_candidate_row(
    configuration: ApplicationConfiguration,
    slide_metadata: SlideMetadata,
    region_of_interest: RegionOfInterestBox,
    random_seed: int,
) -> dict[str, object]:
    """Score a single candidate ROI and return its metadata and quality metrics."""
    readout_patch = read_readouts_region_of_interest(
        slide_metadata.readouts_path,
        region_of_interest,
    )
    quality_metrics = score_region_of_interest_patch(
        readout_patch,
        slide_metadata.marker_names,
        configuration,
    )
    return {
        "selection_mode": "seeded_random_best_of_n",
        "random_seed": random_seed,
        **asdict(region_of_interest),
        **quality_metrics,
        "passes_quality_thresholds": passes_quality_thresholds(
            quality_metrics,
            configuration,
        ),
        "selected": False,
    }


def passes_quality_thresholds(
    quality_metrics: dict[str, float],
    configuration: ApplicationConfiguration,
) -> bool:
    """Check whether tissue and informative channel fractions meet configured minimums."""
    return bool(
        quality_metrics["tissue_fraction"]
        >= configuration.region_of_interest.minimum_tissue_fraction
        and quality_metrics["informative_channel_fraction"]
        >= configuration.region_of_interest.minimum_informative_channel_fraction
    )


def sort_candidate_data_frame(
    candidate_rows: list[dict[str, object]],
) -> pl.DataFrame:
    """Sort candidate rows by pass/quality order without marking selections."""
    return pl.DataFrame(candidate_rows).sort(
        ["passes_quality_thresholds", "quality_score"],
        descending=[True, True],
    )


def select_non_overlapping_top_rows(
    sorted_candidate_data_frame: pl.DataFrame,
    analysis_patch_count: int,
) -> list[int]:
    """Greedily pick the top N candidate row indices whose bounding boxes do not overlap."""
    selected_row_indices: list[int] = []
    selected_boxes: list[RegionOfInterestBox] = []
    for row_index in range(sorted_candidate_data_frame.height):
        if len(selected_row_indices) == analysis_patch_count:
            break
        candidate_box = build_region_of_interest_box_from_row(
            sorted_candidate_data_frame.row(row_index, named=True)
        )
        if any(
            boxes_overlap(candidate_box, selected_box)
            for selected_box in selected_boxes
        ):
            continue
        selected_row_indices.append(row_index)
        selected_boxes.append(candidate_box)
    if len(selected_row_indices) < analysis_patch_count:
        raise ValueError(
            "Unable to select "
            f"{analysis_patch_count} non-overlapping patches from the candidate pool. "
            "Increase region_of_interest.candidate_patch_count or decrease "
            "region_of_interest.analysis_patch_count."
        )
    return selected_row_indices


def mark_selected_rows(
    sorted_candidate_data_frame: pl.DataFrame,
    selected_row_indices: list[int],
) -> pl.DataFrame:
    """Return a new dataframe with a boolean 'selected' column flagging the given rows."""
    selected_index_set = set(selected_row_indices)
    return sorted_candidate_data_frame.with_columns(
        pl.Series(
            "selected",
            [
                row_index in selected_index_set
                for row_index in range(sorted_candidate_data_frame.height)
            ],
        )
    )


def boxes_overlap(
    first_box: RegionOfInterestBox,
    second_box: RegionOfInterestBox,
) -> bool:
    """Return True if two bounding boxes share any pixel (half-open intervals)."""
    horizontal_overlap = (
        first_box.x_pixels < second_box.x_end_pixels
        and second_box.x_pixels < first_box.x_end_pixels
    )
    vertical_overlap = (
        first_box.y_pixels < second_box.y_end_pixels
        and second_box.y_pixels < first_box.y_end_pixels
    )
    return horizontal_overlap and vertical_overlap


def validate_region_of_interest_bounds(
    configuration: ApplicationConfiguration,
    slide_metadata: SlideMetadata,
) -> None:
    """Raise if the configured patch dimensions exceed the slide dimensions."""
    patch_width_pixels = configuration.region_of_interest.patch_width_pixels
    patch_height_pixels = configuration.region_of_interest.patch_height_pixels
    if patch_width_pixels > slide_metadata.width_pixels:
        raise ValueError(
            "Patch width exceeds image width. Reduce region_of_interest.patch_width_pixels."
        )
    if patch_height_pixels > slide_metadata.height_pixels:
        raise ValueError(
            "Patch height exceeds image height. Reduce region_of_interest.patch_height_pixels."
        )


def generate_candidate_regions_of_interest(
    configuration: ApplicationConfiguration,
    slide_metadata: SlideMetadata,
    random_seed: int,
) -> list[RegionOfInterestBox]:
    """Sample random non-duplicate candidate patch positions across the slide."""
    patch_width_pixels = configuration.region_of_interest.patch_width_pixels
    patch_height_pixels = configuration.region_of_interest.patch_height_pixels
    maximum_start_x_pixels = slide_metadata.width_pixels - patch_width_pixels
    maximum_start_y_pixels = slide_metadata.height_pixels - patch_height_pixels
    random_number_generator = np.random.default_rng(random_seed)
    candidate_regions_of_interest: list[RegionOfInterestBox] = []
    seen_coordinates: set[tuple[int, int]] = set()
    maximum_attempts = configuration.region_of_interest.candidate_patch_count

    while len(candidate_regions_of_interest) < maximum_attempts:
        x_pixels = (
            int(random_number_generator.integers(0, maximum_start_x_pixels + 1))
            if maximum_start_x_pixels > 0
            else 0
        )
        y_pixels = (
            int(random_number_generator.integers(0, maximum_start_y_pixels + 1))
            if maximum_start_y_pixels > 0
            else 0
        )

        coordinate_key = (x_pixels, y_pixels)
        if coordinate_key in seen_coordinates:
            continue
        seen_coordinates.add(coordinate_key)
        candidate_regions_of_interest.append(
            RegionOfInterestBox(
                x_pixels=x_pixels,
                y_pixels=y_pixels,
                width_pixels=patch_width_pixels,
                height_pixels=patch_height_pixels,
            )
        )

    return candidate_regions_of_interest


def score_region_of_interest_patch(
    readout_patch: np.ndarray,
    marker_names: list[str],
    configuration: ApplicationConfiguration,
) -> dict[str, float]:
    """Compute tissue fraction, informative channel fraction, and combined quality score."""
    marker_name_to_index = build_marker_name_to_index(marker_names)
    tissue_fraction = compute_tissue_fraction(
        readout_patch[marker_name_to_index[configuration.channels.nuclear_marker]],
        readout_patch[marker_name_to_index[configuration.channels.cytoplasmic_marker]],
    )
    informative_channel_fraction = compute_informative_channel_fraction(
        readout_patch,
        marker_names,
        configuration,
    )
    quality_score = 0.6 * tissue_fraction + 0.4 * informative_channel_fraction
    return {
        "tissue_fraction": float(tissue_fraction),
        "informative_channel_fraction": float(informative_channel_fraction),
        "quality_score": float(quality_score),
    }


def compute_tissue_fraction(
    nuclear_image: np.ndarray,
    cytoplasmic_image: np.ndarray,
) -> float:
    """Estimate the fraction of the patch covered by tissue using Otsu thresholding."""
    normalized_nuclear_image = percentile_normalize_image(nuclear_image)
    normalized_cytoplasmic_image = percentile_normalize_image(cytoplasmic_image)
    tissue_proxy_image = (normalized_nuclear_image + normalized_cytoplasmic_image) / 2.0
    if float(np.max(tissue_proxy_image)) <= 0.0:
        return 0.0
    try:
        threshold_value = float(threshold_otsu(tissue_proxy_image))
    except ValueError:
        threshold_value = 0.0
    tissue_mask = tissue_proxy_image > threshold_value
    tissue_mask_fraction = float(np.mean(tissue_mask))
    nuclear_component_score = compute_nuclear_component_score(normalized_nuclear_image)
    return float((0.5 * tissue_mask_fraction) + (0.5 * nuclear_component_score))


def compute_nuclear_component_score(normalized_nuclear_image: np.ndarray) -> float:
    """Score nuclear signal richness as the ratio of detected components to expected count."""
    if float(np.max(normalized_nuclear_image)) <= 0.0:
        return 0.0
    try:
        threshold_value = float(threshold_otsu(normalized_nuclear_image))
    except ValueError:
        threshold_value = 0.0
    nuclear_mask = normalized_nuclear_image > threshold_value
    nuclear_mask = morphology.remove_small_objects(
        nuclear_mask,
        min_size=50,
    )
    component_count = int(measure.label(nuclear_mask).max())
    expected_component_count = max(
        normalized_nuclear_image.size / 4096.0,
        1.0,
    )
    return float(min(component_count / expected_component_count, 1.0))


def compute_informative_channel_fraction(
    readout_patch: np.ndarray,
    marker_names: list[str],
    configuration: ApplicationConfiguration,
) -> float:
    """Return the fraction of biological channels with sufficient signal spread."""
    informative_channel_count = 0
    biological_marker_count = 0

    for marker_index, marker_name in enumerate(marker_names):
        if marker_name == configuration.channels.autofluorescence_marker:
            continue
        biological_marker_count += 1
        normalized_channel = percentile_normalize_image(readout_patch[marker_index])
        signal_spread = float(
            np.percentile(normalized_channel, 99)
            - np.percentile(normalized_channel, 50)
        )
        if (
            signal_spread
            >= configuration.region_of_interest.minimum_channel_signal_spread
        ):
            informative_channel_count += 1

    if biological_marker_count == 0:
        return 0.0
    return float(informative_channel_count / biological_marker_count)
