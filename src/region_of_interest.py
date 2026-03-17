from __future__ import annotations

import zlib

import numpy as np
import polars as pl
from skimage import measure, morphology
from skimage.filters import threshold_otsu

from src.configuration import ApplicationConfiguration
from src.data_models import RegionOfInterestBox, SlideMetadata
from src.io import read_readouts_region_of_interest


def choose_region_of_interest(
    configuration: ApplicationConfiguration,
    slide_metadata: SlideMetadata,
) -> tuple[RegionOfInterestBox, pl.DataFrame]:
    validate_region_of_interest_bounds(configuration, slide_metadata)
    random_seed = zlib.crc32(configuration.sample_identifier.encode("utf-8"))
    candidate_regions_of_interest = generate_candidate_regions_of_interest(
        configuration,
        slide_metadata,
        random_seed,
    )
    candidate_rows: list[dict[str, object]] = []

    for region_of_interest in candidate_regions_of_interest:
        readout_patch = read_readouts_region_of_interest(
            slide_metadata.readouts_path,
            region_of_interest,
        )
        quality_metrics = score_region_of_interest_patch(
            readout_patch,
            slide_metadata.marker_names,
            configuration,
        )
        passes_quality_thresholds = bool(
            quality_metrics["tissue_fraction"]
            >= configuration.region_of_interest.minimum_tissue_fraction
            and quality_metrics["informative_channel_fraction"]
            >= configuration.region_of_interest.minimum_informative_channel_fraction
        )
        candidate_rows.append(
            {
                "selection_mode": "seeded_random_best_of_n",
                "random_seed": random_seed,
                **region_of_interest.as_dictionary(),
                **quality_metrics,
                "passes_quality_thresholds": passes_quality_thresholds,
                "selected": False,
            }
        )

    candidate_data_frame = pl.DataFrame(candidate_rows).sort(
        ["passes_quality_thresholds", "quality_score"],
        descending=[True, True],
    )
    selected_index = 0
    candidate_data_frame = candidate_data_frame.with_columns(
        pl.Series(
            "selected",
            [
                row_index == selected_index
                for row_index in range(candidate_data_frame.height)
            ],
        )
    )
    selected_row = candidate_data_frame.row(0, named=True)
    selected_region_of_interest = RegionOfInterestBox(
        x_pixels=int(selected_row["x_pixels"]),
        y_pixels=int(selected_row["y_pixels"]),
        width_pixels=int(selected_row["width_pixels"]),
        height_pixels=int(selected_row["height_pixels"]),
    )
    return selected_region_of_interest, candidate_data_frame


def validate_region_of_interest_bounds(
    configuration: ApplicationConfiguration,
    slide_metadata: SlideMetadata,
) -> None:
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
    patch_width_pixels = configuration.region_of_interest.patch_width_pixels
    patch_height_pixels = configuration.region_of_interest.patch_height_pixels
    maximum_start_x_pixels = slide_metadata.width_pixels - patch_width_pixels
    maximum_start_y_pixels = slide_metadata.height_pixels - patch_height_pixels
    random_number_generator = np.random.default_rng(random_seed)
    candidate_regions_of_interest: list[RegionOfInterestBox] = []
    seen_coordinates: set[tuple[int, int]] = set()
    attempt_count = 0
    maximum_attempts = configuration.region_of_interest.candidate_patch_count * 10

    while (
        len(candidate_regions_of_interest)
        < configuration.region_of_interest.candidate_patch_count
        and attempt_count < maximum_attempts
    ):
        attempt_count += 1
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
    if not candidate_regions_of_interest:
        candidate_regions_of_interest.append(
            RegionOfInterestBox(
                x_pixels=0,
                y_pixels=0,
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
    marker_name_to_index = {
        marker_name: marker_index
        for marker_index, marker_name in enumerate(marker_names)
    }
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
    if float(np.max(normalized_nuclear_image)) <= 0.0:
        return 0.0
    try:
        threshold_value = float(threshold_otsu(normalized_nuclear_image))
    except ValueError:
        threshold_value = 0.0
    nuclear_mask = normalized_nuclear_image > threshold_value
    nuclear_mask = morphology.remove_small_objects(nuclear_mask, min_size=50)
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


def percentile_normalize_image(
    image: np.ndarray,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    lower_value = float(np.quantile(image, lower_quantile))
    upper_value = float(np.quantile(image, upper_quantile))
    if upper_value <= lower_value:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lower_value) / (upper_value - lower_value), 0.0, 1.0)
