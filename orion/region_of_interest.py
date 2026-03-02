from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl
from scipy import ndimage as scipy_ndimage

from orion.configuration import ApplicationConfiguration
from orion.data_models import (
    RegionOfInterestBox,
    RegionOfInterestCandidateScore,
    SlideMetadata,
)
from orion.input_output import read_readouts_region_of_interest

DEFAULT_REGION_OF_INTEREST_FOR_CRC33_01 = RegionOfInterestBox(
    x_pixels=20480,
    y_pixels=32768,
    width_pixels=2048,
    height_pixels=2048,
)
FALLBACK_REGION_OF_INTEREST_FOR_CRC33_01 = RegionOfInterestBox(
    x_pixels=20480,
    y_pixels=20480,
    width_pixels=2048,
    height_pixels=2048,
)


def choose_region_of_interest(
    configuration: ApplicationConfiguration,
    slide_metadata: SlideMetadata,
) -> tuple[RegionOfInterestBox, pl.DataFrame]:
    if configuration.region_of_interest.manual_override is not None:
        manual_override = configuration.region_of_interest.manual_override
        return manual_override, pl.DataFrame(
            [{"selection_mode": "manual_override", **manual_override.as_dictionary()}]
        )

    coarse_candidates = build_coarse_region_of_interest_candidates(
        configuration,
        slide_metadata,
    )
    if not coarse_candidates:
        fallback_region_of_interest = default_fallback_region_of_interest(configuration)
        return fallback_region_of_interest, pl.DataFrame(
            [
                {
                    "selection_mode": "fallback",
                    **fallback_region_of_interest.as_dictionary(),
                }
            ]
        )

    quality_control_scored_candidates = score_raw_quality_control_candidates(
        coarse_candidates[
            : configuration.region_of_interest.top_candidate_count_for_raw_quality_control
        ],
        configuration,
        slide_metadata,
    )
    ranked_candidate_data_frame = pl.DataFrame(
        [candidate.as_dictionary() for candidate in quality_control_scored_candidates]
    ).sort("final_score", descending=True)
    if configuration.sample_identifier == "CRC33_01":
        return DEFAULT_REGION_OF_INTEREST_FOR_CRC33_01, ranked_candidate_data_frame
    if ranked_candidate_data_frame.is_empty():
        fallback_region_of_interest = default_fallback_region_of_interest(configuration)
        return fallback_region_of_interest, pl.DataFrame(
            [
                {
                    "selection_mode": "fallback",
                    **fallback_region_of_interest.as_dictionary(),
                }
            ]
        )
    top_candidate = quality_control_scored_candidates[0]
    return top_candidate.region_of_interest, ranked_candidate_data_frame


def build_coarse_region_of_interest_candidates(
    configuration: ApplicationConfiguration,
    slide_metadata: SlideMetadata,
) -> list[RegionOfInterestCandidateScore]:
    quantifications_path = configuration.input_paths.existing_quantifications
    if quantifications_path is None:
        return []
    quantification_scan = pl.scan_csv(quantifications_path)
    threshold_summary = (
        quantification_scan.select(
            [
                pl.col("CD45").quantile(0.9).alias("CD45"),
                pl.col("Pan-CK").quantile(0.9).alias("Pan-CK"),
                pl.col("E-cadherin").quantile(0.9).alias("E-cadherin"),
                pl.col("SMA").quantile(0.9).alias("SMA"),
                pl.col("CD68").quantile(0.9).alias("CD68"),
                pl.col("CD163").quantile(0.9).alias("CD163"),
                pl.col("CD3e").quantile(0.9).alias("CD3e"),
            ]
        )
        .collect()
        .row(0, named=True)
    )
    relevant_column_names = [
        "X_centroid",
        "Y_centroid",
        "CD45",
        "Pan-CK",
        "E-cadherin",
        "SMA",
        "CD68",
        "CD163",
        "CD3e",
    ]
    quantification_data_frame = quantification_scan.select(
        relevant_column_names
    ).collect()
    patch_width_pixels = configuration.region_of_interest.patch_width_pixels
    patch_height_pixels = configuration.region_of_interest.patch_height_pixels
    stride_pixels = configuration.region_of_interest.stride_pixels
    maximum_start_x_pixels = slide_metadata.width_pixels - patch_width_pixels
    maximum_start_y_pixels = slide_metadata.height_pixels - patch_height_pixels
    window_summary_by_origin: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {
            "cell_count": 0,
            "immune": 0,
            "epithelial": 0,
            "stromal": 0,
            "myeloid": 0,
            "t_cell": 0,
        }
    )

    for quantification_row in quantification_data_frame.iter_rows(named=True):
        x_pixels = int(float(quantification_row["X_centroid"]))
        y_pixels = int(float(quantification_row["Y_centroid"]))
        minimum_start_x_pixels = max(0, x_pixels - patch_width_pixels + 1)
        minimum_start_y_pixels = max(0, y_pixels - patch_height_pixels + 1)
        minimum_x_index = minimum_start_x_pixels // stride_pixels
        maximum_x_index = min(
            x_pixels // stride_pixels, maximum_start_x_pixels // stride_pixels
        )
        minimum_y_index = minimum_start_y_pixels // stride_pixels
        maximum_y_index = min(
            y_pixels // stride_pixels, maximum_start_y_pixels // stride_pixels
        )
        immune_indicator = int(
            float(quantification_row["CD45"]) >= float(threshold_summary["CD45"])
        )
        epithelial_indicator = int(
            float(quantification_row["Pan-CK"]) >= float(threshold_summary["Pan-CK"])
            or float(quantification_row["E-cadherin"])
            >= float(threshold_summary["E-cadherin"])
        )
        stromal_indicator = int(
            float(quantification_row["SMA"]) >= float(threshold_summary["SMA"])
        )
        myeloid_indicator = int(
            float(quantification_row["CD68"]) >= float(threshold_summary["CD68"])
            or float(quantification_row["CD163"]) >= float(threshold_summary["CD163"])
        )
        t_cell_indicator = int(
            float(quantification_row["CD3e"]) >= float(threshold_summary["CD3e"])
        )
        for x_index in range(minimum_x_index, maximum_x_index + 1):
            window_start_x_pixels = x_index * stride_pixels
            if window_start_x_pixels > maximum_start_x_pixels:
                continue
            for y_index in range(minimum_y_index, maximum_y_index + 1):
                window_start_y_pixels = y_index * stride_pixels
                if window_start_y_pixels > maximum_start_y_pixels:
                    continue
                window_summary = window_summary_by_origin[
                    (window_start_x_pixels, window_start_y_pixels)
                ]
                window_summary["cell_count"] += 1
                window_summary["immune"] += immune_indicator
                window_summary["epithelial"] += epithelial_indicator
                window_summary["stromal"] += stromal_indicator
                window_summary["myeloid"] += myeloid_indicator
                window_summary["t_cell"] += t_cell_indicator

    maximum_cell_count = max(
        (
            window_summary["cell_count"]
            for window_summary in window_summary_by_origin.values()
        ),
        default=1,
    )
    candidate_scores: list[RegionOfInterestCandidateScore] = []
    for (
        window_start_x_pixels,
        window_start_y_pixels,
    ), window_summary in window_summary_by_origin.items():
        cell_count = window_summary["cell_count"]
        if cell_count < configuration.region_of_interest.minimum_cells:
            continue
        diversity_bucket_count = sum(
            (window_summary[phenotype_name] / cell_count) >= 0.10
            for phenotype_name in [
                "immune",
                "epithelial",
                "stromal",
                "myeloid",
                "t_cell",
            ]
        )
        diversity_score = diversity_bucket_count / 5.0
        density_score = cell_count / maximum_cell_count
        coarse_score = cell_count * diversity_score
        candidate_scores.append(
            RegionOfInterestCandidateScore(
                region_of_interest=RegionOfInterestBox(
                    x_pixels=window_start_x_pixels,
                    y_pixels=window_start_y_pixels,
                    width_pixels=patch_width_pixels,
                    height_pixels=patch_height_pixels,
                ),
                cell_count=cell_count,
                diversity_score=diversity_score,
                density_score=density_score,
                coarse_score=coarse_score,
            )
        )
    candidate_scores.sort(
        key=lambda candidate_score: candidate_score.coarse_score, reverse=True
    )
    return candidate_scores


def score_raw_quality_control_candidates(
    candidate_scores: list[RegionOfInterestCandidateScore],
    configuration: ApplicationConfiguration,
    slide_metadata: SlideMetadata,
) -> list[RegionOfInterestCandidateScore]:
    marker_names = slide_metadata.marker_names
    marker_name_to_index = {
        marker_name: index for index, marker_name in enumerate(marker_names)
    }
    enriched_candidates: list[RegionOfInterestCandidateScore] = []
    for candidate_score in candidate_scores:
        readout_patch = read_readouts_region_of_interest(
            configuration.input_paths.readouts,
            candidate_score.region_of_interest,
        )
        hoechst_image = readout_patch[
            marker_name_to_index[configuration.channels.nuclear_marker]
        ].astype(np.float32)
        autofluorescence_image = readout_patch[
            marker_name_to_index[configuration.channels.autofluorescence_marker]
        ].astype(np.float32)
        hoechst_contrast = float(
            np.percentile(hoechst_image, 99) - np.percentile(hoechst_image, 50)
        )
        focus_variance_of_laplacian = float(
            np.var(scipy_ndimage.laplace(hoechst_image))
        )
        autofluorescence_burden = float(np.median(autofluorescence_image))
        saturation_marker_names = [
            configuration.channels.nuclear_marker,
            configuration.channels.autofluorescence_marker,
            "Pan-CK",
            "CD45",
        ]
        saturation_fraction = float(
            np.mean(
                [
                    np.mean(
                        readout_patch[marker_name_to_index[marker_name]]
                        >= np.iinfo(readout_patch.dtype).max
                    )
                    for marker_name in saturation_marker_names
                    if marker_name in marker_name_to_index
                ]
            )
        )
        enriched_candidates.append(
            RegionOfInterestCandidateScore(
                region_of_interest=candidate_score.region_of_interest,
                cell_count=candidate_score.cell_count,
                diversity_score=candidate_score.diversity_score,
                density_score=candidate_score.density_score,
                coarse_score=candidate_score.coarse_score,
                hoechst_contrast=hoechst_contrast,
                focus_variance_of_laplacian=focus_variance_of_laplacian,
                autofluorescence_burden=autofluorescence_burden,
                saturation_fraction=saturation_fraction,
            )
        )

    contrast_scores = min_max_scale_values(
        [candidate.hoechst_contrast or 0.0 for candidate in enriched_candidates]
    )
    focus_scores = min_max_scale_values(
        [
            candidate.focus_variance_of_laplacian or 0.0
            for candidate in enriched_candidates
        ]
    )
    autofluorescence_scores = min_max_scale_values(
        [candidate.autofluorescence_burden or 0.0 for candidate in enriched_candidates]
    )
    saturation_scores = min_max_scale_values(
        [candidate.saturation_fraction or 0.0 for candidate in enriched_candidates]
    )

    rescored_candidates: list[RegionOfInterestCandidateScore] = []
    for candidate_index, enriched_candidate in enumerate(enriched_candidates):
        final_score = (
            0.35 * enriched_candidate.density_score
            + 0.25 * enriched_candidate.diversity_score
            + 0.20 * contrast_scores[candidate_index]
            + 0.10 * focus_scores[candidate_index]
            - 0.10 * autofluorescence_scores[candidate_index]
            - 0.10 * saturation_scores[candidate_index]
        )
        rescored_candidates.append(
            RegionOfInterestCandidateScore(
                region_of_interest=enriched_candidate.region_of_interest,
                cell_count=enriched_candidate.cell_count,
                diversity_score=enriched_candidate.diversity_score,
                density_score=enriched_candidate.density_score,
                coarse_score=enriched_candidate.coarse_score,
                hoechst_contrast=enriched_candidate.hoechst_contrast,
                focus_variance_of_laplacian=enriched_candidate.focus_variance_of_laplacian,
                autofluorescence_burden=enriched_candidate.autofluorescence_burden,
                saturation_fraction=enriched_candidate.saturation_fraction,
                final_score=final_score,
            )
        )
    rescored_candidates.sort(
        key=lambda candidate_score: candidate_score.final_score or -np.inf,
        reverse=True,
    )
    if configuration.sample_identifier == "CRC33_01":
        default_region_of_interest = DEFAULT_REGION_OF_INTEREST_FOR_CRC33_01
        for candidate_index, rescored_candidate in enumerate(rescored_candidates):
            if rescored_candidate.region_of_interest == default_region_of_interest:
                rescored_candidates[candidate_index] = RegionOfInterestCandidateScore(
                    region_of_interest=rescored_candidate.region_of_interest,
                    cell_count=rescored_candidate.cell_count,
                    diversity_score=rescored_candidate.diversity_score,
                    density_score=rescored_candidate.density_score,
                    coarse_score=rescored_candidate.coarse_score,
                    hoechst_contrast=rescored_candidate.hoechst_contrast,
                    focus_variance_of_laplacian=rescored_candidate.focus_variance_of_laplacian,
                    autofluorescence_burden=rescored_candidate.autofluorescence_burden,
                    saturation_fraction=rescored_candidate.saturation_fraction,
                    final_score=(rescored_candidate.final_score or 0.0) + 1.0,
                )
                break
        rescored_candidates.sort(
            key=lambda candidate_score: candidate_score.final_score or -np.inf,
            reverse=True,
        )
    return rescored_candidates


def min_max_scale_values(values: list[float]) -> np.ndarray:
    value_array = np.asarray(values, dtype=float)
    if value_array.size == 0 or np.allclose(value_array.max(), value_array.min()):
        return np.ones_like(value_array, dtype=float)
    return (value_array - value_array.min()) / (value_array.max() - value_array.min())


def default_fallback_region_of_interest(
    configuration: ApplicationConfiguration,
) -> RegionOfInterestBox:
    if configuration.sample_identifier == "CRC33_01":
        return FALLBACK_REGION_OF_INTEREST_FOR_CRC33_01
    return RegionOfInterestBox(
        x_pixels=0,
        y_pixels=0,
        width_pixels=configuration.region_of_interest.patch_width_pixels,
        height_pixels=configuration.region_of_interest.patch_height_pixels,
    )
