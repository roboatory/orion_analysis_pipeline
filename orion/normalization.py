from __future__ import annotations

import numpy as np
import polars as pl
from skimage.filters import threshold_otsu

from orion.configuration import ApplicationConfiguration


def normalize_and_threshold_marker_intensities(
    raw_cell_measurements: pl.DataFrame,
    marker_names: list[str],
    configuration: ApplicationConfiguration,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    technical_marker_names = set(configuration.channels.technical_markers)
    biological_marker_names = [
        marker_name
        for marker_name in marker_names
        if marker_name not in technical_marker_names
    ]
    normalized_cell_measurements = raw_cell_measurements.clone()
    threshold_rows: list[dict[str, float | str]] = []

    for marker_name in biological_marker_names:
        normalized_column_name = f"{marker_name}_arcsinh"
        normalized_cell_measurements = normalized_cell_measurements.with_columns(
            (pl.col(marker_name) / configuration.normalization.arcsinh_cofactor)
            .arcsinh()
            .alias(normalized_column_name)
        )

    thresholded_cell_measurements = normalized_cell_measurements.clone()
    for marker_name in biological_marker_names:
        normalized_column_name = f"{marker_name}_arcsinh"
        normalized_values = normalized_cell_measurements.get_column(
            normalized_column_name
        ).to_numpy()
        threshold_value, threshold_method = compute_marker_threshold(
            normalized_values,
            configuration,
        )
        positive_fraction = (
            float((normalized_values >= threshold_value).mean())
            if len(normalized_values)
            else 0.0
        )
        thresholded_cell_measurements = thresholded_cell_measurements.with_columns(
            (pl.col(normalized_column_name) >= threshold_value).alias(
                f"{marker_name}_high"
            )
        )
        threshold_rows.append(
            {
                "marker_name": marker_name,
                "threshold_value": threshold_value,
                "threshold_method": threshold_method,
                "positive_fraction": positive_fraction,
            }
        )

    return (
        normalized_cell_measurements,
        thresholded_cell_measurements,
        pl.DataFrame(threshold_rows),
    )


def compute_marker_threshold(
    normalized_values: np.ndarray,
    configuration: ApplicationConfiguration,
) -> tuple[float, str]:
    if len(normalized_values) == 0:
        return 0.0, "empty"
    if np.allclose(normalized_values, normalized_values[0]):
        return float(normalized_values[0]), "constant"
    otsu_threshold = float(threshold_otsu(normalized_values))
    positive_fraction = float((normalized_values >= otsu_threshold).mean())
    if (
        positive_fraction < configuration.normalization.positive_fraction_minimum
        or positive_fraction > configuration.normalization.positive_fraction_maximum
    ):
        fallback_threshold = float(
            np.quantile(
                normalized_values,
                configuration.normalization.fallback_quantile,
            )
        )
        return fallback_threshold, "quantile_fallback"
    return otsu_threshold, "otsu"
