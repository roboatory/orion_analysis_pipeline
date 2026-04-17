from __future__ import annotations

import numpy as np
import polars as pl
from skimage.filters import threshold_otsu

from src.configuration import ApplicationConfiguration


def annotate_cells(
    cell_features: pl.DataFrame,
    configuration: ApplicationConfiguration,
) -> pl.DataFrame:
    """Assign cell type labels using arcsinh-normalized marker thresholds and boolean rules."""
    validate_annotation_marker_columns(
        cell_features,
        configuration,
    )
    annotation_marker_names = configuration.annotation_marker_names
    annotated_cell_measurements = cell_features.clone()

    for marker_name in annotation_marker_names:
        normalized_column_name = f"{marker_name}_arcsinh"
        threshold_column_name = f"{marker_name}_high"
        annotated_cell_measurements = annotated_cell_measurements.with_columns(
            (pl.col(marker_name) / configuration.normalization.arcsinh_cofactor)
            .arcsinh()
            .alias(normalized_column_name)
        )
        normalized_values = annotated_cell_measurements[
            normalized_column_name
        ].to_numpy()
        threshold_value = compute_marker_threshold(
            normalized_values,
            configuration,
        )
        annotated_cell_measurements = annotated_cell_measurements.with_columns(
            (pl.col(normalized_column_name) >= threshold_value).alias(
                threshold_column_name
            )
        )

    thresholded_rows = annotated_cell_measurements.to_dicts()
    annotation_rows: list[dict[str, object]] = []
    cell_type_rules = configuration.annotation.cell_types
    for thresholded_row in thresholded_rows:
        cell_type = "unassigned"
        for rule in cell_type_rules:
            positives_match = all(
                bool(thresholded_row[f"{marker_name}_high"])
                for marker_name in rule.positive_markers
            )
            negatives_match = all(
                not bool(thresholded_row[f"{marker_name}_high"])
                for marker_name in rule.negative_markers
            )
            if positives_match and negatives_match:
                cell_type = rule.name
                break
        annotation_row = {
            "cell_identifier": thresholded_row["cell_identifier"],
            "x_micrometers": thresholded_row["x_micrometers"],
            "y_micrometers": thresholded_row["y_micrometers"],
            "cell_type": cell_type,
        }
        for marker_name in annotation_marker_names:
            annotation_row[f"{marker_name}_high"] = bool(
                thresholded_row[f"{marker_name}_high"]
            )
        annotation_rows.append(annotation_row)
    return pl.DataFrame(annotation_rows)


def validate_annotation_marker_columns(
    cell_features: pl.DataFrame,
    configuration: ApplicationConfiguration,
) -> None:
    """Raise if any annotation marker is missing from the cell features columns."""
    available_columns = set(cell_features.columns)
    missing_marker_names = [
        marker_name
        for marker_name in configuration.annotation_marker_names
        if marker_name not in available_columns
    ]
    if missing_marker_names:
        formatted_missing_marker_names = ", ".join(sorted(missing_marker_names))
        raise ValueError(
            "Configured annotation markers are missing from the quantified cell features: "
            f"{formatted_missing_marker_names}."
        )


def compute_marker_threshold(
    normalized_values: np.ndarray,
    configuration: ApplicationConfiguration,
) -> float:
    """Determine a positive/negative threshold using Otsu with a quantile fallback."""
    if len(normalized_values) == 0:
        return 0.0
    if np.allclose(normalized_values, normalized_values[0]):
        return float(normalized_values[0])
    otsu_threshold = float(threshold_otsu(normalized_values))
    positive_fraction = float((normalized_values >= otsu_threshold).mean())
    if (
        positive_fraction < configuration.normalization.positive_fraction_minimum
        or positive_fraction > configuration.normalization.positive_fraction_maximum
    ):
        return float(
            np.quantile(
                normalized_values,
                configuration.normalization.fallback_quantile,
            )
        )
    return otsu_threshold
