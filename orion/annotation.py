from __future__ import annotations

import polars as pl

from orion.configuration import ApplicationConfiguration


def annotate_cell_types(
    thresholded_cell_measurements: pl.DataFrame,
    configuration: ApplicationConfiguration,
) -> pl.DataFrame:
    validate_thresholded_annotation_columns(
        thresholded_cell_measurements,
        configuration,
    )
    thresholded_rows = thresholded_cell_measurements.to_dicts()
    annotated_rows: list[dict[str, object]] = []
    marker_names_by_cell_type = {
        cell_type_rule.name: list(cell_type_rule.positive_markers)
        for cell_type_rule in configuration.annotation.cell_types
    }
    for thresholded_row in thresholded_rows:
        matched_cell_type_names = []
        for cell_type_rule in configuration.annotation.cell_types:
            own_marker_names = marker_names_by_cell_type[cell_type_rule.name]
            own_markers_positive = all(
                bool(thresholded_row[f"{marker_name}_high"])
                for marker_name in own_marker_names
            )
            other_marker_names = [
                marker_name
                for other_cell_type_name, other_marker_names in marker_names_by_cell_type.items()
                if other_cell_type_name != cell_type_rule.name
                for marker_name in other_marker_names
                if marker_name not in own_marker_names
            ]
            other_markers_negative = all(
                not bool(thresholded_row[f"{marker_name}_high"])
                for marker_name in other_marker_names
            )
            if own_markers_positive and other_markers_negative:
                matched_cell_type_names.append(cell_type_rule.name)

        matched_cell_type_count = len(matched_cell_type_names)
        if matched_cell_type_count == 1:
            cell_type = matched_cell_type_names[0]
        elif matched_cell_type_count > 1:
            cell_type = "Ambiguous"
        else:
            cell_type = "Unassigned"

        thresholded_row["matched_cell_type_names"] = "|".join(matched_cell_type_names)
        thresholded_row["matched_cell_type_count"] = matched_cell_type_count
        thresholded_row["cell_type"] = cell_type
        thresholded_row["cell_subtype"] = ""
        annotated_rows.append(thresholded_row)
    return pl.DataFrame(annotated_rows)


def validate_thresholded_annotation_columns(
    thresholded_cell_measurements: pl.DataFrame,
    configuration: ApplicationConfiguration,
) -> None:
    available_columns = set(thresholded_cell_measurements.columns)
    missing_columns = [
        f"{marker_name}_high"
        for marker_name in configuration.annotation_marker_names
        if f"{marker_name}_high" not in available_columns
    ]
    if missing_columns:
        missing_marker_names = ", ".join(
            sorted(column_name.removesuffix("_high") for column_name in missing_columns)
        )
        raise ValueError(
            "Configured annotation markers did not produce thresholded columns: "
            f"{missing_marker_names}. Check technical marker settings or the normalization stage."
        )
