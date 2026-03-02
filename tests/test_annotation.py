import polars as pl

from orion.configuration import ApplicationConfiguration
from orion.annotation import annotate_cell_types


def build_configuration(
    cell_types: list[dict[str, object]],
) -> ApplicationConfiguration:
    return ApplicationConfiguration.model_validate(
        {
            "sample_identifier": "TEST",
            "input_paths": {
                "readouts": "data/CRC33_01/readouts.tiff",
                "markers": "data/CRC33_01/markers.csv",
                "histology": None,
                "existing_segmentation": None,
                "existing_quantifications": None,
            },
            "output_directory": "outputs",
            "channels": {
                "nuclear_marker": "Hoechst",
                "autofluorescence_marker": "AF1",
                "technical_markers": ["Hoechst", "AF1", "Argo550"],
            },
            "region_of_interest": {
                "selection_mode": "auto",
                "patch_width_pixels": 64,
                "patch_height_pixels": 64,
                "stride_pixels": 32,
                "minimum_cells": 10,
                "top_candidate_count_for_raw_quality_control": 5,
                "manual_override": None,
            },
            "preprocessing": {
                "autofluorescence_subtraction": {
                    "enabled": True,
                    "sample_pixels": 100,
                    "clip_upper_quantile": 0.95,
                },
                "percentile_clip": {"lower_quantile": 0.005, "upper_quantile": 0.995},
            },
            "segmentation": {
                "gaussian_sigma": 1.2,
                "minimum_nucleus_area_pixels": 10,
                "maximum_nucleus_area_pixels": 800,
                "peak_minimum_distance_pixels": 4,
                "cell_expansion_distance_pixels": 4,
            },
            "normalization": {
                "arcsinh_cofactor": 150.0,
                "threshold_method": "otsu_with_fallback",
                "positive_fraction_minimum": 0.005,
                "positive_fraction_maximum": 0.70,
                "fallback_quantile": 0.90,
            },
            "annotation": {"cell_types": cell_types},
            "spatial_analysis": {
                "nearest_neighbor_count": 2,
                "minimum_cells_per_type_for_pairwise_analysis": 1,
                "minimum_cells_per_type_for_clustering": 1,
                "permutation_count": 5,
                "neighborhood_cluster_count": 2,
            },
        }
    )


def test_single_marker_match_assigns_configured_cell_type() -> None:
    configuration = build_configuration(
        [{"name": "B_cell", "positive_markers": ["CD20"]}]
    )
    thresholded_data_frame = pl.DataFrame(
        [
            {
                "CD20_high": True,
            }
        ]
    )
    annotated_data_frame = annotate_cell_types(thresholded_data_frame, configuration)
    assert annotated_data_frame[0, "cell_type"] == "B_cell"
    assert annotated_data_frame[0, "matched_cell_type_count"] == 1


def test_multi_marker_rule_requires_all_positive_markers() -> None:
    configuration = build_configuration(
        [{"name": "Treg", "positive_markers": ["CD4", "FOXP3"]}]
    )
    thresholded_data_frame = pl.DataFrame(
        [
            {
                "CD4_high": True,
                "FOXP3_high": False,
            }
        ]
    )
    annotated_data_frame = annotate_cell_types(thresholded_data_frame, configuration)
    assert annotated_data_frame[0, "cell_type"] == "Unassigned"


def test_cell_with_no_matches_becomes_unassigned() -> None:
    configuration = build_configuration(
        [{"name": "Macrophage", "positive_markers": ["CD68"]}]
    )
    thresholded_data_frame = pl.DataFrame(
        [
            {
                "CD68_high": False,
            }
        ]
    )
    annotated_data_frame = annotate_cell_types(thresholded_data_frame, configuration)
    assert annotated_data_frame[0, "cell_type"] == "Unassigned"
    assert annotated_data_frame[0, "matched_cell_type_names"] == ""


def test_cell_matching_multiple_rules_becomes_ambiguous() -> None:
    configuration = build_configuration(
        [
            {"name": "B_cell", "positive_markers": ["CD20"]},
            {"name": "Plasma_like", "positive_markers": ["CD20"]},
        ]
    )
    thresholded_data_frame = pl.DataFrame([{"CD20_high": True}])
    annotated_data_frame = annotate_cell_types(thresholded_data_frame, configuration)
    assert annotated_data_frame[0, "cell_type"] == "Ambiguous"
    assert annotated_data_frame[0, "matched_cell_type_count"] == 2


def test_exclusive_marker_matching_rejects_other_cell_type_markers() -> None:
    configuration = build_configuration(
        [
            {"name": "B_cell", "positive_markers": ["CD20"]},
            {"name": "T_cell", "positive_markers": ["CD3e"]},
        ]
    )
    thresholded_data_frame = pl.DataFrame([{"CD20_high": True, "CD3e_high": True}])
    annotated_data_frame = annotate_cell_types(thresholded_data_frame, configuration)
    assert annotated_data_frame[0, "cell_type"] == "Unassigned"
    assert annotated_data_frame[0, "matched_cell_type_count"] == 0


def test_missing_thresholded_marker_column_fails_fast() -> None:
    configuration = build_configuration(
        [{"name": "B_cell", "positive_markers": ["CD20"]}]
    )
    thresholded_data_frame = pl.DataFrame([{"CD3e_high": True}])
    try:
        annotate_cell_types(thresholded_data_frame, configuration)
    except ValueError as value_error:
        assert "CD20" in str(value_error)
    else:
        raise AssertionError(
            "Expected annotation to fail when thresholded marker columns are missing."
        )
