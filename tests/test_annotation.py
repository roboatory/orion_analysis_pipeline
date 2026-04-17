import polars as pl

from src.annotation import annotate_cells
from src.configuration import ApplicationConfiguration


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
            },
            "output_directory": "outputs",
            "channels": {
                "nuclear_marker": "Hoechst",
                "cytoplasmic_marker": "Pan-CK",
                "autofluorescence_marker": "AF1",
            },
            "region_of_interest": {
                "patch_width_pixels": 64,
                "patch_height_pixels": 64,
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
                "positive_fraction_minimum": 0.005,
                "positive_fraction_maximum": 0.70,
                "fallback_quantile": 0.90,
            },
            "annotation": {"cell_types": cell_types},
            "spatial_analysis": {
                "nearest_neighbor_count": 2,
                "permutation_count": 5,
                "neighborhood_cluster_count": 2,
            },
        }
    )


def build_cell_features(**marker_values: float) -> pl.DataFrame:
    row = {
        "cell_identifier": 1,
        "x_micrometers": 1.0,
        "y_micrometers": 2.0,
        **marker_values,
    }
    return pl.DataFrame([row])


def test_single_marker_match_assigns_configured_cell_type() -> None:
    configuration = build_configuration(
        [{"name": "B_cell", "positive_markers": ["CD20"]}]
    )
    cell_annotations = annotate_cells(
        build_cell_features(CD20=1000.0),
        configuration,
    )
    assert cell_annotations[0, "cell_type"] == "B_cell"


def test_multi_marker_rule_requires_all_positive_markers() -> None:
    configuration = build_configuration(
        [{"name": "Treg", "positive_markers": ["CD4", "FOXP3"]}]
    )
    cell_annotations = annotate_cells(
        pl.DataFrame(
            [
                {
                    "cell_identifier": 1,
                    "x_micrometers": 1.0,
                    "y_micrometers": 2.0,
                    "CD4": 1000.0,
                    "FOXP3": 0.0,
                },
                {
                    "cell_identifier": 2,
                    "x_micrometers": 3.0,
                    "y_micrometers": 4.0,
                    "CD4": 1000.0,
                    "FOXP3": 1000.0,
                },
            ]
        ),
        configuration,
    )
    assert cell_annotations[0, "cell_type"] == "unassigned"


def test_first_matching_rule_wins() -> None:
    configuration = build_configuration(
        [
            {"name": "Treg", "positive_markers": ["CD3e", "CD4", "FOXP3"]},
            {"name": "T_cell", "positive_markers": ["CD3e"]},
        ]
    )
    cell_annotations = annotate_cells(
        build_cell_features(CD3e=1000.0, CD4=1000.0, FOXP3=1000.0),
        configuration,
    )
    assert cell_annotations[0, "cell_type"] == "Treg"


def test_parent_rule_matches_when_subtype_does_not() -> None:
    configuration = build_configuration(
        [
            {"name": "Treg", "positive_markers": ["CD3e", "CD4", "FOXP3"]},
            {"name": "T_cell", "positive_markers": ["CD3e"]},
        ]
    )
    cell_annotations = annotate_cells(
        pl.DataFrame(
            [
                {
                    "cell_identifier": 1,
                    "x_micrometers": 1.0,
                    "y_micrometers": 2.0,
                    "CD3e": 1000.0,
                    "CD4": 0.0,
                    "FOXP3": 0.0,
                },
                {
                    "cell_identifier": 2,
                    "x_micrometers": 3.0,
                    "y_micrometers": 4.0,
                    "CD3e": 1000.0,
                    "CD4": 1000.0,
                    "FOXP3": 1000.0,
                },
            ]
        ),
        configuration,
    )
    assert cell_annotations[0, "cell_type"] == "T_cell"
    assert cell_annotations[1, "cell_type"] == "Treg"


def test_negative_markers_exclude_cell() -> None:
    configuration = build_configuration(
        [
            {
                "name": "helper_T",
                "positive_markers": ["CD3e", "CD4"],
                "negative_markers": ["FOXP3"],
            },
        ]
    )
    cell_annotations = annotate_cells(
        build_cell_features(CD3e=1000.0, CD4=1000.0, FOXP3=1000.0),
        configuration,
    )
    assert cell_annotations[0, "cell_type"] == "unassigned"


def test_negative_marker_low_allows_match() -> None:
    configuration = build_configuration(
        [
            {
                "name": "helper_T",
                "positive_markers": ["CD3e", "CD4"],
                "negative_markers": ["FOXP3"],
            },
        ]
    )
    cell_annotations = annotate_cells(
        pl.DataFrame(
            [
                {
                    "cell_identifier": 1,
                    "x_micrometers": 1.0,
                    "y_micrometers": 2.0,
                    "CD3e": 1000.0,
                    "CD4": 1000.0,
                    "FOXP3": 0.0,
                },
                {
                    "cell_identifier": 2,
                    "x_micrometers": 3.0,
                    "y_micrometers": 4.0,
                    "CD3e": 1000.0,
                    "CD4": 1000.0,
                    "FOXP3": 1000.0,
                },
            ]
        ),
        configuration,
    )
    assert cell_annotations[0, "cell_type"] == "helper_T"
    assert cell_annotations[1, "cell_type"] == "unassigned"


def test_no_rule_match_labels_unassigned() -> None:
    configuration = build_configuration(
        [{"name": "B_cell", "positive_markers": ["CD20"]}]
    )
    cell_annotations = annotate_cells(
        pl.DataFrame(
            [
                {
                    "cell_identifier": 1,
                    "x_micrometers": 1.0,
                    "y_micrometers": 2.0,
                    "CD20": 0.0,
                },
                {
                    "cell_identifier": 2,
                    "x_micrometers": 3.0,
                    "y_micrometers": 4.0,
                    "CD20": 1000.0,
                },
            ]
        ),
        configuration,
    )
    assert cell_annotations[0, "cell_type"] == "unassigned"
    assert cell_annotations[1, "cell_type"] == "B_cell"


def test_high_flags_include_negative_markers() -> None:
    configuration = build_configuration(
        [
            {
                "name": "helper_T",
                "positive_markers": ["CD3e"],
                "negative_markers": ["FOXP3"],
            },
        ]
    )
    cell_annotations = annotate_cells(
        build_cell_features(CD3e=1000.0, FOXP3=0.0),
        configuration,
    )
    assert "FOXP3_high" in cell_annotations.columns


def test_missing_marker_column_fails_fast() -> None:
    configuration = build_configuration(
        [{"name": "B_cell", "positive_markers": ["CD20"]}]
    )
    try:
        annotate_cells(build_cell_features(CD3e=1000.0), configuration)
    except ValueError as value_error:
        assert "CD20" in str(value_error)
    else:
        raise AssertionError(
            "Expected annotation to fail when marker columns are missing."
        )
