from pathlib import Path

import pytest
import yaml

from src.configuration import load_configuration


def base_configuration_dictionary(temporary_path: Path) -> dict[str, object]:
    readouts_path = temporary_path / "readouts.tiff"
    markers_path = temporary_path / "markers.csv"
    readouts_path.write_bytes(b"fake")
    markers_path.write_text("Hoechst\nAF1\nPan-CK\nCD45\n", encoding="utf-8")
    return {
        "sample_identifier": "TEST",
        "input_paths": {
            "readouts": str(readouts_path),
            "markers": str(markers_path),
            "histology": None,
        },
        "output_directory": str(temporary_path / "outputs"),
        "channels": {
            "nuclear_marker": "Hoechst",
            "cytoplasmic_marker": "Pan-CK",
            "autofluorescence_marker": "AF1",
        },
        "region_of_interest": {
            "patch_width_pixels": 64,
            "patch_height_pixels": 64,
            "candidate_patch_count": 4,
            "minimum_tissue_fraction": 0.25,
            "minimum_informative_channel_fraction": 0.5,
            "minimum_channel_signal_spread": 0.05,
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
            "minimum_nucleus_area_pixels": 60,
            "maximum_nucleus_area_pixels": 1500,
            "peak_minimum_distance_pixels": 6,
            "cell_expansion_distance_pixels": 6,
        },
        "normalization": {
            "arcsinh_cofactor": 150.0,
            "threshold_method": "otsu_with_fallback",
            "positive_fraction_minimum": 0.005,
            "positive_fraction_maximum": 0.70,
            "fallback_quantile": 0.90,
        },
        "annotation": {
            "cell_types": [
                {"name": "Immune", "positive_markers": ["CD45"]},
            ]
        },
        "spatial_analysis": {
            "nearest_neighbor_count": 10,
            "minimum_cells_per_type_for_pairwise_analysis": 25,
            "minimum_cells_per_type_for_clustering": 50,
            "permutation_count": 10,
            "neighborhood_cluster_count": 3,
        },
    }


def write_configuration_file(
    temporary_path: Path,
    configuration_dictionary: dict[str, object],
) -> Path:
    configuration_path = temporary_path / "configuration.yaml"
    configuration_path.write_text(
        yaml.safe_dump(configuration_dictionary),
        encoding="utf-8",
    )
    return configuration_path


def test_missing_required_path_rejected(tmp_path: Path) -> None:
    configuration_dictionary = base_configuration_dictionary(tmp_path)
    configuration_dictionary["input_paths"]["readouts"] = str(tmp_path / "missing.tiff")
    configuration_path = write_configuration_file(
        tmp_path,
        configuration_dictionary,
    )
    with pytest.raises(ValueError):
        load_configuration(configuration_path)


def test_unknown_marker_rejected(tmp_path: Path) -> None:
    configuration_dictionary = base_configuration_dictionary(tmp_path)
    configuration_dictionary["channels"]["nuclear_marker"] = "DAPI"
    configuration_path = write_configuration_file(
        tmp_path,
        configuration_dictionary,
    )
    with pytest.raises(ValueError):
        load_configuration(configuration_path)


def test_annotation_configuration_requires_non_empty_cell_types(tmp_path: Path) -> None:
    configuration_dictionary = base_configuration_dictionary(tmp_path)
    configuration_dictionary["annotation"]["cell_types"] = []
    configuration_path = write_configuration_file(tmp_path, configuration_dictionary)
    with pytest.raises(ValueError):
        load_configuration(configuration_path)


def test_duplicate_annotation_cell_type_names_rejected(tmp_path: Path) -> None:
    configuration_dictionary = base_configuration_dictionary(tmp_path)
    configuration_dictionary["annotation"]["cell_types"] = [
        {"name": "Immune", "positive_markers": ["CD45"]},
        {"name": "Immune", "positive_markers": ["CD45"]},
    ]
    configuration_path = write_configuration_file(tmp_path, configuration_dictionary)
    with pytest.raises(ValueError):
        load_configuration(configuration_path)


def test_unknown_annotation_marker_rejected(tmp_path: Path) -> None:
    configuration_dictionary = base_configuration_dictionary(tmp_path)
    configuration_dictionary["annotation"]["cell_types"] = [
        {"name": "B_cell", "positive_markers": ["CD20"]},
    ]
    configuration_path = write_configuration_file(tmp_path, configuration_dictionary)
    with pytest.raises(ValueError):
        load_configuration(configuration_path)


def test_technical_annotation_marker_rejected(tmp_path: Path) -> None:
    configuration_dictionary = base_configuration_dictionary(tmp_path)
    configuration_dictionary["annotation"]["cell_types"] = [
        {"name": "Background", "positive_markers": ["AF1"]},
    ]
    configuration_path = write_configuration_file(tmp_path, configuration_dictionary)
    with pytest.raises(ValueError):
        load_configuration(configuration_path)


def test_cytoplasmic_marker_must_differ_from_nuclear_marker(tmp_path: Path) -> None:
    configuration_dictionary = base_configuration_dictionary(tmp_path)
    configuration_dictionary["channels"]["cytoplasmic_marker"] = "Hoechst"
    configuration_path = write_configuration_file(tmp_path, configuration_dictionary)
    with pytest.raises(ValueError):
        load_configuration(configuration_path)
