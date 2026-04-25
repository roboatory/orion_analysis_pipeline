from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from src import pipeline as pipeline_module
from src.data_models import PatchEntry, RegionOfInterestBox, SlideMetadata
from src.io import parse_patch_entries, read_patches_manifest, write_patches_manifest


def build_dummy_configuration(
    sample_output_root: Path,
    sample_identifier: str = "TEST_SAMPLE",
) -> SimpleNamespace:
    input_paths = SimpleNamespace(
        readouts=sample_output_root / "readouts.tiff",
        histology=None,
        markers=sample_output_root / "markers.csv",
    )
    configuration = SimpleNamespace(
        sample_identifier=sample_identifier,
        input_paths=input_paths,
        sample_output_directory=sample_output_root / sample_identifier,
        channels=SimpleNamespace(
            nuclear_marker="Hoechst",
            cytoplasmic_marker="Pan-CK",
            autofluorescence_marker="AF1",
        ),
        region_of_interest=SimpleNamespace(
            analysis_patch_count=2,
            candidate_patch_count=4,
            patch_width_pixels=32,
            patch_height_pixels=24,
        ),
    )
    return configuration


def build_slide_metadata() -> SlideMetadata:
    return SlideMetadata(
        readouts_path=Path("readouts.tiff"),
        histology_path=None,
        width_pixels=200,
        height_pixels=100,
        pixel_size_x_micrometers=1.0,
        pixel_size_y_micrometers=1.0,
        marker_names=["Hoechst", "AF1", "Pan-CK", "CD45"],
    )


def test_derive_patch_seed_is_deterministic_and_distinct() -> None:
    first_seed = pipeline_module.derive_patch_seed("SAMPLE_A", 0)
    same_seed = pipeline_module.derive_patch_seed("SAMPLE_A", 0)
    other_patch_seed = pipeline_module.derive_patch_seed("SAMPLE_A", 1)
    other_sample_seed = pipeline_module.derive_patch_seed("SAMPLE_B", 0)

    assert first_seed == same_seed
    assert first_seed != other_patch_seed
    assert first_seed != other_sample_seed


def test_patch_index_is_parsed_from_identifier() -> None:
    assert pipeline_module._patch_index_from_id("patch_000") == 0
    assert pipeline_module._patch_index_from_id("patch_007") == 7
    assert pipeline_module._patch_index_from_id("patch_042") == 42


def test_patches_manifest_roundtrip(tmp_path: Path) -> None:
    slide_metadata = build_slide_metadata()
    patch_entries = [
        PatchEntry(
            patch_id="patch_000",
            region_of_interest=RegionOfInterestBox(0, 0, 32, 24),
        ),
        PatchEntry(
            patch_id="patch_001",
            region_of_interest=RegionOfInterestBox(40, 0, 32, 24),
        ),
    ]
    manifest_path = tmp_path / "patches_manifest.yaml"

    write_patches_manifest(
        manifest_path,
        "TEST_SAMPLE",
        slide_metadata,
        patch_entries,
    )

    payload = read_patches_manifest(manifest_path)
    assert payload["sample_identifier"] == "TEST_SAMPLE"
    assert payload["marker_names"] == slide_metadata.marker_names
    assert parse_patch_entries(payload) == patch_entries


def test_run_select_roi_writes_manifest_and_per_patch_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configuration = build_dummy_configuration(tmp_path)
    slide_metadata = build_slide_metadata()
    selected_regions_of_interest = [
        RegionOfInterestBox(0, 0, 32, 24),
        RegionOfInterestBox(40, 0, 32, 24),
    ]
    candidate_data_frame = pl.DataFrame(
        {
            "x_pixels": [0, 40],
            "y_pixels": [0, 0],
            "width_pixels": [32, 32],
            "height_pixels": [24, 24],
            "quality_score": [0.9, 0.8],
            "selected": [True, True],
        }
    )

    monkeypatch.setattr(
        pipeline_module,
        "load_slide_metadata",
        lambda _configuration: slide_metadata,
    )
    monkeypatch.setattr(
        pipeline_module,
        "choose_region_of_interest",
        lambda _configuration, _slide_metadata: (
            selected_regions_of_interest,
            candidate_data_frame,
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "read_readouts_region_of_interest",
        lambda _path, _region: np.ones((4, 24, 32), dtype=np.float32),
    )

    logger = logging.getLogger("test_run_select_roi")

    pipeline_module.run_select_roi(configuration, logger)

    sample_directory = configuration.sample_output_directory
    assert (sample_directory / "patches_manifest.yaml").exists()
    assert (sample_directory / "candidate_patches.csv").exists()
    assert (sample_directory / "patch_000" / "roi_metadata.yaml").exists()
    assert (sample_directory / "patch_000" / "raw_patch.tif").exists()
    assert (sample_directory / "patch_001" / "roi_metadata.yaml").exists()
    assert (sample_directory / "patch_001" / "raw_patch.tif").exists()

    manifest_payload = read_patches_manifest(sample_directory / "patches_manifest.yaml")
    parsed_entries = parse_patch_entries(manifest_payload)
    assert [entry.patch_id for entry in parsed_entries] == ["patch_000", "patch_001"]
    assert parsed_entries[0].region_of_interest == selected_regions_of_interest[0]
    assert parsed_entries[1].region_of_interest == selected_regions_of_interest[1]


def test_run_segment_builds_cellpose_model_once_for_all_patches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configuration = build_dummy_configuration(tmp_path)
    slide_metadata = build_slide_metadata()
    sample_directory = configuration.sample_output_directory
    sample_directory.mkdir(parents=True, exist_ok=True)

    patch_entries = [
        PatchEntry(
            patch_id="patch_000",
            region_of_interest=RegionOfInterestBox(0, 0, 32, 24),
        ),
        PatchEntry(
            patch_id="patch_001",
            region_of_interest=RegionOfInterestBox(40, 0, 32, 24),
        ),
    ]
    write_patches_manifest(
        sample_directory / "patches_manifest.yaml",
        configuration.sample_identifier,
        slide_metadata,
        patch_entries,
    )
    corrected_patch = np.zeros((4, 24, 32), dtype=np.float32)

    monkeypatch.setattr(
        pipeline_module.tifffile,
        "imread",
        lambda _path: corrected_patch,
    )

    build_model_mock = MagicMock(return_value=MagicMock())
    segment_mock = MagicMock(
        return_value=SimpleNamespace(
            cell_labels=np.zeros((24, 32), dtype=np.int32),
        )
    )
    overlay_mock = MagicMock()
    write_label_mock = MagicMock()

    monkeypatch.setattr(pipeline_module, "build_segmentation_model", build_model_mock)
    monkeypatch.setattr(
        pipeline_module, "segment_cells_from_marker_images", segment_mock
    )
    monkeypatch.setattr(
        pipeline_module, "save_segmentation_overlay_image", overlay_mock
    )
    monkeypatch.setattr(pipeline_module, "write_label_array", write_label_mock)

    logger = logging.getLogger("test_run_segment")
    pipeline_module.run_segment(configuration, logger)

    build_model_mock.assert_called_once_with(configuration)
    assert segment_mock.call_count == 2
    assert write_label_mock.call_count == 2
