from dataclasses import asdict

import numpy as np
import pytest

from src.data_models import RegionOfInterestBox, SlideMetadata
from src.region_of_interest import (
    boxes_overlap,
    choose_region_of_interest,
    score_region_of_interest_patch,
)


class DummyRegionOfInterestConfiguration:
    patch_width_pixels = 32
    patch_height_pixels = 24
    candidate_patch_count = 3
    analysis_patch_count = 1
    minimum_tissue_fraction = 0.2
    minimum_informative_channel_fraction = 0.5
    minimum_channel_signal_spread = 0.05


class DummyChannelConfiguration:
    nuclear_marker = "Hoechst"
    cytoplasmic_marker = "Pan-CK"
    autofluorescence_marker = "AF1"


class DummyConfiguration:
    sample_identifier = "TEST_SAMPLE"
    region_of_interest = DummyRegionOfInterestConfiguration()
    channels = DummyChannelConfiguration()


def build_slide_metadata(
    width_pixels: int = 200, height_pixels: int = 100
) -> SlideMetadata:
    return SlideMetadata(
        readouts_path="readouts.tiff",
        histology_path=None,
        width_pixels=width_pixels,
        height_pixels=height_pixels,
        pixel_size_x_micrometers=1.0,
        pixel_size_y_micrometers=1.0,
        marker_names=["Hoechst", "AF1", "Pan-CK", "CD45"],
    )


def test_random_region_of_interest_is_deterministic_for_sample_identifier(
    monkeypatch,
) -> None:
    slide_metadata = build_slide_metadata()
    monkeypatch.setattr(
        "src.region_of_interest.read_readouts_region_of_interest",
        lambda *_arguments, **_keyword_arguments: np.ones(
            (4, 24, 32), dtype=np.float32
        ),
    )
    first_regions, first_candidates = choose_region_of_interest(
        DummyConfiguration(),
        slide_metadata,
    )
    second_regions, second_candidates = choose_region_of_interest(
        DummyConfiguration(),
        slide_metadata,
    )
    assert first_regions == second_regions
    assert first_candidates.to_dicts() == second_candidates.to_dicts()


def test_random_region_of_interest_respects_patch_bounds(monkeypatch) -> None:
    slide_metadata = build_slide_metadata(width_pixels=80, height_pixels=60)
    monkeypatch.setattr(
        "src.region_of_interest.read_readouts_region_of_interest",
        lambda *_arguments, **_keyword_arguments: np.ones(
            (4, 24, 32), dtype=np.float32
        ),
    )
    regions_of_interest, candidate_data_frame = choose_region_of_interest(
        DummyConfiguration(),
        slide_metadata,
    )
    assert len(regions_of_interest) == 1
    region_of_interest = regions_of_interest[0]
    assert 0 <= region_of_interest.x_pixels <= 48
    assert 0 <= region_of_interest.y_pixels <= 36
    assert (
        candidate_data_frame["selection_mode"].to_list()[0] == "seeded_random_best_of_n"
    )


def test_high_quality_patch_outranks_sparse_patch(monkeypatch) -> None:
    candidate_regions_of_interest = [
        RegionOfInterestBox(0, 0, 32, 24),
        RegionOfInterestBox(40, 0, 32, 24),
        RegionOfInterestBox(80, 0, 32, 24),
    ]

    def fake_generate_candidates(*_arguments, **_keyword_arguments):
        return candidate_regions_of_interest

    blank_patch = np.zeros((4, 24, 32), dtype=np.float32)
    low_quality_patch = np.full((4, 24, 32), 0.2, dtype=np.float32)
    high_quality_patch = np.zeros((4, 24, 32), dtype=np.float32)
    high_quality_patch[0, 5:15, 6:16] = 10.0
    high_quality_patch[2, 3:12, 18:28] = 8.0
    high_quality_patch[3, 12:20, 10:22] = 7.0

    patch_by_coordinate = {
        (0, 0): blank_patch,
        (40, 0): low_quality_patch,
        (80, 0): high_quality_patch,
    }

    def fake_read_patch(_path, region_of_interest):
        return patch_by_coordinate[
            (region_of_interest.x_pixels, region_of_interest.y_pixels)
        ]

    monkeypatch.setattr(
        "src.region_of_interest.generate_candidate_regions_of_interest",
        fake_generate_candidates,
    )
    monkeypatch.setattr(
        "src.region_of_interest.read_readouts_region_of_interest",
        fake_read_patch,
    )
    regions_of_interest, candidate_data_frame = choose_region_of_interest(
        DummyConfiguration(),
        build_slide_metadata(),
    )
    assert regions_of_interest == [candidate_regions_of_interest[2]]
    selected_row = candidate_data_frame.filter(candidate_data_frame["selected"]).row(
        0, named=True
    )
    assert selected_row["passes_quality_thresholds"] is True
    assert selected_row["quality_score"] == max(candidate_data_frame["quality_score"])


def test_best_candidate_is_returned_when_no_patch_passes_thresholds(
    monkeypatch,
) -> None:
    configuration = DummyConfiguration()
    configuration.region_of_interest.minimum_tissue_fraction = 0.95
    configuration.region_of_interest.minimum_informative_channel_fraction = 0.95

    candidate_regions_of_interest = [
        RegionOfInterestBox(0, 0, 32, 24),
        RegionOfInterestBox(40, 0, 32, 24),
    ]

    weak_patch = np.zeros((4, 24, 32), dtype=np.float32)
    weak_patch[0, 6:18, 8:20] = 2.0
    better_patch = np.zeros((4, 24, 32), dtype=np.float32)
    better_patch[0, 4:20, 5:27] = 4.0
    better_patch[2, 3:21, 4:28] = 3.5

    patch_by_coordinate = {
        (0, 0): weak_patch,
        (40, 0): better_patch,
    }

    monkeypatch.setattr(
        "src.region_of_interest.generate_candidate_regions_of_interest",
        lambda *_arguments, **_keyword_arguments: candidate_regions_of_interest,
    )
    monkeypatch.setattr(
        "src.region_of_interest.read_readouts_region_of_interest",
        lambda _path, region_of_interest: patch_by_coordinate[
            (region_of_interest.x_pixels, region_of_interest.y_pixels)
        ],
    )
    regions_of_interest, candidate_data_frame = choose_region_of_interest(
        configuration,
        build_slide_metadata(),
    )
    assert regions_of_interest == [candidate_regions_of_interest[1]]
    assert candidate_data_frame["passes_quality_thresholds"].to_list() == [False, False]
    assert candidate_data_frame["selected"].to_list() == [True, False]


def test_quality_score_prefers_tissue_and_informative_signal() -> None:
    configuration = DummyConfiguration()
    mostly_blank_patch = np.zeros((4, 24, 32), dtype=np.float32)
    informative_patch = np.zeros((4, 24, 32), dtype=np.float32)
    informative_patch[0, 5:15, 6:16] = 6.0
    informative_patch[2, 3:12, 18:28] = 5.0
    informative_patch[3, 12:20, 10:22] = 4.0

    blank_score = score_region_of_interest_patch(
        mostly_blank_patch,
        build_slide_metadata().marker_names,
        configuration,
    )
    informative_score = score_region_of_interest_patch(
        informative_patch,
        build_slide_metadata().marker_names,
        configuration,
    )
    assert informative_score["tissue_fraction"] > blank_score["tissue_fraction"]
    assert (
        informative_score["informative_channel_fraction"]
        > blank_score["informative_channel_fraction"]
    )
    assert informative_score["quality_score"] > blank_score["quality_score"]


def test_region_of_interest_box_as_dict() -> None:
    region_of_interest = RegionOfInterestBox(1, 2, 3, 4)
    assert asdict(region_of_interest) == {
        "x_pixels": 1,
        "y_pixels": 2,
        "width_pixels": 3,
        "height_pixels": 4,
    }


def test_boxes_overlap_detects_shared_pixel() -> None:
    box_a = RegionOfInterestBox(0, 0, 10, 10)
    box_b = RegionOfInterestBox(5, 5, 10, 10)
    assert boxes_overlap(box_a, box_b)


def test_boxes_overlap_rejects_adjacent_boxes() -> None:
    box_a = RegionOfInterestBox(0, 0, 10, 10)
    box_b = RegionOfInterestBox(10, 0, 10, 10)
    assert not boxes_overlap(box_a, box_b)


def test_multi_patch_selects_top_n_non_overlapping_patches(monkeypatch) -> None:
    configuration = DummyConfiguration()
    configuration.region_of_interest.analysis_patch_count = 2
    configuration.region_of_interest.candidate_patch_count = 4

    candidate_regions_of_interest = [
        RegionOfInterestBox(0, 0, 32, 24),
        RegionOfInterestBox(5, 0, 32, 24),
        RegionOfInterestBox(40, 0, 32, 24),
        RegionOfInterestBox(80, 0, 32, 24),
    ]
    informative_patch = np.zeros((4, 24, 32), dtype=np.float32)
    informative_patch[0, 5:15, 6:16] = 10.0
    informative_patch[2, 3:12, 18:28] = 8.0
    informative_patch[3, 12:20, 10:22] = 7.0

    monkeypatch.setattr(
        "src.region_of_interest.generate_candidate_regions_of_interest",
        lambda *_arguments, **_keyword_arguments: candidate_regions_of_interest,
    )
    monkeypatch.setattr(
        "src.region_of_interest.read_readouts_region_of_interest",
        lambda *_arguments, **_keyword_arguments: informative_patch,
    )
    regions_of_interest, candidate_data_frame = choose_region_of_interest(
        configuration,
        build_slide_metadata(width_pixels=200, height_pixels=100),
    )
    assert len(regions_of_interest) == 2
    for first, second in zip(regions_of_interest, regions_of_interest[1:]):
        assert not boxes_overlap(first, second)
    assert sum(candidate_data_frame["selected"].to_list()) == 2


def test_multi_patch_raises_when_non_overlap_cannot_be_satisfied(monkeypatch) -> None:
    configuration = DummyConfiguration()
    configuration.region_of_interest.analysis_patch_count = 3
    configuration.region_of_interest.candidate_patch_count = 3

    candidate_regions_of_interest = [
        RegionOfInterestBox(0, 0, 32, 24),
        RegionOfInterestBox(5, 5, 32, 24),
        RegionOfInterestBox(10, 10, 32, 24),
    ]
    informative_patch = np.zeros((4, 24, 32), dtype=np.float32)
    informative_patch[0, 5:15, 6:16] = 10.0
    informative_patch[2, 3:12, 18:28] = 8.0
    informative_patch[3, 12:20, 10:22] = 7.0

    monkeypatch.setattr(
        "src.region_of_interest.generate_candidate_regions_of_interest",
        lambda *_arguments, **_keyword_arguments: candidate_regions_of_interest,
    )
    monkeypatch.setattr(
        "src.region_of_interest.read_readouts_region_of_interest",
        lambda *_arguments, **_keyword_arguments: informative_patch,
    )
    with pytest.raises(ValueError, match="non-overlapping"):
        choose_region_of_interest(
            configuration,
            build_slide_metadata(),
        )
