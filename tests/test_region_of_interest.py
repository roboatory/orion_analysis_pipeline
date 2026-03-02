from orion.data_models import RegionOfInterestBox, RegionOfInterestCandidateScore


def score_candidate(
    candidate_score: RegionOfInterestCandidateScore,
    contrast_score: float,
    focus_score: float,
    autofluorescence_score: float,
    saturation_score: float,
) -> float:
    return (
        0.35 * candidate_score.density_score
        + 0.25 * candidate_score.diversity_score
        + 0.20 * contrast_score
        + 0.10 * focus_score
        - 0.10 * autofluorescence_score
        - 0.10 * saturation_score
    )


def test_density_preferred_when_diversity_equal() -> None:
    higher_density_candidate = RegionOfInterestCandidateScore(
        RegionOfInterestBox(0, 0, 32, 32),
        100,
        0.6,
        0.9,
        90,
    )
    lower_density_candidate = RegionOfInterestCandidateScore(
        RegionOfInterestBox(32, 0, 32, 32),
        100,
        0.6,
        0.5,
        50,
    )
    assert score_candidate(
        higher_density_candidate,
        0.5,
        0.5,
        0.2,
        0.2,
    ) > score_candidate(lower_density_candidate, 0.5, 0.5, 0.2, 0.2)


def test_diversity_preferred_when_density_similar() -> None:
    higher_diversity_candidate = RegionOfInterestCandidateScore(
        RegionOfInterestBox(0, 0, 32, 32),
        100,
        0.8,
        0.6,
        80,
    )
    lower_diversity_candidate = RegionOfInterestCandidateScore(
        RegionOfInterestBox(32, 0, 32, 32),
        100,
        0.2,
        0.6,
        20,
    )
    assert score_candidate(
        higher_diversity_candidate,
        0.5,
        0.5,
        0.2,
        0.2,
    ) > score_candidate(lower_diversity_candidate, 0.5, 0.5, 0.2, 0.2)


def test_manual_override_box_dictionary() -> None:
    region_of_interest = RegionOfInterestBox(1, 2, 3, 4)
    assert region_of_interest.as_dictionary() == {
        "x_pixels": 1,
        "y_pixels": 2,
        "width_pixels": 3,
        "height_pixels": 4,
    }
