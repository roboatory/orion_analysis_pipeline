import numpy as np

from src.preprocessing import preprocess_region_of_interest_patch


class DummyAutofluorescenceSubtractionConfiguration:
    enabled = True
    sample_pixels = 32
    clip_upper_quantile = 0.95


class DummyPercentileClipConfiguration:
    lower_quantile = 0.005
    upper_quantile = 0.995


class DummyChannelConfiguration:
    nuclear_marker = "Hoechst"
    cytoplasmic_marker = "Pan-CK"
    autofluorescence_marker = "AF1"


class DummyApplicationConfiguration:
    channels = DummyChannelConfiguration()
    preprocessing = type(
        "DummyPreprocessingContainer",
        (),
        {
            "autofluorescence_subtraction": DummyAutofluorescenceSubtractionConfiguration(),
            "percentile_clip": DummyPercentileClipConfiguration(),
        },
    )()


def test_autofluorescence_subtraction_non_negative() -> None:
    raw_image_stack = np.zeros((3, 8, 8), dtype=np.float32)
    raw_image_stack[0] = 50
    raw_image_stack[1] = 5
    raw_image_stack[2] = 25
    preprocessing_result = preprocess_region_of_interest_patch(
        raw_image_stack,
        ["Hoechst", "AF1", "CD45"],
        DummyApplicationConfiguration(),
    )
    assert np.all(preprocessing_result.corrected_image_stack >= 0)


def test_autofluorescence_scale_zero_when_channel_zero() -> None:
    raw_image_stack = np.zeros((3, 8, 8), dtype=np.float32)
    raw_image_stack[0] = 10
    raw_image_stack[2] = 20
    preprocessing_result = preprocess_region_of_interest_patch(
        raw_image_stack,
        ["Hoechst", "AF1", "CD45"],
        DummyApplicationConfiguration(),
    )
    assert preprocessing_result.autofluorescence_scale_by_marker["CD45"] == 0.0
