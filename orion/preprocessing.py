from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orion.configuration import ApplicationConfiguration


@dataclass(frozen=True)
class PreprocessingResult:
    corrected_image_stack: np.ndarray
    autofluorescence_scale_by_marker: dict[str, float]
    corrected_image_by_marker: dict[str, np.ndarray]


def preprocess_region_of_interest_patch(
    raw_image_stack: np.ndarray,
    marker_names: list[str],
    configuration: ApplicationConfiguration,
    random_seed: int = 7,
) -> PreprocessingResult:
    technical_marker_names = set(configuration.channels.technical_markers)
    marker_name_to_index = {
        marker_name: index for index, marker_name in enumerate(marker_names)
    }
    autofluorescence_index = marker_name_to_index[
        configuration.channels.autofluorescence_marker
    ]
    autofluorescence_channel = raw_image_stack[autofluorescence_index].astype(
        np.float32
    )
    corrected_image_stack = raw_image_stack.astype(np.float32).copy()
    autofluorescence_scale_by_marker: dict[str, float] = {}

    if not configuration.preprocessing.autofluorescence_subtraction.enabled:
        corrected_image_by_marker = {
            marker_name: corrected_image_stack[index]
            for index, marker_name in enumerate(marker_names)
        }
        return PreprocessingResult(
            corrected_image_stack=corrected_image_stack,
            autofluorescence_scale_by_marker=autofluorescence_scale_by_marker,
            corrected_image_by_marker=corrected_image_by_marker,
        )

    random_number_generator = np.random.default_rng(random_seed)
    clipped_autofluorescence_values = clip_upper_intensity(
        autofluorescence_channel,
        configuration.preprocessing.autofluorescence_subtraction.clip_upper_quantile,
    ).ravel()
    sample_size = min(
        configuration.preprocessing.autofluorescence_subtraction.sample_pixels,
        clipped_autofluorescence_values.size,
    )
    sampled_pixel_indices = random_number_generator.choice(
        clipped_autofluorescence_values.size,
        size=sample_size,
        replace=False,
    )
    sampled_autofluorescence_values = clipped_autofluorescence_values[
        sampled_pixel_indices
    ]
    autofluorescence_denominator = float(
        np.dot(sampled_autofluorescence_values, sampled_autofluorescence_values)
    )

    for marker_name, marker_index in marker_name_to_index.items():
        current_channel = corrected_image_stack[marker_index]
        if (
            marker_name in technical_marker_names
            or marker_name == configuration.channels.autofluorescence_marker
        ):
            autofluorescence_scale_by_marker[marker_name] = 0.0
            continue
        clipped_channel_values = clip_upper_intensity(
            current_channel,
            configuration.preprocessing.autofluorescence_subtraction.clip_upper_quantile,
        ).ravel()
        sampled_channel_values = clipped_channel_values[sampled_pixel_indices]
        if autofluorescence_denominator == 0.0:
            autofluorescence_scale = 0.0
        else:
            autofluorescence_scale = float(
                np.dot(sampled_channel_values, sampled_autofluorescence_values)
                / autofluorescence_denominator
            )
        corrected_image_stack[marker_index] = np.maximum(
            current_channel - autofluorescence_scale * autofluorescence_channel,
            0.0,
        )
        autofluorescence_scale_by_marker[marker_name] = autofluorescence_scale

    corrected_image_by_marker = {
        marker_name: corrected_image_stack[index]
        for index, marker_name in enumerate(marker_names)
    }
    return PreprocessingResult(
        corrected_image_stack=corrected_image_stack,
        autofluorescence_scale_by_marker=autofluorescence_scale_by_marker,
        corrected_image_by_marker=corrected_image_by_marker,
    )


def clip_upper_intensity(image: np.ndarray, upper_quantile: float) -> np.ndarray:
    upper_intensity = float(np.quantile(image, upper_quantile))
    return np.clip(image, None, upper_intensity)
