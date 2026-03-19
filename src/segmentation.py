from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as scipy_ndimage
from skimage import feature, filters, measure, morphology, segmentation

from src.configuration import ApplicationConfiguration
from src.data_models import SegmentationValidationSummary
from src.io import percentile_normalize_image


@dataclass(frozen=True)
class SegmentationResult:
    nuclei_labels: np.ndarray
    expanded_cell_labels: np.ndarray
    kept_cell_labels: np.ndarray
    chosen_peak_minimum_distance_pixels: int


def segment_cells_from_marker_images(
    nuclear_image: np.ndarray,
    cytoplasmic_image: np.ndarray,
    configuration: ApplicationConfiguration,
    target_cell_count: int | None = None,
) -> SegmentationResult:
    normalized_nuclear_image = percentile_normalize_image(
        nuclear_image,
        configuration.preprocessing.percentile_clip.lower_quantile,
        configuration.preprocessing.percentile_clip.upper_quantile,
    )
    normalized_cytoplasmic_image = percentile_normalize_image(
        cytoplasmic_image,
        configuration.preprocessing.percentile_clip.lower_quantile,
        configuration.preprocessing.percentile_clip.upper_quantile,
    )
    smoothed_nuclear_image = filters.gaussian(
        normalized_nuclear_image,
        sigma=configuration.segmentation.gaussian_sigma,
    )
    otsu_threshold = filters.threshold_otsu(smoothed_nuclear_image)
    binary_nuclear_mask = smoothed_nuclear_image > otsu_threshold
    binary_nuclear_mask = morphology.remove_small_objects(
        binary_nuclear_mask,
        min_size=configuration.segmentation.minimum_nucleus_area_pixels,
    )
    binary_nuclear_mask = morphology.remove_small_holes(
        binary_nuclear_mask,
        area_threshold=64,
    )
    labeled_nuclear_mask = measure.label(binary_nuclear_mask)

    peak_minimum_distance_candidates = [
        configuration.segmentation.peak_minimum_distance_pixels
    ]
    if target_cell_count is not None:
        peak_minimum_distance_candidates = list(
            range(configuration.segmentation.peak_minimum_distance_pixels, 1, -1)
        )
    segmentation_candidates = [
        _segment_binary_mask_with_peak_distance(
            labeled_nuclear_mask=labeled_nuclear_mask,
            normalized_cytoplasmic_image=normalized_cytoplasmic_image,
            configuration=configuration,
            peak_minimum_distance_pixels=peak_minimum_distance_pixels,
        )
        for peak_minimum_distance_pixels in peak_minimum_distance_candidates
    ]
    if target_cell_count is None:
        return segmentation_candidates[0]
    return min(
        segmentation_candidates,
        key=lambda segmentation_result: (
            abs(int(segmentation_result.kept_cell_labels.max()) - target_cell_count),
            abs(
                segmentation_result.chosen_peak_minimum_distance_pixels
                - configuration.segmentation.peak_minimum_distance_pixels
            ),
        ),
    )


def _segment_binary_mask_with_peak_distance(
    labeled_nuclear_mask: np.ndarray,
    normalized_cytoplasmic_image: np.ndarray,
    configuration: ApplicationConfiguration,
    peak_minimum_distance_pixels: int,
) -> SegmentationResult:
    distance_transform = scipy_ndimage.distance_transform_edt(labeled_nuclear_mask > 0)
    local_maxima_coordinates = feature.peak_local_max(
        distance_transform,
        min_distance=peak_minimum_distance_pixels,
        labels=labeled_nuclear_mask > 0,
    )
    watershed_markers = np.zeros_like(labeled_nuclear_mask, dtype=np.int32)
    if len(local_maxima_coordinates) > 0:
        watershed_markers[tuple(local_maxima_coordinates.T)] = np.arange(
            1,
            len(local_maxima_coordinates) + 1,
            dtype=np.int32,
        )
    else:
        watershed_markers = measure.label(labeled_nuclear_mask > 0)
    watershed_labels = segmentation.watershed(
        -distance_transform,
        watershed_markers,
        mask=labeled_nuclear_mask > 0,
    )
    watershed_labels = remove_large_segmentation_labels(
        watershed_labels,
        maximum_area=configuration.segmentation.maximum_nucleus_area_pixels,
    )
    cell_body_mask = build_cell_body_mask(
        labeled_nuclear_mask > 0,
        normalized_cytoplasmic_image,
        configuration.segmentation.cell_expansion_distance_pixels,
    )
    expanded_cell_labels = segmentation.watershed(
        -normalized_cytoplasmic_image,
        watershed_labels,
        mask=cell_body_mask,
    )
    if int(expanded_cell_labels.max()) == 0:
        expanded_cell_labels = segmentation.expand_labels(
            watershed_labels,
            distance=configuration.segmentation.cell_expansion_distance_pixels,
        )
    expanded_cell_labels = remove_small_segmentation_labels(
        expanded_cell_labels,
        minimum_area=80,
    )
    boundary_touching_labels = find_labels_touching_boundary(expanded_cell_labels)
    kept_cell_labels = expanded_cell_labels.copy()
    if boundary_touching_labels:
        boundary_touching_mask = np.isin(
            kept_cell_labels,
            np.fromiter(boundary_touching_labels, dtype=np.int32),
        )
        kept_cell_labels[boundary_touching_mask] = 0
    kept_cell_labels = relabel_sequentially(kept_cell_labels)
    return SegmentationResult(
        nuclei_labels=relabel_sequentially(watershed_labels),
        expanded_cell_labels=expanded_cell_labels,
        kept_cell_labels=kept_cell_labels,
        chosen_peak_minimum_distance_pixels=peak_minimum_distance_pixels,
    )


def build_cell_body_mask(
    nuclear_mask: np.ndarray,
    normalized_cytoplasmic_image: np.ndarray,
    expansion_distance_pixels: int,
) -> np.ndarray:
    if np.allclose(normalized_cytoplasmic_image, normalized_cytoplasmic_image.flat[0]):
        return morphology.binary_dilation(
            nuclear_mask,
            morphology.disk(expansion_distance_pixels),
        )
    cytoplasmic_threshold = filters.threshold_otsu(normalized_cytoplasmic_image)
    cytoplasmic_mask = normalized_cytoplasmic_image > cytoplasmic_threshold
    if not np.any(cytoplasmic_mask):
        cytoplasmic_mask = nuclear_mask.copy()
    expanded_nuclear_mask = morphology.binary_dilation(
        nuclear_mask,
        morphology.disk(expansion_distance_pixels),
    )
    return cytoplasmic_mask | expanded_nuclear_mask


def remove_large_segmentation_labels(
    labels: np.ndarray, maximum_area: int
) -> np.ndarray:
    output_labels = labels.copy()
    for region_properties in measure.regionprops(labels):
        if region_properties.area > maximum_area:
            output_labels[labels == region_properties.label] = 0
    return relabel_sequentially(output_labels)


def remove_small_segmentation_labels(
    labels: np.ndarray, minimum_area: int
) -> np.ndarray:
    output_labels = labels.copy()
    for region_properties in measure.regionprops(labels):
        if region_properties.area < minimum_area:
            output_labels[labels == region_properties.label] = 0
    return relabel_sequentially(output_labels)


def find_labels_touching_boundary(labels: np.ndarray) -> set[int]:
    boundary_labels = set(np.unique(labels[0, :])) | set(np.unique(labels[-1, :]))
    boundary_labels |= set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    boundary_labels.discard(0)
    return boundary_labels


def relabel_sequentially(labels: np.ndarray) -> np.ndarray:
    relabeled_labels, _, _ = segmentation.relabel_sequential(labels)
    return relabeled_labels.astype(np.int32)


def summarize_segmentation_validation(
    existing_labels: np.ndarray | None,
    new_labels: np.ndarray,
) -> SegmentationValidationSummary | None:
    if existing_labels is None:
        return None
    relabeled_existing_labels = relabel_sequentially(existing_labels.astype(np.int32))
    relabeled_new_labels = relabel_sequentially(new_labels.astype(np.int32))
    existing_region_properties = measure.regionprops(relabeled_existing_labels)
    new_region_properties = measure.regionprops(relabeled_new_labels)
    existing_areas = np.array(
        [region_properties.area for region_properties in existing_region_properties],
        dtype=float,
    )
    new_areas = np.array(
        [region_properties.area for region_properties in new_region_properties],
        dtype=float,
    )
    centroid_density_overlap = compute_centroid_density_overlap(
        existing_region_properties,
        new_region_properties,
        relabeled_existing_labels.shape,
    )
    return SegmentationValidationSummary(
        existing_cell_count=int(relabeled_existing_labels.max()),
        new_cell_count=int(relabeled_new_labels.max()),
        existing_median_area_square_pixels=float(np.median(existing_areas))
        if existing_areas.size
        else 0.0,
        new_median_area_square_pixels=float(np.median(new_areas))
        if new_areas.size
        else 0.0,
        centroid_density_overlap=centroid_density_overlap,
    )


def compute_centroid_density_overlap(
    existing_region_properties: list[measure._regionprops.RegionProperties],
    new_region_properties: list[measure._regionprops.RegionProperties],
    image_shape: tuple[int, int],
) -> float:
    y_bin_count = max(image_shape[0] // 128, 2)
    x_bin_count = max(image_shape[1] // 128, 2)
    existing_points = np.array(
        [
            region_properties.centroid
            for region_properties in existing_region_properties
        ],
        dtype=float,
    )
    new_points = np.array(
        [region_properties.centroid for region_properties in new_region_properties],
        dtype=float,
    )
    if len(existing_points) == 0 or len(new_points) == 0:
        return 0.0
    existing_histogram, _, _ = np.histogram2d(
        existing_points[:, 0],
        existing_points[:, 1],
        bins=(y_bin_count, x_bin_count),
    )
    new_histogram, _, _ = np.histogram2d(
        new_points[:, 0],
        new_points[:, 1],
        bins=(y_bin_count, x_bin_count),
    )
    if np.std(existing_histogram) == 0 or np.std(new_histogram) == 0:
        return 0.0
    return float(np.corrcoef(existing_histogram.ravel(), new_histogram.ravel())[0, 1])
