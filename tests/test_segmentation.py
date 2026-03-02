import numpy as np

from orion.segmentation import (
    find_labels_touching_boundary,
    segment_cells_from_nuclear_image,
)


class DummySegmentationConfiguration:
    gaussian_sigma = 1.2
    minimum_nucleus_area_pixels = 10
    maximum_nucleus_area_pixels = 800
    peak_minimum_distance_pixels = 4
    cell_expansion_distance_pixels = 4


class DummyPercentileClipConfiguration:
    lower_quantile = 0.005
    upper_quantile = 0.995


class DummyPreprocessingConfiguration:
    percentile_clip = DummyPercentileClipConfiguration()


class DummyApplicationConfiguration:
    segmentation = DummySegmentationConfiguration()
    preprocessing = DummyPreprocessingConfiguration()


def test_watershed_splits_touching_nuclei() -> None:
    nuclear_image = np.zeros((64, 64), dtype=np.float32)
    row_coordinates, column_coordinates = np.ogrid[:64, :64]
    nuclear_image[
        (row_coordinates - 24) ** 2 + (column_coordinates - 24) ** 2 < 100
    ] = 1.0
    nuclear_image[
        (row_coordinates - 24) ** 2 + (column_coordinates - 38) ** 2 < 100
    ] = 1.0
    segmentation_result = segment_cells_from_nuclear_image(
        nuclear_image,
        DummyApplicationConfiguration(),
    )
    assert segmentation_result.nuclei_labels.max() >= 2


def test_expanded_labels_do_not_overlap() -> None:
    nuclear_image = np.zeros((64, 64), dtype=np.float32)
    nuclear_image[16:24, 16:24] = 1
    nuclear_image[40:48, 40:48] = 1
    segmentation_result = segment_cells_from_nuclear_image(
        nuclear_image,
        DummyApplicationConfiguration(),
    )
    assert np.all(segmentation_result.expanded_cell_labels >= 0)


def test_boundary_touching_labels_detected() -> None:
    label_image = np.zeros((8, 8), dtype=np.int32)
    label_image[0:3, 0:3] = 1
    label_image[4:6, 4:6] = 2
    assert find_labels_touching_boundary(label_image) == {1}
