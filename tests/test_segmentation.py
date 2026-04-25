from unittest.mock import MagicMock, patch

import numpy as np

from src.segmentation import (
    build_segmentation_model,
    segment_cells_from_marker_images,
)


class DummySegmentationConfiguration:
    cell_diameter_pixels = 30.0
    use_gpu = False


class DummyPreprocessingConfiguration:
    pass


class DummyApplicationConfiguration:
    segmentation = DummySegmentationConfiguration()
    preprocessing = DummyPreprocessingConfiguration()


def build_mock_model_with_labels(labels: np.ndarray) -> MagicMock:
    mock_model = MagicMock()
    mock_model.eval.return_value = (labels, None, None)
    return mock_model


def test_cellpose_returns_sequential_int32_labels() -> None:
    mock_labels = np.array(
        [[0, 0, 5, 5], [0, 0, 5, 5], [3, 3, 0, 0], [3, 3, 0, 0]],
        dtype=np.int32,
    )
    mock_model = build_mock_model_with_labels(mock_labels)

    result = segment_cells_from_marker_images(
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
        DummyApplicationConfiguration(),
        mock_model,
    )

    assert result.cell_labels.dtype == np.int32
    unique_labels = set(np.unique(result.cell_labels)) - {0}
    assert unique_labels == {1, 2}


def test_cellpose_called_without_channels() -> None:
    mock_model = build_mock_model_with_labels(np.zeros((4, 4), dtype=np.int32))

    segment_cells_from_marker_images(
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
        DummyApplicationConfiguration(),
        mock_model,
    )

    mock_model.eval.assert_called_once()
    call_kwargs = mock_model.eval.call_args
    assert "channels" not in call_kwargs.kwargs


def test_cellpose_receives_diameter() -> None:
    mock_model = build_mock_model_with_labels(np.zeros((4, 4), dtype=np.int32))

    segment_cells_from_marker_images(
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
        DummyApplicationConfiguration(),
        mock_model,
    )

    call_kwargs = mock_model.eval.call_args
    assert call_kwargs.kwargs["diameter"] == 30.0


@patch("src.segmentation.CellposeModel")
def test_build_segmentation_model_passes_configuration(
    mock_cellpose_class: MagicMock,
) -> None:
    build_segmentation_model(DummyApplicationConfiguration())
    mock_cellpose_class.assert_called_once_with(
        pretrained_model="cpsam",
        gpu=False,
    )


@patch("src.segmentation.CellposeModel")
def test_build_segmentation_model_called_once_for_multiple_patches(
    mock_cellpose_class: MagicMock,
) -> None:
    mock_model = build_mock_model_with_labels(np.zeros((4, 4), dtype=np.int32))
    mock_cellpose_class.return_value = mock_model

    configuration = DummyApplicationConfiguration()
    model = build_segmentation_model(configuration)

    for _ in range(3):
        segment_cells_from_marker_images(
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
            configuration,
            model,
        )

    mock_cellpose_class.assert_called_once()
    assert mock_model.eval.call_count == 3
