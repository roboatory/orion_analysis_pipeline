from unittest.mock import MagicMock, patch

import numpy as np

from src.segmentation import segment_cells_from_marker_images


class DummySegmentationConfiguration:
    cell_diameter_pixels = 30.0
    use_gpu = False


class DummyPreprocessingConfiguration:
    pass


class DummyApplicationConfiguration:
    segmentation = DummySegmentationConfiguration()
    preprocessing = DummyPreprocessingConfiguration()


@patch("src.segmentation.CellposeModel")
def test_cellpose_returns_sequential_int32_labels(
    mock_cellpose_class: MagicMock,
) -> None:
    mock_model = MagicMock()
    mock_cellpose_class.return_value = mock_model
    mock_labels = np.array(
        [[0, 0, 5, 5], [0, 0, 5, 5], [3, 3, 0, 0], [3, 3, 0, 0]],
        dtype=np.int32,
    )
    mock_model.eval.return_value = (mock_labels, None, None)

    result = segment_cells_from_marker_images(
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
        DummyApplicationConfiguration(),
    )

    assert result.cell_labels.dtype == np.int32
    unique_labels = set(np.unique(result.cell_labels)) - {0}
    assert unique_labels == {1, 2}


@patch("src.segmentation.CellposeModel")
def test_cellpose_called_without_channels(mock_cellpose_class: MagicMock) -> None:
    mock_model = MagicMock()
    mock_cellpose_class.return_value = mock_model
    mock_model.eval.return_value = (np.zeros((4, 4), dtype=np.int32), None, None)

    segment_cells_from_marker_images(
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
        DummyApplicationConfiguration(),
    )

    mock_model.eval.assert_called_once()
    call_kwargs = mock_model.eval.call_args
    assert "channels" not in call_kwargs.kwargs


@patch("src.segmentation.CellposeModel")
def test_cellpose_receives_diameter_and_gpu(mock_cellpose_class: MagicMock) -> None:
    mock_model = MagicMock()
    mock_cellpose_class.return_value = mock_model
    mock_model.eval.return_value = (np.zeros((4, 4), dtype=np.int32), None, None)

    segment_cells_from_marker_images(
        np.zeros((4, 4), dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
        DummyApplicationConfiguration(),
    )

    mock_cellpose_class.assert_called_once_with(
        pretrained_model="cpsam",
        gpu=False,
    )
    call_kwargs = mock_model.eval.call_args
    assert call_kwargs.kwargs["diameter"] == 30.0
