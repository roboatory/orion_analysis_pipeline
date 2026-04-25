from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cellpose.models import CellposeModel
from skimage import segmentation

from src.configuration import ApplicationConfiguration


@dataclass(frozen=True)
class SegmentationResult:
    cell_labels: np.ndarray


def build_segmentation_model(
    configuration: ApplicationConfiguration,
) -> CellposeModel:
    """Construct a Cellpose-SAM model once so it can be reused across patches."""
    return CellposeModel(
        pretrained_model="cpsam",
        gpu=configuration.segmentation.use_gpu,
    )


def segment_cells_from_marker_images(
    nuclear_image: np.ndarray,
    cytoplasmic_image: np.ndarray,
    configuration: ApplicationConfiguration,
    model: CellposeModel,
) -> SegmentationResult:
    """Segment cells using Cellpose-SAM with nuclear and cytoplasmic channels."""
    image_stack = np.stack(
        [cytoplasmic_image, nuclear_image],
        axis=-1,
    )
    labels, _, _ = model.eval(
        image_stack,
        diameter=configuration.segmentation.cell_diameter_pixels,
    )
    labels = relabel_sequentially(labels)
    return SegmentationResult(cell_labels=labels)


def relabel_sequentially(labels: np.ndarray) -> np.ndarray:
    """Renumber labels to a contiguous 1..N sequence, preserving background as 0."""
    relabeled_labels, _, _ = segmentation.relabel_sequential(labels)
    return relabeled_labels.astype(np.int32)
