from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RegionOfInterestBox:
    x_pixels: int
    y_pixels: int
    width_pixels: int
    height_pixels: int

    @property
    def x_end_pixels(self) -> int:
        return self.x_pixels + self.width_pixels

    @property
    def y_end_pixels(self) -> int:
        return self.y_pixels + self.height_pixels

    def as_dictionary(self) -> dict[str, int]:
        return {
            "x_pixels": self.x_pixels,
            "y_pixels": self.y_pixels,
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
        }


@dataclass(frozen=True)
class SlideMetadata:
    readouts_path: Path
    histology_path: Path | None
    width_pixels: int
    height_pixels: int
    channel_count: int
    pixel_size_x_micrometers: float
    pixel_size_y_micrometers: float
    open_microscopy_environment_channel_names: list[str]
    marker_names: list[str]

    def as_dictionary(self) -> dict[str, Any]:
        return {
            "readouts_path": str(self.readouts_path),
            "histology_path": str(self.histology_path) if self.histology_path else None,
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
            "channel_count": self.channel_count,
            "pixel_size_x_micrometers": self.pixel_size_x_micrometers,
            "pixel_size_y_micrometers": self.pixel_size_y_micrometers,
            "open_microscopy_environment_channel_names": self.open_microscopy_environment_channel_names,
            "marker_names": self.marker_names,
        }


@dataclass(frozen=True)
class SegmentationValidationSummary:
    existing_cell_count: int
    new_cell_count: int
    existing_median_area_square_pixels: float
    new_median_area_square_pixels: float
    centroid_density_overlap: float

    def as_dictionary(self) -> dict[str, Any]:
        return {
            "existing_cell_count": self.existing_cell_count,
            "new_cell_count": self.new_cell_count,
            "existing_median_area_square_pixels": self.existing_median_area_square_pixels,
            "new_median_area_square_pixels": self.new_median_area_square_pixels,
            "centroid_density_overlap": self.centroid_density_overlap,
        }
