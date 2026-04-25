from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegionOfInterestBox:
    x_pixels: int
    y_pixels: int
    width_pixels: int
    height_pixels: int

    @property
    def x_end_pixels(self) -> int:
        """Return the exclusive right-edge x coordinate of the box."""
        return self.x_pixels + self.width_pixels

    @property
    def y_end_pixels(self) -> int:
        """Return the exclusive bottom-edge y coordinate of the box."""
        return self.y_pixels + self.height_pixels


@dataclass(frozen=True)
class SlideMetadata:
    readouts_path: Path
    histology_path: Path | None
    width_pixels: int
    height_pixels: int
    pixel_size_x_micrometers: float
    pixel_size_y_micrometers: float
    marker_names: list[str]


@dataclass(frozen=True)
class PatchEntry:
    patch_id: str
    region_of_interest: RegionOfInterestBox
