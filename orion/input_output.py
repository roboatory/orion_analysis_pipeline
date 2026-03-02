from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import tifffile
import yaml

from orion.data_models import RegionOfInterestBox


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_marker_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file_handle:
        return [line.strip() for line in file_handle if line.strip()]


def read_readouts_region_of_interest(
    path: Path, region_of_interest: RegionOfInterestBox
) -> Any:
    return tifffile.imread(
        path,
        selection=(
            slice(None),
            slice(region_of_interest.y_pixels, region_of_interest.y_end_pixels),
            slice(region_of_interest.x_pixels, region_of_interest.x_end_pixels),
        ),
    )


def read_histology_region_of_interest(
    path: Path, region_of_interest: RegionOfInterestBox
) -> Any:
    return tifffile.imread(
        path,
        selection=(
            slice(region_of_interest.y_pixels, region_of_interest.y_end_pixels),
            slice(region_of_interest.x_pixels, region_of_interest.x_end_pixels),
            slice(None),
        ),
    )


def read_segmentation_region_of_interest(
    path: Path, region_of_interest: RegionOfInterestBox
) -> Any:
    return tifffile.imread(
        path,
        selection=(
            slice(region_of_interest.y_pixels, region_of_interest.y_end_pixels),
            slice(region_of_interest.x_pixels, region_of_interest.x_end_pixels),
        ),
    )


def write_data_frame(data_frame: pl.DataFrame, base_path: Path) -> list[Path]:
    ensure_directory(base_path.parent)
    comma_separated_values_path = base_path.with_suffix(".csv")
    parquet_path = base_path.with_suffix(".parquet")
    data_frame.write_csv(comma_separated_values_path)
    data_frame.write_parquet(parquet_path)
    return [comma_separated_values_path, parquet_path]


def write_yaml_file(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(payload, file_handle, sort_keys=False)


def write_image_stack(path: Path, image_stack: Any, marker_names: list[str]) -> Path:
    ensure_directory(path.parent)
    tifffile.imwrite(
        path,
        image_stack,
        metadata={"axes": "CYX", "markers": marker_names},
    )
    return path
