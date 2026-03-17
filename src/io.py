from __future__ import annotations

import xml.etree.ElementTree as xml_element_tree
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import tifffile
import yaml

from src.data_models import RegionOfInterestBox, SlideMetadata

if TYPE_CHECKING:
    from src.configuration import ApplicationConfiguration


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_marker_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file_handle:
        return [line.strip() for line in file_handle if line.strip()]


def load_slide_metadata(configuration: ApplicationConfiguration) -> SlideMetadata:
    marker_names = read_marker_names(configuration.input_paths.markers)
    with tifffile.TiffFile(configuration.input_paths.readouts) as tiff_file:
        primary_series = tiff_file.series[0]
        (
            open_microscopy_environment_channel_names,
            physical_size_x_micrometers,
            physical_size_y_micrometers,
        ) = parse_open_microscopy_environment_metadata(tiff_file.ome_metadata)
        channel_count, height_pixels, width_pixels = primary_series.shape
    if physical_size_x_micrometers is None or physical_size_y_micrometers is None:
        raise ValueError(
            "Physical pixel size is missing from the open microscopy environment metadata. "
            "Add it to the source data before running the pipeline."
        )
    if len(marker_names) != channel_count:
        raise ValueError(
            "Marker count mismatch: markers.csv has "
            f"{len(marker_names)} rows but TIFF has {channel_count} channels."
        )
    return SlideMetadata(
        readouts_path=configuration.input_paths.readouts,
        histology_path=configuration.input_paths.histology,
        width_pixels=width_pixels,
        height_pixels=height_pixels,
        channel_count=channel_count,
        pixel_size_x_micrometers=physical_size_x_micrometers,
        pixel_size_y_micrometers=physical_size_y_micrometers,
        open_microscopy_environment_channel_names=open_microscopy_environment_channel_names,
        marker_names=marker_names,
    )


def parse_open_microscopy_environment_metadata(
    open_microscopy_environment_xml: str | None,
) -> tuple[list[str], float | None, float | None]:
    if not open_microscopy_environment_xml:
        return [], None, None
    xml_root = xml_element_tree.fromstring(open_microscopy_environment_xml)
    namespace = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    pixel_element = xml_root.find(".//ome:Pixels", namespace)
    if pixel_element is None:
        return [], None, None
    physical_size_x_micrometers = pixel_element.attrib.get("PhysicalSizeX")
    physical_size_y_micrometers = pixel_element.attrib.get("PhysicalSizeY")
    channel_elements = xml_root.findall(".//ome:Channel", namespace)
    open_microscopy_environment_channel_names = [
        channel_element.attrib.get("Name", "") for channel_element in channel_elements
    ]
    return (
        open_microscopy_environment_channel_names,
        float(physical_size_x_micrometers)
        if physical_size_x_micrometers is not None
        else None,
        float(physical_size_y_micrometers)
        if physical_size_y_micrometers is not None
        else None,
    )


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


def write_csv(data_frame: pl.DataFrame, path: Path) -> Path:
    ensure_directory(path.parent)
    data_frame.write_csv(path)
    return path


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


def write_label_image(path: Path, label_image: Any) -> Path:
    ensure_directory(path.parent)
    tifffile.imwrite(path, label_image)
    return path


def write_rgb_image(path: Path, image: Any) -> Path:
    ensure_directory(path.parent)
    tifffile.imwrite(path, image)
    return path
