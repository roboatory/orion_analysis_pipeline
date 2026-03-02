from __future__ import annotations

import xml.etree.ElementTree as xml_element_tree

import tifffile

from orion.configuration import ApplicationConfiguration
from orion.data_models import SlideMetadata
from orion.input_output import read_marker_names


def load_slide_metadata(configuration: ApplicationConfiguration) -> SlideMetadata:
    marker_names = read_marker_names(configuration.input_paths.markers)
    with tifffile.TiffFile(configuration.input_paths.readouts) as tiff_file:
        primary_series = tiff_file.series[0]
        open_microscopy_environment_xml = tiff_file.ome_metadata
        (
            open_microscopy_environment_channel_names,
            physical_size_x_micrometers,
            physical_size_y_micrometers,
        ) = parse_open_microscopy_environment_metadata(open_microscopy_environment_xml)
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
        segmentation_path=configuration.input_paths.existing_segmentation,
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
