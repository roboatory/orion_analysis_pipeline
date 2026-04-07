from __future__ import annotations

import logging
import zlib
from dataclasses import asdict
from pathlib import Path

import polars as pl
import tifffile
import yaml

from src.annotation import annotate_cells
from src.configuration import ApplicationConfiguration
from src.data_models import RegionOfInterestBox
from src.io import (
    build_marker_name_to_index,
    ensure_directory,
    load_slide_metadata,
    read_histology_region_of_interest,
    read_readouts_region_of_interest,
    save_cell_assignment_map,
    save_preprocessing_comparison,
    save_segmentation_overlay_image,
    write_csv,
    write_image_stack,
    write_label_image,
    write_yaml_file,
)
from src.preprocessing import preprocess_region_of_interest_patch
from src.quantification import quantify_cells_in_region_of_interest
from src.region_of_interest import choose_region_of_interest
from src.segmentation import segment_cells_from_marker_images
from src.spatial_analysis import compute_spatial_analysis

STAGES = [
    "select-roi",
    "preprocess",
    "segment",
    "quantify",
    "annotate",
    "spatial",
]


def run_patch_pipeline(
    configuration: ApplicationConfiguration,
    logger: logging.Logger,
    stage: str | None = None,
) -> None:
    """Execute all pipeline stages in order, or a single stage if specified."""
    if stage is not None:
        _STAGE_FUNCTIONS[stage](configuration, logger)
    else:
        for stage_name in STAGES:
            _STAGE_FUNCTIONS[stage_name](configuration, logger)

    write_yaml_file(
        configuration.sample_output_directory / "configuration_snapshot.yaml",
        configuration.model_dump(mode="json"),
    )

    logger.info("pipeline_output_directory: %s", configuration.sample_output_directory)


def run_select_roi(
    configuration: ApplicationConfiguration,
    logger: logging.Logger,
) -> None:
    """Select the best patch from the slide and write the raw patch and ROI metadata."""
    output_directory = ensure_directory(configuration.sample_output_directory)
    slide_metadata = load_slide_metadata(configuration)

    logger.info("mode: %s", "patch")
    logger.info("sample_identifier: %s", configuration.sample_identifier)
    logger.info("readouts: %s", configuration.input_paths.readouts)
    logger.info("marker_count: %s", len(slide_metadata.marker_names))
    logger.info("image_width_pixels: %s", slide_metadata.width_pixels)
    logger.info("image_height_pixels: %s", slide_metadata.height_pixels)
    logger.info("pixel_size_x_micrometers: %s", slide_metadata.pixel_size_x_micrometers)
    logger.info("pixel_size_y_micrometers: %s", slide_metadata.pixel_size_y_micrometers)

    selected_roi, candidate_data_frame = choose_region_of_interest(
        configuration, slide_metadata
    )
    selected_candidate = candidate_data_frame.filter(pl.col("selected")).to_dicts()
    if selected_candidate:
        logger.info("region_of_interest_quality: %s", selected_candidate[0])

    raw_patch = read_readouts_region_of_interest(
        configuration.input_paths.readouts, selected_roi
    )
    write_image_stack(
        output_directory / "raw_patch.tif",
        raw_patch,
        slide_metadata.marker_names,
    )

    if configuration.input_paths.histology:
        histology_patch = read_histology_region_of_interest(
            configuration.input_paths.histology, selected_roi
        )
        tifffile.imwrite(
            output_directory / "histology_patch.tif",
            histology_patch,
        )

    write_yaml_file(
        output_directory / "roi_metadata.yaml",
        {
            **asdict(selected_roi),
            "pixel_size_x_micrometers": slide_metadata.pixel_size_x_micrometers,
            "pixel_size_y_micrometers": slide_metadata.pixel_size_y_micrometers,
            "marker_names": slide_metadata.marker_names,
        },
    )

    logger.info("selected_region_of_interest: %s", asdict(selected_roi))


def run_preprocess(
    configuration: ApplicationConfiguration,
    logger: logging.Logger,
) -> None:
    """Subtract autofluorescence from the raw patch and save the corrected stack."""
    output_directory = configuration.sample_output_directory
    roi_metadata = _load_roi_metadata(output_directory)
    marker_names = roi_metadata["marker_names"]

    raw_patch = tifffile.imread(output_directory / "raw_patch.tif")

    random_seed = zlib.crc32(configuration.sample_identifier.encode("utf-8"))
    preprocessing_result = preprocess_region_of_interest_patch(
        raw_patch,
        marker_names,
        configuration,
        random_seed,
    )

    write_image_stack(
        output_directory / "corrected_patch.tif",
        preprocessing_result.corrected_image_stack,
        marker_names,
    )

    marker_name_to_index = build_marker_name_to_index(marker_names)
    comparison_index = marker_name_to_index[configuration.channels.cytoplasmic_marker]
    save_preprocessing_comparison(
        raw_patch[comparison_index],
        preprocessing_result.corrected_image_stack[comparison_index],
        configuration.channels.cytoplasmic_marker,
        output_directory / "preprocessing_comparison.png",
    )

    logger.info("preprocessing complete")


def run_segment(
    configuration: ApplicationConfiguration,
    logger: logging.Logger,
) -> None:
    """Segment cells from the corrected patch and write the label mask and overlay."""
    output_directory = configuration.sample_output_directory
    roi_metadata = _load_roi_metadata(output_directory)
    marker_name_to_index = build_marker_name_to_index(roi_metadata["marker_names"])

    corrected_patch = tifffile.imread(output_directory / "corrected_patch.tif")
    nuclear_index = marker_name_to_index[configuration.channels.nuclear_marker]
    cytoplasmic_index = marker_name_to_index[configuration.channels.cytoplasmic_marker]

    segmentation_result = segment_cells_from_marker_images(
        corrected_patch[nuclear_index],
        corrected_patch[cytoplasmic_index],
        configuration,
    )

    write_label_image(
        output_directory / "segmentation_mask.tif",
        segmentation_result.kept_cell_labels,
    )

    histology_path = output_directory / "histology_patch.tif"
    background_image = (
        tifffile.imread(histology_path)
        if histology_path.exists()
        else corrected_patch[cytoplasmic_index]
    )
    save_segmentation_overlay_image(
        background_image,
        segmentation_result.kept_cell_labels,
        output_directory / "segmentation_overlay.tif",
    )

    logger.info(
        "segmentation complete: %d cells",
        int(segmentation_result.kept_cell_labels.max()),
    )


def run_quantify(
    configuration: ApplicationConfiguration,
    logger: logging.Logger,
) -> None:
    """Measure per-cell morphology and marker intensities and write cell_features.csv."""
    output_directory = configuration.sample_output_directory
    roi_metadata = _load_roi_metadata(output_directory)
    marker_names = roi_metadata["marker_names"]
    marker_name_to_index = build_marker_name_to_index(marker_names)

    corrected_patch = tifffile.imread(output_directory / "corrected_patch.tif")
    raw_patch = tifffile.imread(output_directory / "raw_patch.tif")
    label_image = tifffile.imread(output_directory / "segmentation_mask.tif")

    region_of_interest = RegionOfInterestBox(
        x_pixels=roi_metadata["x_pixels"],
        y_pixels=roi_metadata["y_pixels"],
        width_pixels=roi_metadata["width_pixels"],
        height_pixels=roi_metadata["height_pixels"],
    )

    intensity_image_by_marker = {
        marker_name: (
            raw_patch[marker_index].astype(float)
            if marker_name == configuration.channels.autofluorescence_marker
            else corrected_patch[marker_index].astype(float)
        )
        for marker_name, marker_index in marker_name_to_index.items()
    }

    cell_features = quantify_cells_in_region_of_interest(
        label_image,
        intensity_image_by_marker,
        marker_names,
        region_of_interest,
        roi_metadata["pixel_size_x_micrometers"],
        roi_metadata["pixel_size_y_micrometers"],
    )

    write_csv(
        cell_features,
        output_directory / "cell_features.csv",
    )

    logger.info("quantification complete: %d cells", cell_features.height)


def run_annotate(
    configuration: ApplicationConfiguration,
    logger: logging.Logger,
) -> None:
    """Assign cell type labels from quantified features and save annotations and map."""
    output_directory = configuration.sample_output_directory
    cell_features = pl.read_csv(output_directory / "cell_features.csv")

    cell_annotations = annotate_cells(
        cell_features,
        configuration,
    )

    write_csv(
        cell_annotations,
        output_directory / "cell_annotations.csv",
    )

    save_cell_assignment_map(
        cell_annotations,
        "cell_type",
        "Cell type map",
        output_directory / "cell_type_map.png",
    )

    logger.info("annotation complete")


def run_spatial(
    configuration: ApplicationConfiguration,
    logger: logging.Logger,
) -> None:
    """Compute spatial neighborhoods, domains, and adjacency enrichment metrics."""
    output_directory = configuration.sample_output_directory
    cell_annotations = pl.read_csv(output_directory / "cell_annotations.csv")

    if "spatial_domain" in cell_annotations.columns:
        cell_annotations = cell_annotations.drop("spatial_domain")

    random_seed = zlib.crc32(configuration.sample_identifier.encode("utf-8"))
    spatial_analysis_result = compute_spatial_analysis(
        cell_annotations,
        configuration,
        random_seed,
    )

    write_csv(
        spatial_analysis_result.cell_annotations_with_domains,
        output_directory / "cell_annotations.csv",
    )
    write_csv(
        spatial_analysis_result.spatial_metrics,
        output_directory / "spatial_metrics.csv",
    )

    save_cell_assignment_map(
        spatial_analysis_result.cell_annotations_with_domains,
        "spatial_domain",
        "Spatial domain map",
        output_directory / "spatial_domain_map.png",
    )

    logger.info("spatial analysis complete")


def _load_roi_metadata(output_directory: Path) -> dict:
    """Load ROI metadata from the YAML file written by the select-roi stage."""
    metadata_path = output_directory / "roi_metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"ROI metadata not found at {metadata_path}. "
            "Run the select-roi stage first."
        )
    with metadata_path.open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


_STAGE_FUNCTIONS = {
    "select-roi": run_select_roi,
    "preprocess": run_preprocess,
    "segment": run_segment,
    "quantify": run_quantify,
    "annotate": run_annotate,
    "spatial": run_spatial,
}
