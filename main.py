from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from runtime_logging import capture_runtime_logging, resolve_log_path
from src.annotation import AnnotationResult, annotate_cells
from src.configuration import (
    ApplicationConfiguration,
    convert_model_to_dictionary,
    load_configuration,
)
from src.data_models import SlideMetadata
from src.io import (
    ensure_directory,
    load_slide_metadata,
    read_histology_region_of_interest,
    read_readouts_region_of_interest,
    write_csv,
    write_image_stack,
    write_label_image,
    write_yaml_file,
)
from src.preprocessing import PreprocessingResult, preprocess_region_of_interest_patch
from src.quality_control import (
    save_cell_assignment_map,
    save_preprocessing_comparison,
    save_segmentation_overlay_image,
)
from src.quantification import quantify_cells_in_region_of_interest
from src.region_of_interest import choose_region_of_interest
from src.segmentation import SegmentationResult, segment_cells_from_marker_images
from src.spatial_analysis import SpatialAnalysisResult, compute_spatial_analysis


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Orion patch-first multiplex pipeline"
    )
    argument_parser.add_argument("command", choices=["run"])
    argument_parser.add_argument("--configuration", required=True)
    argument_parser.add_argument(
        "--mode",
        choices=["patch", "whole-slide"],
        default="patch",
    )
    return argument_parser


def run_patch_pipeline(
    configuration: ApplicationConfiguration,
    logger: logging.Logger,
) -> None:
    slide_metadata: SlideMetadata = load_slide_metadata(configuration)
    logger.info("mode: %s", "patch")
    logger.info("sample_identifier: %s", configuration.sample_identifier)
    logger.info("readouts: %s", configuration.input_paths.readouts)
    logger.info("marker_count: %s", len(slide_metadata.marker_names))
    logger.info("image_width_pixels: %s", slide_metadata.width_pixels)
    logger.info("image_height_pixels: %s", slide_metadata.height_pixels)
    logger.info("pixel_size_x_micrometers: %s", slide_metadata.pixel_size_x_micrometers)
    logger.info("pixel_size_y_micrometers: %s", slide_metadata.pixel_size_y_micrometers)
    sample_output_directory = ensure_directory(configuration.sample_output_directory)
    selected_region_of_interest, candidate_data_frame = choose_region_of_interest(
        configuration,
        slide_metadata,
    )
    selected_candidate = candidate_data_frame.filter(pl.col("selected")).to_dicts()
    if selected_candidate:
        logger.info("region_of_interest_quality: %s", selected_candidate[0])

    raw_readout_stack = read_readouts_region_of_interest(
        configuration.input_paths.readouts,
        selected_region_of_interest,
    )
    histology_patch = (
        read_histology_region_of_interest(
            configuration.input_paths.histology,
            selected_region_of_interest,
        )
        if configuration.input_paths.histology
        else None
    )
    preprocessing_result: PreprocessingResult = preprocess_region_of_interest_patch(
        raw_readout_stack,
        slide_metadata.marker_names,
        configuration,
    )

    marker_name_to_index = {
        marker_name: index
        for index, marker_name in enumerate(slide_metadata.marker_names)
    }
    nuclear_index = marker_name_to_index[configuration.channels.nuclear_marker]
    cytoplasmic_index = marker_name_to_index[configuration.channels.cytoplasmic_marker]
    segmentation_result: SegmentationResult = segment_cells_from_marker_images(
        preprocessing_result.corrected_image_stack[nuclear_index],
        preprocessing_result.corrected_image_stack[cytoplasmic_index],
        configuration,
    )

    intensity_image_by_marker = {
        marker_name: (
            raw_readout_stack[marker_index].astype(float)
            if marker_name == configuration.channels.autofluorescence_marker
            else preprocessing_result.corrected_image_stack[marker_index].astype(float)
        )
        for marker_name, marker_index in marker_name_to_index.items()
    }
    cell_features: pl.DataFrame = quantify_cells_in_region_of_interest(
        segmentation_result.kept_cell_labels,
        intensity_image_by_marker,
        slide_metadata.marker_names,
        selected_region_of_interest,
        slide_metadata.pixel_size_x_micrometers,
        slide_metadata.pixel_size_y_micrometers,
    )
    annotation_result: AnnotationResult = annotate_cells(cell_features, configuration)
    spatial_analysis_result: SpatialAnalysisResult = compute_spatial_analysis(
        annotation_result.cell_annotations,
        configuration,
    )

    write_image_stack(
        sample_output_directory / "corrected_patch.tif",
        preprocessing_result.corrected_image_stack,
        slide_metadata.marker_names,
    )
    write_label_image(
        sample_output_directory / "segmentation_mask.tif",
        segmentation_result.kept_cell_labels,
    )
    comparison_marker_name = configuration.channels.cytoplasmic_marker
    comparison_marker_index = marker_name_to_index[comparison_marker_name]
    save_preprocessing_comparison(
        raw_readout_stack[comparison_marker_index],
        preprocessing_result.corrected_image_stack[comparison_marker_index],
        comparison_marker_name,
        sample_output_directory / "preprocessing_comparison.png",
    )
    background_image = (
        histology_patch
        if histology_patch is not None
        else preprocessing_result.corrected_image_stack[cytoplasmic_index]
    )
    save_segmentation_overlay_image(
        background_image,
        segmentation_result.kept_cell_labels,
        sample_output_directory / "segmentation_overlay.tif",
    )
    write_csv(cell_features, sample_output_directory / "cell_features.csv")
    write_csv(
        spatial_analysis_result.cell_annotations_with_domains,
        sample_output_directory / "cell_annotations.csv",
    )
    write_csv(
        spatial_analysis_result.spatial_metrics,
        sample_output_directory / "spatial_metrics.csv",
    )
    save_cell_assignment_map(
        spatial_analysis_result.cell_annotations_with_domains,
        "cell_type",
        "Cell type map",
        sample_output_directory / "cell_type_map.png",
    )
    save_cell_assignment_map(
        spatial_analysis_result.cell_annotations_with_domains,
        "spatial_domain",
        "Spatial domain map",
        sample_output_directory / "spatial_domain_map.png",
    )
    write_yaml_file(
        sample_output_directory / "configuration_snapshot.yaml",
        convert_model_to_dictionary(configuration),
    )
    logger.info("pipeline_output_directory: %s", sample_output_directory)
    logger.info(
        "selected_region_of_interest: %s",
        selected_region_of_interest.as_dictionary(),
    )


def main(argument_values: list[str] | None = None) -> int:
    parsed_arguments = build_argument_parser().parse_args(argument_values)
    log_path = resolve_log_path(Path(parsed_arguments.configuration))
    with capture_runtime_logging(log_path) as logger:
        try:
            if parsed_arguments.mode == "whole-slide":
                logger.error(
                    "Whole-slide mode is reserved but not implemented yet. Use --mode patch for now."
                )
                return 1
            configuration = load_configuration(parsed_arguments.configuration)
            run_patch_pipeline(configuration, logger)
            return 0
        except Exception:
            logger.exception("Pipeline execution failed.")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
