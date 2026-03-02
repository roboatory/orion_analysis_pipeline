from __future__ import annotations

import argparse
import time

import polars as pl

from orion.annotation import annotate_cell_types
from orion.configuration import (
    ApplicationConfiguration,
    convert_model_to_dictionary,
    load_configuration,
)
from orion.image_metadata import load_slide_metadata
from orion.input_output import (
    ensure_directory,
    read_histology_region_of_interest,
    read_readouts_region_of_interest,
    read_segmentation_region_of_interest,
    write_data_frame,
    write_image_stack,
    write_yaml_file,
)
from orion.normalization import normalize_and_threshold_marker_intensities
from orion.preprocessing import preprocess_region_of_interest_patch
from orion.quality_control import (
    save_cell_type_map,
    save_marker_panels,
    save_neighborhood_map,
    save_region_of_interest_preview,
    save_segmentation_overlay,
    save_threshold_histograms,
)
from orion.quantification import quantify_cells_in_region_of_interest
from orion.region_of_interest import choose_region_of_interest
from orion.segmentation import (
    segment_cells_from_nuclear_image,
    summarize_segmentation_validation,
)
from orion.spatial_analysis import compute_spatial_analysis_outputs


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Orion patch-first multiplex pipeline"
    )
    argument_parser.add_argument("command", choices=["run"])
    argument_parser.add_argument("--configuration", required=True)
    return argument_parser


def print_run_startup_summary(
    configuration: ApplicationConfiguration,
    slide_metadata,
) -> None:
    inspection_summary = {
        "sample_identifier": configuration.sample_identifier,
        "readouts": str(configuration.input_paths.readouts),
        "marker_count": len(slide_metadata.marker_names),
        "image_width_pixels": slide_metadata.width_pixels,
        "image_height_pixels": slide_metadata.height_pixels,
        "pixel_size_x_micrometers": slide_metadata.pixel_size_x_micrometers,
        "pixel_size_y_micrometers": slide_metadata.pixel_size_y_micrometers,
        "marker_names": slide_metadata.marker_names,
        "open_microscopy_environment_channel_names": (
            slide_metadata.open_microscopy_environment_channel_names
        ),
    }
    for key, value in inspection_summary.items():
        print(f"{key}: {value}")
    configured_cell_type_names = [
        cell_type_rule.name for cell_type_rule in configuration.annotation.cell_types
    ]
    print(f"configured_cell_types: {configured_cell_type_names}")
    for cell_type_rule in configuration.annotation.cell_types:
        print(f"{cell_type_rule.name}_markers: {cell_type_rule.positive_markers}")


def run_pipeline_command(configuration: ApplicationConfiguration) -> None:
    start_time_seconds = time.time()
    slide_metadata = load_slide_metadata(configuration)
    print_run_startup_summary(configuration, slide_metadata)
    sample_output_directory = ensure_directory(configuration.sample_output_directory)
    quality_control_directory = ensure_directory(
        sample_output_directory / "quality_control"
    )
    spatial_analysis_directory = ensure_directory(
        sample_output_directory / "spatial_analysis"
    )
    extracted_patch_directory = ensure_directory(sample_output_directory / "patch")

    (
        selected_region_of_interest,
        ranked_candidate_data_frame,
    ) = choose_region_of_interest(configuration, slide_metadata)
    ranked_candidate_data_frame.write_csv(
        quality_control_directory / "region_of_interest_candidates.csv"
    )

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
    existing_segmentation_patch = (
        read_segmentation_region_of_interest(
            configuration.input_paths.existing_segmentation,
            selected_region_of_interest,
        )
        if configuration.input_paths.existing_segmentation
        else None
    )

    if histology_patch is not None:
        save_region_of_interest_preview(
            histology_patch,
            selected_region_of_interest,
            quality_control_directory / "region_of_interest_preview.png",
        )

    preprocessing_result = preprocess_region_of_interest_patch(
        raw_readout_stack,
        slide_metadata.marker_names,
        configuration,
    )
    corrected_patch_stack_path = write_image_stack(
        extracted_patch_directory / "corrected_region_of_interest_stack.tiff",
        preprocessing_result.corrected_image_stack,
        slide_metadata.marker_names,
    )

    marker_name_to_index = {
        marker_name: index
        for index, marker_name in enumerate(slide_metadata.marker_names)
    }
    target_cell_count = target_cell_count_for_region_of_interest(
        configuration,
        selected_region_of_interest,
    )
    segmentation_result = segment_cells_from_nuclear_image(
        preprocessing_result.corrected_image_stack[
            marker_name_to_index[configuration.channels.nuclear_marker]
        ],
        configuration,
        target_cell_count=target_cell_count,
    )
    segmentation_validation_summary = summarize_segmentation_validation(
        existing_segmentation_patch,
        segmentation_result.kept_cell_labels,
    )

    intensity_image_by_marker = {}
    for marker_name, marker_index in marker_name_to_index.items():
        if marker_name in configuration.channels.technical_markers:
            intensity_image_by_marker[marker_name] = raw_readout_stack[
                marker_index
            ].astype(float)
        else:
            intensity_image_by_marker[marker_name] = (
                preprocessing_result.corrected_image_stack[marker_index].astype(float)
            )

    raw_cell_measurements = quantify_cells_in_region_of_interest(
        segmentation_result.kept_cell_labels,
        intensity_image_by_marker,
        slide_metadata.marker_names,
        selected_region_of_interest,
        slide_metadata.pixel_size_x_micrometers,
        slide_metadata.pixel_size_y_micrometers,
    )
    (
        normalized_cell_measurements,
        thresholded_cell_measurements,
        threshold_summary,
    ) = normalize_and_threshold_marker_intensities(
        raw_cell_measurements,
        slide_metadata.marker_names,
        configuration,
    )
    annotated_cell_measurements = annotate_cell_types(
        thresholded_cell_measurements,
        configuration,
    )
    spatial_analysis_outputs = compute_spatial_analysis_outputs(
        annotated_cell_measurements,
        configuration,
    )

    write_data_frame(
        raw_cell_measurements,
        sample_output_directory / "cells_raw_measurements",
    )
    write_data_frame(
        normalized_cell_measurements,
        sample_output_directory / "cells_normalized_measurements",
    )
    write_data_frame(
        thresholded_cell_measurements,
        sample_output_directory / "cells_thresholded_measurements",
    )
    write_data_frame(
        annotated_cell_measurements,
        sample_output_directory / "cells_annotated_measurements",
    )
    threshold_summary.write_csv(sample_output_directory / "marker_thresholds.csv")
    spatial_analysis_outputs.adjoining_cell_type_pairs.write_csv(
        spatial_analysis_directory / "adjoining_cell_type_pairs.csv"
    )
    spatial_analysis_outputs.nearest_neighbor_distances.write_csv(
        spatial_analysis_directory / "nearest_neighbor_distances.csv"
    )
    spatial_analysis_outputs.neighborhood_features.write_parquet(
        spatial_analysis_directory / "neighborhood_features.parquet"
    )
    spatial_analysis_outputs.neighborhood_clusters.write_csv(
        spatial_analysis_directory / "neighborhood_clusters.csv"
    )
    spatial_analysis_outputs.cell_type_clusters.write_csv(
        spatial_analysis_directory / "cell_type_clusters.csv"
    )

    marker_panel_images = {
        marker_name: preprocessing_result.corrected_image_by_marker[marker_name]
        for marker_name in ["Hoechst", "CD45", "Pan-CK", "CD31", "CD68", "SMA"]
        if marker_name in preprocessing_result.corrected_image_by_marker
    }
    if marker_panel_images:
        save_marker_panels(
            marker_panel_images,
            quality_control_directory / "marker_panels.png",
            list(marker_panel_images.keys()),
        )
    if histology_patch is not None:
        save_segmentation_overlay(
            histology_patch,
            segmentation_result.kept_cell_labels,
            quality_control_directory / "segmentation_overlay.png",
        )
    save_threshold_histograms(
        normalized_cell_measurements,
        threshold_summary,
        configuration.biological_marker_names,
        quality_control_directory / "threshold_histograms.png",
    )
    save_cell_type_map(
        annotated_cell_measurements,
        quality_control_directory / "cell_type_map.png",
    )
    save_neighborhood_map(
        spatial_analysis_outputs.neighborhood_clusters,
        annotated_cell_measurements,
        quality_control_directory / "neighborhood_cluster_map.png",
    )

    manifest_payload = {
        "sample_identifier": configuration.sample_identifier,
        "selected_region_of_interest": selected_region_of_interest.as_dictionary(),
        "configuration": convert_model_to_dictionary(configuration),
        "slide_metadata": slide_metadata.as_dictionary(),
        "preprocessing": {
            "autofluorescence_scale_by_marker": (
                preprocessing_result.autofluorescence_scale_by_marker
            ),
            "corrected_region_of_interest_stack": str(corrected_patch_stack_path),
        },
        "annotation_configuration": convert_model_to_dictionary(
            configuration.annotation
        ),
        "annotation_marker_names": configuration.annotation_marker_names,
        "annotation_summary": (
            annotated_cell_measurements.group_by("cell_type")
            .len()
            .sort("cell_type")
            .to_dicts()
        ),
        "segmentation": {
            "boundary_touching_count": segmentation_result.boundary_touching_count,
            "cell_count": int(segmentation_result.kept_cell_labels.max()),
            "chosen_peak_minimum_distance_pixels": (
                segmentation_result.chosen_peak_minimum_distance_pixels
            ),
            "validation_summary": (
                segmentation_validation_summary.as_dictionary()
                if segmentation_validation_summary
                else None
            ),
        },
        "outputs": {
            "cells_raw_measurements": str(
                sample_output_directory / "cells_raw_measurements.parquet"
            ),
            "cells_normalized_measurements": str(
                sample_output_directory / "cells_normalized_measurements.parquet"
            ),
            "cells_thresholded_measurements": str(
                sample_output_directory / "cells_thresholded_measurements.parquet"
            ),
            "cells_annotated_measurements": str(
                sample_output_directory / "cells_annotated_measurements.parquet"
            ),
            "marker_thresholds": str(sample_output_directory / "marker_thresholds.csv"),
            "region_of_interest_candidates": str(
                quality_control_directory / "region_of_interest_candidates.csv"
            ),
            "region_of_interest_preview": str(
                quality_control_directory / "region_of_interest_preview.png"
            ),
            "segmentation_overlay": str(
                quality_control_directory / "segmentation_overlay.png"
            ),
            "marker_panels": str(quality_control_directory / "marker_panels.png"),
            "threshold_histograms": str(
                quality_control_directory / "threshold_histograms.png"
            ),
            "cell_type_map": str(quality_control_directory / "cell_type_map.png"),
            "neighborhood_cluster_map": str(
                quality_control_directory / "neighborhood_cluster_map.png"
            ),
            "adjoining_cell_type_pairs": str(
                spatial_analysis_directory / "adjoining_cell_type_pairs.csv"
            ),
            "nearest_neighbor_distances": str(
                spatial_analysis_directory / "nearest_neighbor_distances.csv"
            ),
            "neighborhood_features": str(
                spatial_analysis_directory / "neighborhood_features.parquet"
            ),
            "neighborhood_clusters": str(
                spatial_analysis_directory / "neighborhood_clusters.csv"
            ),
            "cell_type_clusters": str(
                spatial_analysis_directory / "cell_type_clusters.csv"
            ),
        },
        "runtime_seconds": time.time() - start_time_seconds,
        "bootstrap_inputs": {
            "used_existing_quantifications_for_region_of_interest_selection": (
                configuration.input_paths.existing_quantifications is not None
            ),
            "used_existing_segmentation_for_validation": (
                configuration.input_paths.existing_segmentation is not None
            ),
        },
    }
    write_yaml_file(sample_output_directory / "manifest.yaml", manifest_payload)
    print(f"pipeline_output_directory: {sample_output_directory}")
    print(
        "selected_region_of_interest: " f"{selected_region_of_interest.as_dictionary()}"
    )


def target_cell_count_for_region_of_interest(
    configuration: ApplicationConfiguration,
    region_of_interest,
) -> int | None:
    if configuration.input_paths.existing_quantifications is None:
        return None
    return int(
        pl.scan_csv(configuration.input_paths.existing_quantifications)
        .filter(
            (pl.col("X_centroid") >= region_of_interest.x_pixels)
            & (pl.col("X_centroid") < region_of_interest.x_end_pixels)
            & (pl.col("Y_centroid") >= region_of_interest.y_pixels)
            & (pl.col("Y_centroid") < region_of_interest.y_end_pixels)
        )
        .select(pl.len())
        .collect()
        .item()
    )


def main(argument_values: list[str] | None = None) -> int:
    argument_parser = build_argument_parser()
    parsed_arguments = argument_parser.parse_args(argument_values)
    configuration = load_configuration(parsed_arguments.configuration)
    run_pipeline_command(configuration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
