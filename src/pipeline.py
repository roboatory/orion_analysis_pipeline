from __future__ import annotations

import logging
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
import tifffile

from src.annotation import annotate_cells
from src.configuration import ApplicationConfiguration
from src.constants import STAGES
from src.data_models import PatchEntry, SlideMetadata
from src.io import (
    build_marker_name_to_index,
    ensure_directory,
    format_patch_identifier,
    load_slide_metadata,
    parse_patch_entries,
    patch_output_directory,
    read_histology_region_of_interest,
    read_patches_manifest,
    read_readouts_region_of_interest,
    save_cell_assignment_map,
    save_preprocessing_comparison,
    save_segmentation_overlay_image,
    write_csv,
    write_image_stack,
    write_label_array,
    write_patches_manifest,
    write_yaml_file,
)
from src.preprocessing import preprocess_region_of_interest_patch
from src.quantification import quantify_cells_in_region_of_interest
from src.region_of_interest import choose_region_of_interest
from src.segmentation import (
    build_segmentation_model,
    segment_cells_from_marker_images,
)
from src.spatial_analysis import compute_spatial_analysis


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
    """Select non-overlapping analysis patches and write the manifest and per-patch ROI files."""
    sample_output_directory = ensure_directory(configuration.sample_output_directory)
    slide_metadata = load_slide_metadata(configuration)

    logger.info("mode: %s", "patch")
    logger.info("sample_identifier: %s", configuration.sample_identifier)
    logger.info("readouts: %s", configuration.input_paths.readouts)
    logger.info("marker_count: %s", len(slide_metadata.marker_names))
    logger.info("image_width_pixels: %s", slide_metadata.width_pixels)
    logger.info("image_height_pixels: %s", slide_metadata.height_pixels)
    logger.info("pixel_size_x_micrometers: %s", slide_metadata.pixel_size_x_micrometers)
    logger.info("pixel_size_y_micrometers: %s", slide_metadata.pixel_size_y_micrometers)
    logger.info(
        "analysis_patch_count: %s",
        configuration.region_of_interest.analysis_patch_count,
    )

    selected_regions_of_interest, candidate_data_frame = choose_region_of_interest(
        configuration, slide_metadata
    )

    write_csv(
        candidate_data_frame,
        sample_output_directory / "candidate_patches.csv",
    )

    patch_entries = [
        PatchEntry(
            patch_id=format_patch_identifier(patch_index),
            region_of_interest=region_of_interest,
        )
        for patch_index, region_of_interest in enumerate(selected_regions_of_interest)
    ]

    write_patches_manifest(
        sample_output_directory / "patches_manifest.yaml",
        configuration.sample_identifier,
        slide_metadata,
        patch_entries,
    )

    for patch_entry in patch_entries:
        _write_patch_inputs(
            configuration,
            slide_metadata,
            patch_entry,
            sample_output_directory,
            logger,
        )


def run_preprocess(
    configuration: ApplicationConfiguration,
    logger: logging.Logger,
) -> None:
    """Subtract autofluorescence from every patch and save the corrected stacks."""
    manifest_payload, patch_entries = _load_manifest_entries(configuration)
    marker_names = manifest_payload["marker_names"]

    for patch_entry in patch_entries:
        patch_directory = patch_output_directory(
            configuration.sample_output_directory, patch_entry.patch_id
        )
        raw_patch = tifffile.imread(patch_directory / "raw_patch.tif")
        patch_seed = derive_patch_seed(
            configuration.sample_identifier,
            _patch_index_from_id(patch_entry.patch_id),
        )
        preprocessing_result = preprocess_region_of_interest_patch(
            raw_patch,
            marker_names,
            configuration,
            patch_seed,
        )

        write_image_stack(
            patch_directory / "corrected_patch.tif",
            preprocessing_result.corrected_image_stack,
            marker_names,
        )

        marker_name_to_index = build_marker_name_to_index(marker_names)
        comparison_index = marker_name_to_index[
            configuration.channels.cytoplasmic_marker
        ]
        save_preprocessing_comparison(
            raw_patch[comparison_index],
            preprocessing_result.corrected_image_stack[comparison_index],
            configuration.channels.cytoplasmic_marker,
            patch_directory / "preprocessing_comparison.png",
        )

        logger.info("preprocessing complete: %s", patch_entry.patch_id)


def run_segment(
    configuration: ApplicationConfiguration,
    logger: logging.Logger,
) -> None:
    """Segment cells for every patch, building the Cellpose model once and reusing it."""
    manifest_payload, patch_entries = _load_manifest_entries(configuration)
    marker_name_to_index = build_marker_name_to_index(manifest_payload["marker_names"])
    segmentation_model = build_segmentation_model(configuration)

    for patch_entry in patch_entries:
        patch_directory = patch_output_directory(
            configuration.sample_output_directory, patch_entry.patch_id
        )
        corrected_patch = tifffile.imread(patch_directory / "corrected_patch.tif")
        nuclear_index = marker_name_to_index[configuration.channels.nuclear_marker]
        cytoplasmic_index = marker_name_to_index[
            configuration.channels.cytoplasmic_marker
        ]

        segmentation_result = segment_cells_from_marker_images(
            corrected_patch[nuclear_index],
            corrected_patch[cytoplasmic_index],
            configuration,
            segmentation_model,
        )

        write_label_array(
            patch_directory / "segmentation_mask.npy",
            segmentation_result.cell_labels,
        )

        histology_path = patch_directory / "histology_patch.tif"
        background_image = (
            tifffile.imread(histology_path)
            if histology_path.exists()
            else corrected_patch[cytoplasmic_index]
        )
        save_segmentation_overlay_image(
            background_image,
            segmentation_result.cell_labels,
            patch_directory / "segmentation_overlay.tif",
        )

        logger.info(
            "segmentation complete: %s, %d cells",
            patch_entry.patch_id,
            int(segmentation_result.cell_labels.max()),
        )


def run_quantify(
    configuration: ApplicationConfiguration,
    logger: logging.Logger,
) -> None:
    """Measure per-cell features and write cell_features.csv for every patch."""
    manifest_payload, patch_entries = _load_manifest_entries(configuration)
    marker_names = manifest_payload["marker_names"]
    marker_name_to_index = build_marker_name_to_index(marker_names)
    autofluorescence_marker = configuration.channels.autofluorescence_marker
    biological_marker_names = [
        name for name in marker_names if name != autofluorescence_marker
    ]
    pixel_size_x_micrometers = float(manifest_payload["pixel_size_x_micrometers"])
    pixel_size_y_micrometers = float(manifest_payload["pixel_size_y_micrometers"])

    for patch_entry in patch_entries:
        patch_directory = patch_output_directory(
            configuration.sample_output_directory, patch_entry.patch_id
        )
        corrected_patch = tifffile.imread(patch_directory / "corrected_patch.tif")
        label_image = np.load(patch_directory / "segmentation_mask.npy")

        intensity_image_by_marker = {
            marker_name: corrected_patch[marker_name_to_index[marker_name]].astype(
                float
            )
            for marker_name in biological_marker_names
        }

        cell_features = quantify_cells_in_region_of_interest(
            label_image,
            intensity_image_by_marker,
            biological_marker_names,
            patch_entry.region_of_interest,
            pixel_size_x_micrometers,
            pixel_size_y_micrometers,
        )

        write_csv(
            cell_features,
            patch_directory / "cell_features.csv",
        )

        logger.info(
            "quantification complete: %s, %d cells",
            patch_entry.patch_id,
            cell_features.height,
        )


def run_annotate(
    configuration: ApplicationConfiguration,
    logger: logging.Logger,
) -> None:
    """Assign cell type labels for every patch and save annotations and maps."""
    _, patch_entries = _load_manifest_entries(configuration)

    for patch_entry in patch_entries:
        patch_directory = patch_output_directory(
            configuration.sample_output_directory, patch_entry.patch_id
        )
        cell_features = pl.read_csv(patch_directory / "cell_features.csv")

        cell_annotations = annotate_cells(
            cell_features,
            configuration,
        )

        write_csv(
            cell_annotations,
            patch_directory / "cell_annotations.csv",
        )

        save_cell_assignment_map(
            cell_annotations,
            "cell_type",
            "Cell type map",
            patch_directory / "cell_type_map.png",
        )

        logger.info("annotation complete: %s", patch_entry.patch_id)


def run_spatial(
    configuration: ApplicationConfiguration,
    logger: logging.Logger,
) -> None:
    """Compute spatial neighborhoods, domains, and adjacency metrics for every patch."""
    _, patch_entries = _load_manifest_entries(configuration)

    for patch_entry in patch_entries:
        patch_directory = patch_output_directory(
            configuration.sample_output_directory, patch_entry.patch_id
        )
        cell_annotations = pl.read_csv(patch_directory / "cell_annotations.csv")

        if "spatial_domain" in cell_annotations.columns:
            cell_annotations = cell_annotations.drop("spatial_domain")

        patch_seed = derive_patch_seed(
            configuration.sample_identifier,
            _patch_index_from_id(patch_entry.patch_id),
        )
        spatial_analysis_result = compute_spatial_analysis(
            cell_annotations,
            configuration,
            patch_seed,
        )

        write_csv(
            spatial_analysis_result.cell_annotations_with_domains,
            patch_directory / "cell_annotations.csv",
        )
        write_csv(
            spatial_analysis_result.spatial_metrics,
            patch_directory / "spatial_metrics.csv",
        )

        save_cell_assignment_map(
            spatial_analysis_result.cell_annotations_with_domains,
            "spatial_domain",
            "Spatial domain map",
            patch_directory / "spatial_domain_map.png",
        )

        logger.info("spatial analysis complete: %s", patch_entry.patch_id)


def derive_patch_seed(sample_identifier: str, patch_index: int) -> int:
    """Derive a deterministic RNG seed unique to a (sample_identifier, patch_index) pair."""
    return zlib.crc32(f"{sample_identifier}:patch_{patch_index:03d}".encode("utf-8"))


def _write_patch_inputs(
    configuration: ApplicationConfiguration,
    slide_metadata: SlideMetadata,
    patch_entry: PatchEntry,
    sample_output_directory: Path,
    logger: logging.Logger,
) -> None:
    """Write per-patch raw inputs (raw_patch.tif, optional histology, roi_metadata.yaml)."""
    patch_directory = ensure_directory(
        patch_output_directory(sample_output_directory, patch_entry.patch_id)
    )

    raw_patch = read_readouts_region_of_interest(
        configuration.input_paths.readouts, patch_entry.region_of_interest
    )
    write_image_stack(
        patch_directory / "raw_patch.tif",
        raw_patch,
        slide_metadata.marker_names,
    )

    if configuration.input_paths.histology:
        histology_patch = read_histology_region_of_interest(
            configuration.input_paths.histology, patch_entry.region_of_interest
        )
        tifffile.imwrite(
            patch_directory / "histology_patch.tif",
            histology_patch,
        )

    write_yaml_file(
        patch_directory / "roi_metadata.yaml",
        {
            "patch_id": patch_entry.patch_id,
            **asdict(patch_entry.region_of_interest),
        },
    )

    logger.info(
        "selected_region_of_interest: %s %s",
        patch_entry.patch_id,
        asdict(patch_entry.region_of_interest),
    )


def _load_manifest_entries(
    configuration: ApplicationConfiguration,
) -> tuple[dict, list[PatchEntry]]:
    """Load the patches manifest and return both the raw payload and parsed patch entries."""
    manifest_path = configuration.sample_output_directory / "patches_manifest.yaml"
    manifest_payload = read_patches_manifest(manifest_path)
    patch_entries = parse_patch_entries(manifest_payload)
    return manifest_payload, patch_entries


def _patch_index_from_id(patch_id: str) -> int:
    """Parse the integer index out of a patch identifier like 'patch_003'."""
    return int(patch_id.rsplit("_", 1)[-1])


_STAGE_FUNCTIONS = {
    "select-roi": run_select_roi,
    "preprocess": run_preprocess,
    "segment": run_segment,
    "quantify": run_quantify,
    "annotate": run_annotate,
    "spatial": run_spatial,
}
