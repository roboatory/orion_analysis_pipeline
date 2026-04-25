# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```bash
uv run python3 main.py --configuration <config.yaml> --mode patch                  # full pipeline
uv run python3 main.py --configuration <config.yaml> --mode patch --stage segment  # single stage
uv run pytest                                                            # all tests
uv run pytest tests/test_segmentation.py                                 # single file
uv run pytest tests/test_segmentation.py::test_function_name -v          # single test
uv run ruff check src/ tests/ main.py                                    # lint
uv run ruff format src/ tests/ main.py                                   # format
uv run pre-commit run --all-files                                        # all hooks
```

Available `--stage` values: `select-roi`, `preprocess`, `segment`, `quantify`, `annotate`, `spatial`. Each stage reads its inputs from the output directory (written by previous stages) and writes its own outputs there. Omitting `--stage` executes all stages in order. `--mode` is required; `whole-slide` is reserved but not yet implemented.

## Architecture

This is a **multiplex imaging analysis pipeline** for patch-based ORION-style multiplex immunofluorescence. It processes OME-TIFF images through sequential preprocessing, segmentation, quantification, normalization, annotation, and spatial analysis to generate fully annotated single-cell datasets with cell type spatial metrics. Spatial analysis uses only cell type labels and centroids—not pixel intensities. `main.py` handles CLI argument parsing; pipeline control is in `src/pipeline.py`; domain logic resides in `src/`.

### Inputs and outputs

**Inputs**: multiplex image as OME-TIFF, channel metadata file mapping channel index to marker name, optional registered H&E image as OME-TIFF for anatomical context.

**Outputs**: AF-corrected image stack (TIFF), labeled segmentation mask overlaid onto image (TIFF), per-cell feature table with marker intensities, morphology, and coordinates (CSV), cell type annotations per cell (CSV), spatial metrics table derived from cell type labels (CSV), visualization figures (PNG), and configuration snapshot (YAML).

### Pipeline stages (linear, deterministic)

```
Configuration → ROI Selection → Preprocessing → Segmentation → Quantification → Annotation → Spatial Analysis
```

Each stage in `pipeline.py` is a self-contained function that reads inputs from disk and writes outputs to the sample output directory. The full pipeline calls them in sequence. Intermediate outputs allow stages to be re-run individually.

1. **Configuration** (`configuration.py` → `load_configuration()`): Reads a YAML file into validated Pydantic models. Checks that input files exist, marker names match the TIFF channel count, and annotation rules reference valid markers.

2. **ROI Selection** (`region_of_interest.py` → `choose_region_of_interest()`): This stage randomly samples candidate patches, scores each one on tissue coverage and signal quality, and picks the best patch as the region of interest for all downstream work.

3. **Preprocessing** (`preprocessing.py` → `preprocess_region_of_interest_patch()`): Multiplex images contain autofluorescence — background glow that contaminates every channel. This stage estimates how much of the autofluorescence channel leaks into each biological channel (via least-squares scaling) and subtracts it, producing a corrected image stack.

4. **Segmentation** (`segmentation.py` → `segment_cells_from_marker_images()`): Finds individual cells in the image using Cellpose-SAM (cpsam), a SAM-based transformer model. The nuclear and cytoplasmic marker channels are stacked and passed to the model, which produces instance segmentation masks. Labels are relabeled sequentially after inference.

5. **Quantification** (`quantification.py` → `quantify_cells_in_region_of_interest()`): With cell boundaries defined, this stage measures each cell: its centroid coordinates, area, shape (eccentricity, solidity), and the mean intensity of every marker channel within its mask. The result is a single row per cell in a feature table.

6. **Annotation** (`annotation.py` → `annotate_cells()`): Converts raw marker intensities into cell type labels. Each marker is arcsinh-normalized, then split into positive/negative using an Otsu threshold (with a quantile fallback). Cells are assigned a type based on which markers they are positive for, following boolean rules defined in the configuration.

7. **Spatial Analysis** (`spatial_analysis.py` → `compute_spatial_analysis()`): Characterizes how cell types are arranged relative to each other. Builds a k-nearest-neighbor graph from cell centroids, computes neighborhood composition features, clusters cells into spatial domains via k-means, and tests whether cell type pairs are spatially enriched or depleted using permutation-based statistics.

| Stage | Module | Entry function |
|---|---|---|
| Config loading & validation | `configuration.py` | `load_configuration()` |
| TIFF metadata + marker names | `io.py` | `load_slide_metadata()` |
| Best-patch selection | `region_of_interest.py` | `choose_region_of_interest()` |
| Autofluorescence subtraction | `preprocessing.py` | `preprocess_region_of_interest_patch()` |
| Cellpose-SAM cell segmentation | `segmentation.py` | `segment_cells_from_marker_images()` |
| Per-cell morphology & intensity | `quantification.py` | `quantify_cells_in_region_of_interest()` |
| Marker thresholding & cell typing | `annotation.py` | `annotate_cells()` |
| k-NN neighborhoods, domains, adjacency | `spatial_analysis.py` | `compute_spatial_analysis()` |
| All file writing & visualization | `io.py` | various `write_*` / `save_*` functions |
| Pipeline orchestration | `pipeline.py` | `run_patch_pipeline()` / `run_<stage>()` |
| Logging capture | `logging.py` | `capture_runtime_logging()` |

### Configuration

YAML-driven, validated by nested Pydantic v2 models in `configuration.py`. Root model is `ApplicationConfiguration`. Cross-field validators ensure markers exist in the markers file, channel names are distinct, and annotation rules reference valid markers. `convert_model_to_dictionary()` serializes back to plain dicts for YAML snapshots.

### Data models

Frozen dataclasses in `data_models.py`: `RegionOfInterestBox`, `SlideMetadata`, `SegmentationValidationSummary`, plus result dataclasses (`PreprocessingResult`, `SegmentationResult`, `AnnotationResult`, `SpatialAnalysisResult`) returned by each pipeline stage.

### Key conventions

- **Image arrays**: `(channels, height, width)` for stacks, `(height, width)` for single-channel, `(height, width, 3)` for RGB
- **Label images**: int32, 0 = background, 1+ = cell IDs (sequential after relabeling)
- **Coordinates**: pixel origin top-left; cell centroids offset by ROI origin; micrometers = pixels × pixel_size
- **DataFrames**: Polars throughout (not pandas)
- **Determinism**: all randomness seeded from `sample_identifier` via crc32; identical inputs → identical outputs
- **Patch-first**: full whole-slide processing is not yet implemented; patch-based analysis avoids loading the entire OME-TIFF into memory on limited-RAM systems
- **Transparency**: no complex preprocessing beyond autofluorescence subtraction; normalization is arcsinh with a fixed cofactor; cell typing uses simple boolean logic on marker thresholds defined in config

### Output structure

<details>
<summary><strong>Output Directory Structure</strong></summary>

```
output_directory/<sample_identifier>/
├── raw_patch.tif
├── roi_metadata.yaml
├── histology_patch.tif
├── corrected_patch.tif
├── preprocessing_comparison.png
├── segmentation_mask.npy
├── segmentation_overlay.tif
├── cell_features.csv
├── cell_annotations.csv
├── cell_type_map.png
├── spatial_metrics.csv
├── spatial_domain_map.png
└── configuration_snapshot.yaml
```

</details>

## Testing

- One test file per module (`tests/test_<module>.py`)
- Unit tests use dummy config objects and synthetic images (circles, rectangles)
- `tests/conftest.py` adds repo root to `sys.path`
