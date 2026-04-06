# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run python3 main.py run --configuration <config.yaml> --mode patch   # run pipeline
uv run pytest                                                            # all tests
uv run pytest tests/test_segmentation.py                                 # single file
uv run pytest tests/test_segmentation.py::test_function_name -v          # single test
uv run ruff check src/ tests/ main.py                                    # lint
uv run ruff format src/ tests/ main.py                                   # format
uv run pre-commit run --all-files                                        # all hooks
```

## Architecture

Orion is a **patch-first multiplex imaging pipeline** for single-image ORION-style multiplex immunofluorescence whole-slide analysis. It moves from preprocessing through segmentation, quantification, normalization, annotation, and spatial characterization to produce a fully annotated single-cell dataset with cell type-level spatial metrics. All spatial analyses operate on cell type labels and centroid coordinates, not pixel-level intensities. `main.py` is the CLI entry point; all domain logic lives under `src/`.

### Inputs and outputs

**Inputs**: multiplex image as OME-TIFF, channel metadata file mapping channel index to marker name, optional registered H&E image as OME-TIFF for anatomical context.

**Outputs**: AF-corrected image stack (TIFF), labeled segmentation mask overlaid onto image (TIFF), per-cell feature table with marker intensities, morphology, and coordinates (CSV), cell type annotations per cell (CSV), spatial metrics table derived from cell type labels (CSV), visualization figures (PNG), and configuration snapshot (YAML).

### Pipeline stages (linear, deterministic)

```
Configuration → Slide Metadata → ROI Selection → Preprocessing
→ Segmentation → Quantification → Annotation → Spatial Analysis → Output
```

| Stage | Module | Entry function |
|---|---|---|
| Config loading & validation | `configuration.py` | `load_configuration()` |
| TIFF metadata + marker names | `io.py` | `load_slide_metadata()` |
| Best-patch selection | `region_of_interest.py` | `choose_region_of_interest()` |
| Autofluorescence subtraction | `preprocessing.py` | `preprocess_region_of_interest_patch()` |
| Nuclear segmentation + cell expansion | `segmentation.py` | `segment_cells_from_marker_images()` |
| Per-cell morphology & intensity | `quantification.py` | `quantify_cells_in_region_of_interest()` |
| Marker thresholding & cell typing | `annotation.py` | `annotate_cells()` |
| k-NN neighborhoods, domains, adjacency | `spatial_analysis.py` | `compute_spatial_analysis()` |
| All file writing & visualization | `io.py` | various `write_*` / `save_*` functions |

`runtime_logging.py` wraps the pipeline run to capture stdout/stderr and Python warnings into a timestamped log file.

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

```
output_directory/<sample_identifier>/
    corrected_patch.tif, segmentation_mask.tif, segmentation_overlay.tif,
    cell_features.csv, cell_annotations.csv, spatial_metrics.csv,
    preprocessing_comparison.png, cell_type_map.png, spatial_domain_map.png,
    configuration_snapshot.yaml
```

## Testing

- One test file per module (`tests/test_<module>.py`)
- Unit tests use dummy config objects and synthetic images (circles, rectangles)
- Real-data tests gate on file existence with `@pytest.mark.skipif`; reference data lives under `data/`
- `tests/conftest.py` adds repo root to `sys.path`
- Integration test in `test_main.py` runs the full pipeline and checks all expected outputs exist
