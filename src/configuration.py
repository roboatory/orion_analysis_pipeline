from __future__ import annotations

from pathlib import Path

import pydantic
import yaml
from pydantic import BaseModel, Field, model_validator

from src.io import read_marker_names


class InputPathConfiguration(BaseModel):
    readouts: Path
    markers: Path
    histology: Path | None = None


class ChannelConfiguration(BaseModel):
    nuclear_marker: str
    cytoplasmic_marker: str
    autofluorescence_marker: str


class RegionOfInterestConfiguration(BaseModel):
    patch_width_pixels: int
    patch_height_pixels: int
    candidate_patch_count: int = Field(default=12, ge=1)
    minimum_tissue_fraction: float = Field(default=0.35, ge=0.0, le=1.0)
    minimum_informative_channel_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    minimum_channel_signal_spread: float = Field(default=0.05, ge=0.0, le=1.0)


class AutofluorescenceSubtractionConfiguration(BaseModel):
    enabled: bool = True
    sample_pixels: int = 100_000
    clip_upper_quantile: float = 0.95


class PercentileClipConfiguration(BaseModel):
    lower_quantile: float = 0.005
    upper_quantile: float = 0.995


class PreprocessingConfiguration(BaseModel):
    autofluorescence_subtraction: AutofluorescenceSubtractionConfiguration
    percentile_clip: PercentileClipConfiguration


class SegmentationConfiguration(BaseModel):
    cell_diameter_pixels: float | None = None
    use_gpu: bool = False


class NormalizationConfiguration(BaseModel):
    arcsinh_cofactor: float = 150.0
    positive_fraction_minimum: float = 0.005
    positive_fraction_maximum: float = 0.70
    fallback_quantile: float = 0.90


class SpatialAnalysisConfiguration(BaseModel):
    nearest_neighbor_count: int = 10
    permutation_count: int = 100
    neighborhood_cluster_count: int = 6


class CellTypeAnnotationRuleConfiguration(BaseModel):
    name: str
    positive_markers: list[str] = Field(min_length=1)


class AnnotationConfiguration(BaseModel):
    cell_types: list[CellTypeAnnotationRuleConfiguration] = Field(min_length=1)


class ApplicationConfiguration(BaseModel):
    sample_identifier: str
    input_paths: InputPathConfiguration
    output_directory: Path
    channels: ChannelConfiguration
    region_of_interest: RegionOfInterestConfiguration
    preprocessing: PreprocessingConfiguration
    segmentation: SegmentationConfiguration
    normalization: NormalizationConfiguration
    spatial_analysis: SpatialAnalysisConfiguration
    annotation: AnnotationConfiguration

    @model_validator(mode="after")
    def validate_configuration_paths(self) -> "ApplicationConfiguration":
        """Verify that all configured input file paths exist on disk."""
        required_paths = [self.input_paths.readouts, self.input_paths.markers]
        for required_path in required_paths:
            if not required_path.exists():
                raise ValueError(f"Required path does not exist: {required_path}")
        optional_paths = [self.input_paths.histology]
        for optional_path in optional_paths:
            if optional_path is not None and not optional_path.exists():
                raise ValueError(f"Optional path does not exist: {optional_path}")
        return self

    def validate_marker_names(
        self,
        marker_names: list[str],
    ) -> "ApplicationConfiguration":
        """Check that all configured marker references resolve to markers in the panel file."""
        allowed_marker_names = set(marker_names)
        for required_marker_name in [
            self.channels.nuclear_marker,
            self.channels.cytoplasmic_marker,
            self.channels.autofluorescence_marker,
        ]:
            if required_marker_name not in allowed_marker_names:
                raise ValueError(
                    f"Unknown marker referenced in configuration: {required_marker_name}"
                )
        segmentation_marker_names = {
            self.channels.nuclear_marker,
            self.channels.cytoplasmic_marker,
        }
        if len(segmentation_marker_names) != 2:
            raise ValueError(
                "Nuclear and cytoplasmic markers must be different channels."
            )
        if self.channels.autofluorescence_marker in segmentation_marker_names:
            raise ValueError(
                "Autofluorescence marker must be different from segmentation markers."
            )
        cell_type_names = [
            cell_type_rule.name for cell_type_rule in self.annotation.cell_types
        ]
        if len(cell_type_names) != len(set(cell_type_names)):
            raise ValueError(
                "Duplicate cell type names are not allowed in annotation configuration."
            )
        for cell_type_rule in self.annotation.cell_types:
            if not cell_type_rule.positive_markers:
                raise ValueError(
                    f"Cell type '{cell_type_rule.name}' must define at least one positive marker."
                )
            for positive_marker_name in cell_type_rule.positive_markers:
                if positive_marker_name not in allowed_marker_names:
                    raise ValueError(
                        "Configured annotation marker "
                        f"'{positive_marker_name}' is not present in the panel markers file."
                    )
                if positive_marker_name == self.channels.autofluorescence_marker:
                    raise ValueError(
                        "Configured annotation marker "
                        f"'{positive_marker_name}' cannot be the autofluorescence marker."
                    )
        return self

    @property
    def marker_names(self) -> list[str]:
        """Read marker names from the configured markers file."""
        return read_marker_names(self.input_paths.markers)

    @property
    def sample_output_directory(self) -> Path:
        """Return the per-sample output directory path."""
        return self.output_directory / self.sample_identifier

    @property
    def annotation_marker_names(self) -> list[str]:
        """Return deduplicated, insertion-ordered marker names used across all annotation rules."""
        seen_marker_names: set[str] = set()
        ordered_marker_names: list[str] = []
        for cell_type_rule in self.annotation.cell_types:
            for marker_name in cell_type_rule.positive_markers:
                if marker_name not in seen_marker_names:
                    ordered_marker_names.append(marker_name)
                    seen_marker_names.add(marker_name)
        return ordered_marker_names


def load_configuration(path: str | Path) -> ApplicationConfiguration:
    """Load and validate an application configuration from a YAML file."""
    configuration_path = Path(path)
    with configuration_path.open("r", encoding="utf-8") as file_handle:
        configuration_payload = yaml.safe_load(file_handle)
    try:
        configuration = ApplicationConfiguration.model_validate(configuration_payload)
    except pydantic.ValidationError as validation_error:
        raise ValueError(str(validation_error)) from validation_error
    marker_names = read_marker_names(configuration.input_paths.markers)
    return configuration.validate_marker_names(marker_names)
