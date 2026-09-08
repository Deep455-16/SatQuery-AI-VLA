"""
satquery/controller/evidence_layer.py

Evidence Layer for SatQuery AI.

Builds structured evidence bundles from:
- Image metadata (geospatial, sensor, spectral)
- Image statistics (NDVI, brightness, band stats)
- Change detection results (for bi-temporal)
- Modality information (optical vs SAR characteristics)
- Preprocessing notes

The evidence bundle is passed to the Gemini adapter alongside the image,
grounding the LLM's reasoning in verifiable computed data rather than
visual guesswork alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from satquery.utils.image_io import ImageMeta
from satquery.utils.image_preprocessor import (
    ChangeDetectionResult,
    extract_image_statistics,
    format_statistics_for_evidence,
)


@dataclass
class EvidenceBundle:
    """All evidence collected before the LLM call."""
    image_metadata: list[str] = field(default_factory=list)    # per-image metadata strings
    image_statistics: list[str] = field(default_factory=list)  # per-image stats strings
    change_detection: Optional[str] = None                     # change detection evidence
    modality_notes: list[str] = field(default_factory=list)    # modality-specific notes
    preprocessing_notes: list[str] = field(default_factory=list)
    intent: str = ""
    input_type: str = ""

    def to_context_string(self) -> str:
        """Format all evidence into a single string for the Gemini prompt."""
        sections = []

        sections.append(f"Analysis type: {self.input_type.replace('_', ' ').title()}")
        sections.append(f"User intent: {self.intent.replace('_', ' ')}")

        if self.image_metadata:
            sections.append("\n--- Satellite Image Metadata ---")
            sections.extend(self.image_metadata)

        if self.image_statistics:
            sections.append("\n--- Image Statistics (computed) ---")
            sections.extend(self.image_statistics)

        if self.change_detection:
            sections.append("\n--- Change Detection Results (OpenCV) ---")
            sections.append(self.change_detection)

        if self.modality_notes:
            sections.append("\n--- Modality Notes ---")
            sections.extend(self.modality_notes)

        if self.preprocessing_notes:
            sections.append("\n--- Processing Notes ---")
            sections.extend(self.preprocessing_notes)

        return "\n".join(sections)


def build_evidence(
    arrays_and_metas: list[tuple],
    intent: str,
    input_type: str,
    change_result: Optional[ChangeDetectionResult] = None,
) -> EvidenceBundle:
    """
    Build an EvidenceBundle from images and their metadata.

    Args:
        arrays_and_metas: list of (numpy_array, ImageMeta) tuples
        intent: classified intent string (e.g., "vegetation_analysis")
        input_type: "single" | "cross_modal_pair" | "bitemporal_pair"
        change_result: populated for bi-temporal analysis
    """
    bundle = EvidenceBundle(intent=intent, input_type=input_type)

    for i, (arr, meta) in enumerate(arrays_and_metas):
        # Metadata string
        label = ""
        if len(arrays_and_metas) == 2:
            if input_type == "bitemporal_pair":
                label = f"[Image {'A (Before/Earlier)' if i == 0 else 'B (After/Later)'}] "
            elif input_type == "cross_modal_pair":
                label = f"[{'Optical' if meta.modality == 'optical' else 'SAR'} Image] "

        bundle.image_metadata.append(f"{label}Satellite image metadata:\n{meta.to_evidence_string()}")

        # Statistics
        try:
            stats = extract_image_statistics(arr, meta)
            stats_str = format_statistics_for_evidence(stats)
            bundle.image_statistics.append(f"{label}{stats_str}")
        except Exception as e:
            bundle.preprocessing_notes.append(f"Statistics extraction failed: {e}")

        # Modality notes
        if meta.modality == "sar":
            bundle.modality_notes.append(
                "SAR (Synthetic Aperture Radar) imagery characteristics:\n"
                "- Records microwave backscatter, not optical reflectance\n"
                "- Penetrates clouds, fog, and smoke — all-weather capability\n"
                "- Sensitive to surface roughness, moisture, and dielectric properties\n"
                "- Bright returns: rough/metal surfaces, buildings, water with ripples\n"
                "- Dark returns: smooth water, roads, sand\n"
                "- Does NOT provide color information; texture and structure are primary cues"
            )
        elif meta.modality == "optical":
            if meta.is_sentinel2:
                bundle.modality_notes.append(
                    "Sentinel-2 multispectral imagery:\n"
                    "- 13 spectral bands from 443nm (Blue) to 2190nm (SWIR2)\n"
                    "- Supports vegetation indices (NDVI, EVI), water/snow detection\n"
                    "- 10m resolution for RGB+NIR, 20m for Red-Edge+SWIR, 60m for coastal/water vapor"
                )

    # Change detection evidence
    if change_result is not None:
        change_lines = [
            f"Change detection algorithm: OpenCV absolute difference + adaptive thresholding",
            f"Change coverage: {change_result.change_pct:.1f}% of image area",
            f"Change score (0–1): {change_result.change_score:.3f}",
            f"Number of changed regions: {change_result.num_changed_regions}",
            f"Summary: {change_result.changed_area_summary}",
            f"Evidence strength: {change_result.evidence_strength}",
        ]
        if change_result.notes:
            change_lines.extend([f"Note: {n}" for n in change_result.notes])
        bundle.change_detection = "\n".join(change_lines)

    return bundle
