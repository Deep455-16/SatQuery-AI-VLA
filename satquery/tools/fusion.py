"""
satquery/tools/fusion.py

Optical + SAR cross-modal fusion tool.
"""
from __future__ import annotations

from satquery.utils.image_io import ImageMeta
from satquery.model.builder import get_model

def _format_optical_evidence(image, meta: ImageMeta = None) -> str:
    if not meta:
        return "Optical image evidence: no metadata provided."
    return f"Optical evidence: acquired {getattr(meta, 'acquisition_date', 'unknown')}, resolution {getattr(meta, 'resolution_x', 'unknown')}m, sensor {getattr(meta, 'sensor', 'unknown')}"

def _format_sar_evidence(image, meta: ImageMeta = None) -> str:
    if not meta:
        return "SAR image evidence: no metadata provided."
    return f"SAR evidence: acquired {getattr(meta, 'acquisition_date', 'unknown')}, resolution {getattr(meta, 'resolution_x', 'unknown')}m, sensor {getattr(meta, 'sensor', 'unknown')}"

def run_fusion_analysis(optical_image, sar_image, query, optical_meta=None, sar_meta=None) -> dict:
    """
    Returns:
    {
        'task': 'OPTICAL_SAR_FUSION',
        'tool': str,
        'model': str,
        'optical_evidence': str,  # what optical shows
        'sar_evidence': str,      # what SAR shows
        'joint_analysis_note': str,  # guidance for Gemini
        'specialist_result': dict or None,
    }
    """
    # Format optical evidence from meta
    optical_ev = _format_optical_evidence(optical_image, optical_meta)
    sar_ev = _format_sar_evidence(sar_image, sar_meta)
    
    joint_note = (
        'Optical analysis: Use color, texture, and spectral reflectance to identify '
        'vegetation, water, and urban land cover. '
        'SAR analysis: Use backscatter intensity and texture to identify structural '
        'features, surface roughness, and moisture. '
        'Joint: Synthesize both modalities — SAR confirms structure where optical is ambiguous.'
    )
    
    specialist = get_model("fusion")
    if specialist and specialist.available:
        try:
            result = specialist.run(
                optical_image=optical_image,
                sar_image=sar_image,
                query=query,
                task="fusion",
                optical_meta=optical_meta,
                sar_meta=sar_meta,
            )
            if result is not None:
                return {
                    'task': 'OPTICAL_SAR_FUSION',
                    'tool': f'Specialist Fusion Model ({specialist.name})',
                    'model': specialist.name,
                    'optical_evidence': optical_ev,
                    'sar_evidence': sar_ev,
                    'joint_analysis_note': joint_note,
                    'specialist_result': result,
                }
        except Exception:
            pass

    return {
        'task': 'OPTICAL_SAR_FUSION',
        'tool': 'Gemini 2.5 Flash (cross-modal reasoning)',
        'model': 'Gemini 2.5 Flash',
        'optical_evidence': optical_ev,
        'sar_evidence': sar_ev,
        'joint_analysis_note': joint_note,
        'specialist_result': None,
    }
