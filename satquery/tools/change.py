"""
satquery/tools/change.py

Bi-temporal change detection tool.
"""
from __future__ import annotations

from satquery.model.builder import get_model
from satquery.utils.image_preprocessor import generate_change_map

_CHANGE_VQA_KEYWORDS = ['has', 'increased', 'decreased', 'grown', 'shrunk', 'more', 'less', 'still', 'remained', 'unchanged', 'same']

def _is_change_vqa(query: str) -> bool:
    return any(kw in query.lower() for kw in _CHANGE_VQA_KEYWORDS) if query else False

def run_change_analysis(image_a, image_b, meta_a=None, meta_b=None, query=None) -> dict:
    """
    Returns:
    {
        'task': 'CHANGE_VQA' or 'CHANGE_DETECTION',
        'tool': str,
        'model': str,
        'change_result': ChangeDetectionResult or None,
        'change_map_pil': PIL.Image or None,
        'dates': {'before': str or None, 'after': str or None},
        'evidence': {
            'change_pct': float,
            'change_score': float,
            'num_regions': int,
            'evidence_strength': str,
            'summary': str,
        } or None,
        'error': str or None,
    }
    """
    task = 'CHANGE_VQA' if _is_change_vqa(query) else 'CHANGE_DETECTION'
    
    date_a = getattr(meta_a, 'acquisition_date', None) if meta_a else None
    date_b = getattr(meta_b, 'acquisition_date', None) if meta_b else None
    
    try:
        dummy_meta_a = meta_a or _dummy_meta()
        dummy_meta_b = meta_b or _dummy_meta()
        
        change_result = generate_change_map(image_a, image_b, dummy_meta_a, dummy_meta_b)
        
        from PIL import Image
        change_map_pil = None
        if change_result and change_result.change_map is not None:
            import numpy as np
            cm_img = (change_result.change_map * 255).astype(np.uint8)
            change_map_pil = Image.fromarray(cm_img, mode='L')
            
        return {
            'task': task,
            'tool': 'satquery.tools.change.run_change_analysis (opencv)',
            'model': 'OpenCV change detection + Gemini 2.5 Flash',
            'change_result': change_result,
            'change_map_pil': change_map_pil,
            'dates': {'before': date_a, 'after': date_b},
            'evidence': {
                'optical_before': f'Date: {date_a}' if date_a else '',
                'optical_after': f'Date: {date_b}' if date_b else '',
                'change_stats': {
                    'change_pct': float(change_result.change_fraction * 100.0) if change_result else 0.0,
                    'change_score': float(change_result.confidence_score) if change_result else 0.0,
                    'num_regions': int(len(change_result.regions)) if change_result else 0,
                    'evidence_strength': 'High' if change_result and change_result.confidence_score > 0.7 else 'Medium',
                    'summary': f"{change_result.change_fraction * 100.0:.1f}% area changed" if change_result else "No significant change",
                }
            },
            'error': None,
        }
    except Exception as e:
         return {
            'task': task,
            'tool': 'satquery.tools.change.run_change_analysis (failed)',
            'model': 'None',
            'change_result': None,
            'change_map_pil': None,
            'dates': {'before': date_a, 'after': date_b},
            'evidence': None,
            'error': str(e),
        }

def _dummy_meta():
    """Return a minimal ImageMeta-like object for when meta is not available."""
    from satquery.utils.image_io import ImageMeta
    return ImageMeta(
        path="",
        modality="optical",
        width=0,
        height=0,
        bands=3,
    )
