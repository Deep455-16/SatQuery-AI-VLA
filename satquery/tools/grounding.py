"""
satquery/tools/grounding.py

Text-guided region grounding tool.
"""
from __future__ import annotations

from satquery.model.builder import get_model

def run_grounding(arr, query: str, meta=None) -> dict:
    """
    Returns:
    {
        'task': 'GROUNDING',
        'tool': str,
        'model': str,
        'grounding_available': bool,
        'bboxes': list,  # [] if no specialist
        'description': str,  # Natural language location description
        'note': str,  # Explains grounding status
    }
    Never returns fake bboxes.
    """
    # Try specialist first (will be UNAVAILABLE unless checkpoint configured)
    specialist = get_model('grounding')
    if specialist and specialist.available:
        try:
            result = specialist.run(image=arr, query=query, task='grounding')
            if result and 'bboxes' in result:
                return {
                    'task': 'GROUNDING',
                    'tool': f'Specialist Grounding Model ({specialist.name})',
                    'model': specialist.name,
                    'grounding_available': True,
                    'bboxes': result['bboxes'],
                    'labels': result.get('labels', []),
                    'scores': result.get('scores', []),
                    'description': '',
                    'note': 'Specialist grounding model used.',
                }
        except Exception:
            pass
    
    # No specialist: use Gemini for verbal region description
    # The agent_controller will call Gemini with a grounding-specific prompt
    return {
        'task': 'GROUNDING',
        'tool': 'Gemini 2.5 Flash (verbal grounding — no specialist checkpoint)',
        'model': 'Gemini 2.5 Flash',
        'grounding_available': False,
        'bboxes': [],
        'description': None,  # will be filled by Gemini
        'note': (
            'Precise bounding-box grounding requires a specialist checkpoint '
            '(e.g., GeoChat weights). This deployment returns a natural-language '
            'region description instead. Set VLM_MODEL_PATH to enable specialist grounding.'
        ),
    }
