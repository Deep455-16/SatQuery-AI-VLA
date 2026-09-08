"""
VQA and captioning tool for SatQuery AI.
Primary: TerraQ-VL; Fallback: Gemini 2.5 Flash with evidence.
"""

def run_vqa(arr, query: str, meta=None) -> dict:
    """
    Returns:
    {
        'task': 'VQA',
        'tool': 'TerraQ-VL VQA' or 'Gemini 2.5 Flash (fallback)',
        'model': str,
        'answer': str or None,  # None means agent_controller should call Gemini
        'confidence': float or None,
        'terraq_available': bool,
        'terraq_status': str,  # READY/BLOCKED/UNAVAILABLE/etc.
    }
    """
    # Try TerraQ-VL first
    try:
        from satquery.model.terraq_vl import get_terraq_vl
        model = get_terraq_vl()
        if model.is_available():
            from PIL import Image
            import numpy as np
            # Convert array to PIL for TerraQ-VL
            if hasattr(arr, 'shape'):
                if arr.ndim == 2:
                    img = Image.fromarray(arr.astype(np.uint8), mode='L').convert('RGB')
                elif arr.shape[-1] == 1:
                    img = Image.fromarray(arr[:,:,0].astype(np.uint8), mode='L').convert('RGB')
                else:
                    img = Image.fromarray(arr[:,:,:3].astype(np.uint8), mode='RGB')
            else:
                img = arr
            result = model.run_vqa(img, query)
            return {
                'task': 'VQA',
                'tool': 'TerraQ-VL VQA',
                'model': result.get('model', 'TerraQ-VL'),
                'answer': result.get('answer'),
                'confidence': result.get('confidence'),
                'terraq_available': True,
                'terraq_status': 'READY',
            }
    except Exception:
        pass  # Fall through to Gemini
    
    # Gemini fallback
    return {
        'task': 'VQA',
        'tool': 'Gemini 2.5 Flash (TerraQ-VL unavailable — see VLM_MODEL_PATH)',
        'model': 'Gemini 2.5 Flash',
        'answer': None,  # agent_controller calls Gemini directly
        'confidence': None,
        'terraq_available': False,
        'terraq_status': _get_terraq_status(),
    }

def run_caption(arr, meta=None) -> dict:
    # Same pattern as run_vqa but uses model.run_caption()
    # Query for caption: 'Describe the land-cover and major features visible in this satellite image.'
    query = 'Describe the land-cover and major features visible in this satellite image.'
    try:
        from satquery.model.terraq_vl import get_terraq_vl
        model = get_terraq_vl()
        if model.is_available():
            from PIL import Image
            import numpy as np
            # Convert array to PIL for TerraQ-VL
            if hasattr(arr, 'shape'):
                if arr.ndim == 2:
                    img = Image.fromarray(arr.astype(np.uint8), mode='L').convert('RGB')
                elif arr.shape[-1] == 1:
                    img = Image.fromarray(arr[:,:,0].astype(np.uint8), mode='L').convert('RGB')
                else:
                    img = Image.fromarray(arr[:,:,:3].astype(np.uint8), mode='RGB')
            else:
                img = arr
            result = model.run_caption(img, query)
            return {
                'task': 'CAPTION',
                'tool': 'TerraQ-VL Caption',
                'model': result.get('model', 'TerraQ-VL'),
                'answer': result.get('answer'),
                'confidence': result.get('confidence'),
                'terraq_available': True,
                'terraq_status': 'READY',
            }
    except Exception:
        pass  # Fall through to Gemini
    
    # Gemini fallback
    return {
        'task': 'CAPTION',
        'tool': 'Gemini 2.5 Flash (TerraQ-VL unavailable — see VLM_MODEL_PATH)',
        'model': 'Gemini 2.5 Flash',
        'answer': None,  # agent_controller calls Gemini directly
        'confidence': None,
        'terraq_available': False,
        'terraq_status': _get_terraq_status(),
    }

def _get_terraq_status() -> str:
    try:
        from satquery.model.terraq_vl import get_terraq_vl
        return get_terraq_vl().status
    except Exception:
        return 'UNAVAILABLE'
