from .terraq_vl import get_terraq_vl, TerraQVLStatus
from .specialist_registry import get_registry

MODEL_DISPLAY_NAMES = {
    'vqa': 'TerraQ-VL (VQA)',
    'caption': 'TerraQ-VL (Captioning)',
    'grounding': 'TerraQ-VL (Grounding)',
    'change_analysis': 'TerraQ-VL (Change Analysis)',
    'change_vqa': 'TerraQ-VL (Change-VQA)',
    'optical_sar_fusion': 'TerraQ-VL (Optical-SAR Fusion)'
}

class SpecialistModel:
    """Backward-compatible base class for specialist models."""
    def __init__(self, model_name=None):
        self.model_name = model_name

    def load(self):
        pass

    def run(self, *args, **kwargs):
        raise NotImplementedError

_LEGACY_MODELS = {}

def register_model(task_name, model_class):
    """Backward-compatible model registration."""
    _LEGACY_MODELS[task_name] = model_class

def get_model(task: str):
    """
    Returns the TerraQ-VL model if available, else a stub/legacy model.
    """
    vl = get_terraq_vl()
    if vl.is_available():
        return vl
    
    # Fallback to legacy/stub if any
    if task in _LEGACY_MODELS:
        return _LEGACY_MODELS[task]()
        
    # Return a basic stub if no legacy model
    class StubModel(SpecialistModel):
        def run(self, *args, **kwargs):
            return {"error": "TerraQ-VL not available, stub fallback"}
    return StubModel(task)

def get_model_status() -> dict:
    """Returns full status dict including hardware."""
    vl = get_terraq_vl()
    registry = get_registry()
    
    return {
        'terraq_vl': vl.get_status_dict(),
        'registry': registry.get_status()
    }

def load_pretrained_model(model_name: str, task: str):
    """
    Attempts to return the real TerraQ-VL or legacy models.
    """
    vl = get_terraq_vl()
    if vl.is_available():
        return vl
        
    return get_model(task)
