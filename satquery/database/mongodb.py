"""
satquery/database/mongodb.py

(MongoDB was removed. This module now serves as a local JSON file-based history storage 
to provide full history functionality without requiring a database service.)
"""
import json
import os
import time
from uuid import uuid4
from typing import List, Dict, Any

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'history.json')

def _load_history() -> List[Dict[str, Any]]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def _json_fallback(obj):
    import dataclasses
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    return str(obj)

def _save_history(data: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=_json_fallback)

def get_database():
    return None

def is_connected() -> bool:
    return True  # Local file is always available

def save_image_metadata(*args, **kwargs):
    return None

def save_analysis(session_id: str, query: str, result_dict: dict, images: list = None) -> str:
    """Save an analysis result to the local JSON history file."""
    record = {
        "id": str(uuid4()),
        "session_id": session_id,
        "timestamp": time.time(),
        "query": query,
        "result": result_dict,
        "images": images or []
    }
    
    history = _load_history()
    history.append(record)
    # Keep last 100 queries max
    if len(history) > 100:
        history = history[-100:]
        
    _save_history(history)
    return record["id"]

def save_model_run(*args, **kwargs):
    return None

def save_feedback(*args, **kwargs):
    return None

def get_analysis_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve analysis history from the local JSON file."""
    history = _load_history()
    # Sort newest first
    history.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return history[:limit]

# Aliases for api_app.py
def get_session_history(*args, **kwargs):
    return get_analysis_history()
