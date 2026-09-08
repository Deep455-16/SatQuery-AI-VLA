"""satquery/database/__init__.py — MongoDB removed. All stubs."""
from satquery.database.mongodb import (
    get_database,
    is_connected,
    save_image_metadata,
    save_analysis,
    save_model_run,
    save_feedback,
    get_analysis_history,
)

def create_session(*a, **kw): return None
def get_session_history(*a, **kw): return []
def delete_analysis(*a, **kw): return None

__all__ = [
    "get_database", "is_connected", "create_session",
    "save_image_metadata", "save_analysis", "save_model_run",
    "save_feedback", "get_analysis_history",
    "get_session_history", "delete_analysis",
]
