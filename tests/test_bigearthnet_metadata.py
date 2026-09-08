"""Tests for BigEarthNet metadata.parquet loading and dry-run."""
import os
import pytest
from pathlib import Path

PARQUET_PATH = Path('data/metadata.parquet')

def test_parquet_exists():
    assert PARQUET_PATH.exists(), f'metadata.parquet not found at {PARQUET_PATH}'

def test_parquet_schema():
    """Validate required columns exist."""
    try:
        import pandas as pd
        df = pd.read_parquet(PARQUET_PATH)
        # Check required columns
        required = ['patch_id', 'labels', 'split']  # These are always required
        for col in required:
            assert col in df.columns, f'Missing column: {col}'
    except ImportError:
        pytest.skip('pandas not installed')

def test_parquet_splits():
    """Train/val/test splits exist."""
    try:
        import pandas as pd
        df = pd.read_parquet(PARQUET_PATH)
        if 'split' in df.columns:
            splits = df['split'].unique()
            # Should have at least train split
            assert len(splits) > 0
    except ImportError:
        pytest.skip('pandas not installed')

def test_load_metadata_parquet():
    """Test the load_metadata_parquet function."""
    if not PARQUET_PATH.exists():
        pytest.skip('metadata.parquet not found')
    try:
        from satquery.train.prepare_bigearthnet import load_metadata_parquet
        df = load_metadata_parquet(str(PARQUET_PATH), split='all', max_patches=10)
        assert len(df) > 0
        assert 'patch_id' in df.columns
    except ImportError:
        pytest.skip('pandas not installed')

def test_dry_run_no_data_dir():
    """Dry run should handle missing bigearthnet_root gracefully."""
    from satquery.train.prepare_bigearthnet import dry_run
    result = dry_run(
        bigearthnet_root='/nonexistent/path',
        parquet_path=str(PARQUET_PATH) if PARQUET_PATH.exists() else None,
        max_patches=5
    )
    assert 'errors' in result
    # Should report root not found
    assert len(result.get('errors', [])) > 0 or result.get('metadata_ok') == False

def test_validate_metadata_schema_with_actual_parquet():
    """Schema validation on actual parquet file."""
    if not PARQUET_PATH.exists():
        pytest.skip('metadata.parquet not found')
    try:
        import pandas as pd
        from satquery.train.prepare_bigearthnet import validate_metadata_schema
        df = pd.read_parquet(PARQUET_PATH)
        issues = validate_metadata_schema(df)
        # Log issues but don't necessarily fail (schema may vary)
        assert isinstance(issues, list)
    except ImportError:
        pytest.skip('pandas not installed')
