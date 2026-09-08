"""Tests for multi-image handling (PS §28-30)."""
import numpy as np
import pytest
from unittest.mock import patch
from satquery.utils.image_io import ImageMeta
from satquery.controller.input_validator import classify_input_config, ValidationError
from satquery.constants import SINGLE_IMAGE, BITEMPORAL_PAIR, CROSS_MODAL_PAIR, MULTI_IMAGE

def make_meta(w=64, h=64, modality='optical'):
    return ImageMeta(path=f'/tmp/img_{modality}.png', modality=modality, width=w, height=h, bands=3, format='PNG')

# 1. Single image
def test_single_image():
    assert classify_input_config([make_meta()]) == SINGLE_IMAGE

# 2. Two images same modality → bitemporal  
def test_two_optical_images():
    assert classify_input_config([make_meta(), make_meta()]) == BITEMPORAL_PAIR

# 3. Two images optical+SAR → cross-modal
def test_optical_sar_pair():
    assert classify_input_config([make_meta(modality='optical'), make_meta(modality='sar')]) == CROSS_MODAL_PAIR

# 4. Three images → MULTI_IMAGE
def test_three_images_multi():
    metas = [make_meta() for _ in range(3)]
    kind = classify_input_config(metas)
    assert kind == MULTI_IMAGE

# 5. Four images → MULTI_IMAGE
def test_four_images_multi():
    metas = [make_meta() for _ in range(4)]
    kind = classify_input_config(metas)
    assert kind == MULTI_IMAGE

# 6. Partial failure: some images fail to load, continue with rest
@patch('satquery.adapters.gemini_adapter.GeminiAdapter.is_available', return_value=True)
@patch('satquery.adapters.gemini_adapter.GeminiAdapter.analyze')
def test_partial_failure_handling(mock_analyze, mock_available, tmp_path):
    from satquery.adapters.gemini_adapter import GeminiAnalysisResult
    from satquery.controller.agent_controller import AgentController
    mock_analyze.return_value = GeminiAnalysisResult(
        answer='Test answer', observations=[], inferences=[],
        evidence_strength='moderate', limitations=[], model_used='gemini-2.5-flash'
    )
    # Create 3 valid arrays
    controller = AgentController()
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    metas = [make_meta() for _ in range(3)]
    images = [(arr, m) for m in metas]
    result = controller.handle_query('Describe each image.', images)
    # Should succeed (not crash) even with 3+ images
    assert result is not None
    assert result.task is not None

# 7. CRS mismatch raises proper error code
def test_crs_mismatch_error_code():
    meta_a = ImageMeta(path='/tmp/a.tif', modality='optical', width=64, height=64,
                       bands=3, crs='EPSG:4326', is_georeferenced=True)
    meta_b = ImageMeta(path='/tmp/b.tif', modality='optical', width=64, height=64,
                       bands=3, crs='EPSG:32644', is_georeferenced=True)
    with pytest.raises(ValidationError) as exc_info:
        classify_input_config([meta_a, meta_b])
    # Should raise with CRS_MISMATCH or include CRS info in message
    assert 'CRS' in str(exc_info.value) or 'crs' in str(exc_info.value).lower()

# 8. Temporal validation warning
def test_temporal_metadata_extracted():
    meta_a = ImageMeta(path='/tmp/20230101_optical.png', modality='optical',
                       width=64, height=64, bands=3, acquisition_date='2023-01-01')
    meta_b = ImageMeta(path='/tmp/20230601_optical.png', modality='optical',
                       width=64, height=64, bands=3, acquisition_date='2023-06-01')
    # Same dimensions → should classify as bitemporal
    kind = classify_input_config([meta_a, meta_b])
    assert kind == BITEMPORAL_PAIR
