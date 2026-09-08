"""Tests for the formal task classifier (PS §24)."""
import pytest
from satquery.controller.agent_controller import classify_task
from satquery.controller.input_validator import validate_query, ValidationError
from satquery.constants import SINGLE_IMAGE, BITEMPORAL_PAIR, CROSS_MODAL_PAIR, MULTI_IMAGE

# Test all 8 formal task categories
def test_vqa_classification():
    task = classify_task('What is the main feature in this image?', SINGLE_IMAGE)
    assert task in ('VQA', 'CAPTION')

def test_caption_classification():
    task = classify_task('Describe the scene.', SINGLE_IMAGE)
    assert task in ('VQA', 'CAPTION')

def test_grounding_classification():
    task = classify_task('Highlight the stadium.', SINGLE_IMAGE)
    assert task == 'GROUNDING'

def test_change_detection_from_bitemporal():
    task = classify_task('What changed between the two dates?', BITEMPORAL_PAIR)
    assert task == 'CHANGE_DETECTION'

def test_change_vqa_from_bitemporal():
    task = classify_task('Did the water level decrease?', BITEMPORAL_PAIR)
    assert task == 'CHANGE_VQA'

def test_optical_sar_from_cross_modal():
    task = classify_task('Fuse the optical and SAR data.', CROSS_MODAL_PAIR)
    assert task == 'OPTICAL_SAR_FUSION'

def test_comparison_multi_image():
    task = classify_task('Compare these images.', MULTI_IMAGE)
    assert isinstance(task, str)

def test_unsupported_query():
    task = classify_task('Can you write a poem about space?', SINGLE_IMAGE)
    assert isinstance(task, str)

# Test the 5 representative queries from PS §46
def test_representative_query_1_caption():
    """'Describe the land-cover and major objects visible in this image.'"""
    task = classify_task('Describe the land-cover and major objects visible in this image.', SINGLE_IMAGE)
    assert task in ('CAPTION', 'VQA')

def test_representative_query_2_grounding():
    """'Highlight the water body referred to in the query.'"""
    task = classify_task('Highlight the water body referred to in the query.', SINGLE_IMAGE)
    assert task == 'GROUNDING'

def test_representative_query_3_change_detection():
    """'What changed between these two dates, and where did the change occur?'"""
    task = classify_task('What changed between these two dates, and where did the change occur?', BITEMPORAL_PAIR)
    assert task == 'CHANGE_DETECTION'

def test_representative_query_4_optical_sar():
    """'Use the optical and SAR images together to identify built-up and water-covered regions.'"""
    task = classify_task('Use the optical and SAR images together to identify built-up and water-covered regions.', CROSS_MODAL_PAIR)
    assert task == 'OPTICAL_SAR_FUSION'

def test_representative_query_5_change_vqa():
    """'Has the built-up area increased, decreased, or remained unchanged?'"""
    task = classify_task('Has the built-up area increased, decreased, or remained unchanged?', BITEMPORAL_PAIR)
    assert task == 'CHANGE_VQA'

def test_validate_empty_query():
    with pytest.raises(ValidationError) as exc:
        validate_query('')
    assert 'empty' in str(exc.value).lower() or hasattr(exc.value, 'code')

def test_validate_valid_query():
    assert validate_query('valid question') is None
