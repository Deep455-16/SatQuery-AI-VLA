"""
tests/test_gemini_adapter.py

Tests for the Gemini adapter — all external calls are mocked.
No production credentials required.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from satquery.adapters.gemini_adapter import (
    GeminiAdapter,
    GeminiUnavailableError,
    GeminiRateLimitError,
    GeminiAnalysisResult,
)
from PIL import Image


def _make_image():
    return Image.new("RGB", (64, 64), color=(100, 150, 200))


# ─── Constructor tests ────────────────────────────────────────────────────────

def test_adapter_no_key():
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        adapter = GeminiAdapter()
        assert not adapter.is_available()


def test_adapter_with_key_but_client_fails():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel", side_effect=Exception("init fail")):
            adapter = GeminiAdapter()
            assert not adapter.is_available()


def test_check_available_no_key():
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        adapter = GeminiAdapter()
        with pytest.raises(GeminiUnavailableError, match="GEMINI_API_KEY"):
            adapter.check_available()


# ─── Analysis tests ───────────────────────────────────────────────────────────

@patch("google.generativeai.configure")
@patch("google.generativeai.GenerativeModel")
def test_analyze_success(mock_model_cls, mock_configure):
    mock_response = MagicMock()
    mock_response.text = (
        "ANSWER:\nThe image shows agricultural land.\n\n"
        "OBSERVATIONS:\n- Green rectangular patches visible.\n\n"
        "INFERENCES:\n- Possibly rice paddies.\n\n"
        "EVIDENCE_STRENGTH: moderate\n\n"
        "LIMITATIONS:\n- Crop type cannot be confirmed."
    )
    mock_instance = MagicMock()
    mock_instance.generate_content.return_value = mock_response
    mock_model_cls.return_value = mock_instance

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-2.5-flash"}):
        adapter = GeminiAdapter()
        adapter._client = mock_instance
        result = adapter.analyze(
            query="What is this?",
            images=[_make_image()],
            evidence_context="NDVI: 0.45",
        )

    assert result.answer == "The image shows agricultural land."
    assert "Green rectangular patches visible." in result.observations
    assert result.evidence_strength == "moderate"
    assert result.limitations


@patch("google.generativeai.configure")
@patch("google.generativeai.GenerativeModel")
def test_analyze_rate_limit(mock_model_cls, mock_configure):
    mock_instance = MagicMock()
    mock_instance.generate_content.side_effect = Exception("429 quota exceeded")
    mock_model_cls.return_value = mock_instance

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        adapter = GeminiAdapter()
        adapter._client = mock_instance
        with pytest.raises(GeminiRateLimitError):
            adapter.analyze(query="test", images=[_make_image()])


@patch("google.generativeai.configure")
@patch("google.generativeai.GenerativeModel")
def test_analyze_invalid_key(mock_model_cls, mock_configure):
    mock_instance = MagicMock()
    mock_instance.generate_content.side_effect = Exception("403 api_key invalid")
    mock_model_cls.return_value = mock_instance

    with patch.dict(os.environ, {"GEMINI_API_KEY": "bad-key"}):
        adapter = GeminiAdapter()
        adapter._client = mock_instance
        with pytest.raises(GeminiUnavailableError):
            adapter.analyze(query="test", images=[_make_image()])


# ─── Response parsing tests ───────────────────────────────────────────────────

def test_parse_well_formed_response():
    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter._model_name = "gemini-2.5-flash"

    raw = (
        "ANSWER:\nUrban settlement visible.\n\n"
        "OBSERVATIONS:\n- Dense building pattern.\n- Road network visible.\n\n"
        "INFERENCES:\n- Likely residential area.\n\n"
        "EVIDENCE_STRENGTH: high\n\n"
        "LIMITATIONS:\n- Population density cannot be estimated."
    )
    result = adapter._parse_response(raw)
    assert result.answer == "Urban settlement visible."
    assert len(result.observations) == 2
    assert result.evidence_strength == "high"
    assert result.limitations


def test_parse_unstructured_response():
    """If model doesn't follow format, use full text as answer."""
    adapter = GeminiAdapter.__new__(GeminiAdapter)
    adapter._model_name = "gemini-2.5-flash"

    raw = "The image shows a river delta."
    result = adapter._parse_response(raw)
    assert "river delta" in result.answer
