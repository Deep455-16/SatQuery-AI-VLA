"""
tests/test_pipeline.py

Core pipeline tests for SatQuery AI.
Tests run WITHOUT production credentials (all external calls are mocked).

Run with:
    python -m pytest tests/ -v --tb=short
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from satquery.controller.agent_controller import AgentController, classify_intent
from satquery.controller.input_validator import classify_input_config, ValidationError
from satquery.utils.image_io import load_image, ImageMeta
from satquery.utils.image_preprocessor import (
    generate_rgb_visualization,
    generate_change_map,
    normalize_bands,
    extract_image_statistics,
    resize_for_llm,
)
from satquery.constants import SINGLE_IMAGE, CROSS_MODAL_PAIR, BITEMPORAL_PAIR


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_png(path: str, size: tuple = (64, 64), mode: str = "RGB"):
    arr = (np.random.rand(*size, 3) * 255).astype("uint8") if mode == "RGB" else \
          (np.random.rand(*size) * 255).astype("uint8")
    if mode == "RGB":
        Image.fromarray(arr, mode="RGB").save(path)
    else:
        Image.fromarray(arr, mode="L").save(path)


def make_optical_meta(w=64, h=64):
    return ImageMeta(
        path="/tmp/optical.png",
        modality="optical",
        width=w, height=h, bands=3,
        format="PNG",
        band_names=["Red", "Green", "Blue"],
    )


def make_sar_meta(w=64, h=64):
    return ImageMeta(
        path="/tmp/sar.png",
        modality="sar",
        width=w, height=h, bands=1,
        format="PNG",
        band_names=["VV"],
        is_sar=True,
    )


# ─── Image loading tests ──────────────────────────────────────────────────────

def test_load_png_rgb(tmp_path):
    p = str(tmp_path / "scene.png")
    make_png(p, size=(64, 64))
    arr, meta = load_image(p)
    assert arr.shape == (64, 64, 3)
    assert meta.width == 64
    assert meta.height == 64
    assert meta.bands == 3
    assert meta.format == "PNG"


def test_load_grayscale_png(tmp_path):
    p = str(tmp_path / "gray.png")
    make_png(p, size=(32, 32), mode="L")
    arr, meta = load_image(p)
    assert meta.bands == 1


def test_unsupported_format(tmp_path):
    p = str(tmp_path / "scene.bmp")
    Image.new("RGB", (32, 32)).save(p)
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_image(p)


def test_meta_display_dict():
    meta = make_optical_meta()
    d = meta.to_display_dict()
    assert "Format" in d
    assert "Modality" in d
    assert "Optical" in d["Modality"]


def test_meta_evidence_string():
    meta = make_optical_meta()
    es = meta.to_evidence_string()
    assert "optical" in es
    assert "64" in es


# ─── Preprocessing tests ──────────────────────────────────────────────────────

def test_generate_rgb_visualization():
    arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    meta = make_optical_meta()
    img, desc = generate_rgb_visualization(arr, meta)
    assert isinstance(img, Image.Image)
    assert img.size[0] > 0
    assert isinstance(desc, str)


def test_generate_rgb_sar():
    arr = (np.random.rand(64, 64, 1) * 1000).astype("float32")
    meta = make_sar_meta()
    img, desc = generate_rgb_visualization(arr, meta)
    assert isinstance(img, Image.Image)
    assert "SAR" in desc or "sar" in desc.lower()


def test_normalize_bands():
    arr = np.random.rand(32, 32, 3).astype("float32") * 10000
    norm = normalize_bands(arr, is_sar=False)
    assert norm.dtype == np.uint8
    assert norm.max() <= 255


def test_resize_for_llm():
    img = Image.new("RGB", (2000, 1500))
    resized = resize_for_llm(img, max_size=1024)
    assert max(resized.size) <= 1024


def test_resize_for_llm_small():
    img = Image.new("RGB", (100, 100))
    resized = resize_for_llm(img, max_size=1024)
    assert resized.size == (100, 100)  # no change needed


def test_extract_image_statistics():
    arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    meta = make_optical_meta()
    stats = extract_image_statistics(arr, meta)
    assert "image_size_pixels" in stats
    assert "band_statistics" in stats
    assert len(stats["band_statistics"]) == 3


# ─── Change detection tests ───────────────────────────────────────────────────

def test_change_map_identical():
    arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    meta = make_optical_meta()
    result = generate_change_map(arr, arr, meta, meta)
    assert result.change_pct < 5.0  # identical images → minimal change


def test_change_map_very_different():
    arr_a = np.zeros((64, 64, 3), dtype="uint8")
    arr_b = np.ones((64, 64, 3), dtype="uint8") * 255
    meta = make_optical_meta()
    result = generate_change_map(arr_a, arr_b, meta, meta)
    assert result.change_pct > 50.0  # max contrast → large change
    assert result.change_map_pil is not None


def test_change_map_different_sizes():
    arr_a = np.zeros((64, 64, 3), dtype="uint8")
    arr_b = np.zeros((32, 32, 3), dtype="uint8")
    meta_a = make_optical_meta(64, 64)
    meta_b = make_optical_meta(32, 32)
    result = generate_change_map(arr_a, arr_b, meta_a, meta_b)
    # Should handle resize gracefully
    assert result is not None
    assert "resize" in " ".join(result.notes).lower() or result.change_pct >= 0


# ─── Input validation tests ───────────────────────────────────────────────────

def test_single_image_classification():
    meta = make_optical_meta()
    kind = classify_input_config([meta])
    assert kind == SINGLE_IMAGE


def test_bitemporal_classification():
    meta_a = make_optical_meta()
    meta_b = make_optical_meta()
    kind = classify_input_config([meta_a, meta_b])
    assert kind == BITEMPORAL_PAIR


def test_cross_modal_classification():
    meta_opt = make_optical_meta()
    meta_sar = make_sar_meta()
    kind = classify_input_config([meta_opt, meta_sar])
    assert kind == CROSS_MODAL_PAIR


def test_dimension_mismatch_raises():
    meta_a = make_optical_meta(64, 64)
    meta_b = make_optical_meta(32, 32)
    with pytest.raises(ValidationError, match="Dimension mismatch"):
        classify_input_config([meta_a, meta_b])


def test_too_many_images():
    """3+ images now return MULTI_IMAGE instead of raising — PS §28 compliance."""
    from satquery.constants import MULTI_IMAGE
    meta = make_optical_meta()
    result = classify_input_config([meta, meta, meta])
    assert result == MULTI_IMAGE


# ─── Intent classification tests (updated for PS 26167 formal task tokens) ────

def test_intent_change_detection():
    # Bitemporal + non-VQA query → CHANGE_DETECTION
    assert classify_intent("Where is the change?", BITEMPORAL_PAIR) == "CHANGE_DETECTION"


def test_intent_optical_sar():
    # Cross-modal always → OPTICAL_SAR_FUSION
    assert classify_intent("anything", CROSS_MODAL_PAIR) == "OPTICAL_SAR_FUSION"


def test_intent_water():
    # Water detection query → VQA (no grounding trigger)
    intent = classify_intent("Is there a river visible?", SINGLE_IMAGE)
    assert intent in ("VQA", "CAPTION")  # PS formal tokens


def test_intent_vegetation():
    # Description query → CAPTION
    intent = classify_intent("Describe the vegetation and forest areas.", SINGLE_IMAGE)
    assert intent in ("CAPTION", "VQA")


def test_intent_description():
    # Describe → CAPTION
    intent = classify_intent("Describe this image.", SINGLE_IMAGE)
    assert intent == "CAPTION"


def test_intent_fallback():
    # Unknown query → VQA
    intent = classify_intent("xyzzy plugh", SINGLE_IMAGE)
    assert intent == "VQA"


# ─── Agent controller tests (mocked Gemini) ───────────────────────────────────

def _mock_gemini_result():
    from satquery.adapters.gemini_adapter import GeminiAnalysisResult
    return GeminiAnalysisResult(
        answer="The image appears to show agricultural land.",
        observations=["Green rectangular patches visible in the center."],
        inferences=["These patterns are consistent with agricultural fields."],
        evidence_strength="moderate",
        limitations=["Crop type cannot be determined without spectral analysis."],
        model_used="gemini-2.5-flash",
    )


@patch("satquery.adapters.gemini_adapter.GeminiAdapter.is_available", return_value=True)
@patch("satquery.adapters.gemini_adapter.GeminiAdapter.analyze")
def test_single_image_vqa(mock_analyze, mock_available, tmp_path):
    mock_analyze.return_value = _mock_gemini_result()
    p = str(tmp_path / "scene.png")
    make_png(p)
    arr, meta = load_image(p)

    controller = AgentController()
    result = controller.handle_query("Are there agricultural fields?", [(arr, meta)])

    assert result.answer == "The image appears to show agricultural land."
    assert result.observations
    assert result.evidence_strength == "moderate"
    assert result.processing_time_ms >= 0


@patch("satquery.adapters.gemini_adapter.GeminiAdapter.is_available", return_value=True)
@patch("satquery.adapters.gemini_adapter.GeminiAdapter.analyze")
def test_bitemporal_query(mock_analyze, mock_available, tmp_path):
    mock_analyze.return_value = _mock_gemini_result()
    p1 = str(tmp_path / "before_optical.png")
    p2 = str(tmp_path / "after_optical.png")
    make_png(p1)
    make_png(p2)
    arr1, meta1 = load_image(p1)
    arr2, meta2 = load_image(p2)

    controller = AgentController()
    result = controller.handle_query("What changed?", [(arr1, meta1), (arr2, meta2)])

    assert result.task in ("CHANGE_VQA", "CHANGE_DETECTION", "change_detection")
    assert result.answer


@patch("satquery.adapters.gemini_adapter.GeminiAdapter.is_available", return_value=False)
def test_missing_api_key(mock_available, tmp_path):
    p = str(tmp_path / "scene.png")
    make_png(p)
    arr, meta = load_image(p)

    controller = AgentController()
    result = controller.handle_query("What is this?", [(arr, meta)])

    # Should return limitations, not crash
    assert not result.answer
    assert result.limitations
    assert any("GEMINI_API_KEY" in lim or "api" in lim.lower() for lim in result.limitations)


def test_dimension_mismatch_rejected(tmp_path):
    p1 = str(tmp_path / "a_optical.png")
    p2 = str(tmp_path / "b_optical.png")
    make_png(p1, size=(64, 64))
    make_png(p2, size=(32, 32))
    arr1, meta1 = load_image(p1)
    arr2, meta2 = load_image(p2)

    controller = AgentController()
    result = controller.handle_query("What changed?", [(arr1, meta1), (arr2, meta2)])
    assert result.task == "input_validation_failed"


# ─── PDF generation test ──────────────────────────────────────────────────────

def test_pdf_generation_no_crash(tmp_path):
    from satquery.utils.report_builder import build_pdf_report

    result_dict = {
        "task": "general_description",
        "answer": "The image shows agricultural land.",
        "observations": ["Green fields visible."],
        "inferences": ["Possibly wheat or rice crops."],
        "evidence_strength": "moderate",
        "limitations": ["Crop type uncertain."],
        "tools_used": ["gemini"],
        "models_used": ["gemini-2.5-flash"],
        "processing_time_ms": 1200,
        "parameters": {"input_type": "single"},
        "warnings": [],
    }
    pdf_path = str(tmp_path / "test_report.pdf")
    out = build_pdf_report("What is this image?", result_dict, pdf_path)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000  # non-trivial PDF


def test_pdf_no_dir_crash():
    """Original bug: dirname('') crashed makedirs. Verify it no longer does."""
    from satquery.utils.report_builder import build_pdf_report
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        pdf_path = tf.name

    result_dict = {
        "task": "test",
        "answer": "Test answer.",
        "observations": [],
        "inferences": [],
        "evidence_strength": "limited",
        "limitations": [],
        "tools_used": [],
        "models_used": [],
        "processing_time_ms": 0,
        "parameters": {},
        "warnings": [],
    }
    try:
        build_pdf_report("Test query", result_dict, pdf_path)
        assert os.path.exists(pdf_path)
    finally:
        try:
            os.unlink(pdf_path)
        except Exception:
            pass


# ─── Markdown report test ─────────────────────────────────────────────────────

def test_markdown_report():
    from satquery.utils.report_builder import build_markdown_report

    result_dict = {
        "task": "vegetation_analysis",
        "answer": "Dense vegetation visible.",
        "observations": ["Dark green areas present."],
        "inferences": ["May indicate forest cover."],
        "evidence_strength": "high",
        "limitations": [],
        "tools_used": ["gemini"],
        "models_used": ["gemini-2.5-flash"],
        "processing_time_ms": 900,
        "parameters": {},
        "warnings": [],
    }
    md = build_markdown_report("Describe vegetation.", result_dict)
    assert "SatQuery AI" in md
    assert "Dense vegetation" in md
    assert "Observations" in md


# ─── Formal task classification tests (PS §46 representative queries) ──────────

def test_representative_query_vqa():
    """PS §46 Query 1: Captioning/VQA route."""
    from satquery.controller.agent_controller import classify_task
    task = classify_task(
        'Describe the land-cover and major objects visible in this image.',
        SINGLE_IMAGE
    )
    assert task in ('CAPTION', 'VQA')

def test_representative_query_grounding():
    """PS §46 Query 2: Grounding route."""
    from satquery.controller.agent_controller import classify_task
    task = classify_task('Highlight the water body referred to in the query.', SINGLE_IMAGE)
    assert task == 'GROUNDING'

def test_representative_query_change_detection():
    """PS §46 Query 3: Change detection route."""
    from satquery.controller.agent_controller import classify_task
    task = classify_task(
        'What changed between these two dates, and where did the change occur?',
        BITEMPORAL_PAIR
    )
    assert task == 'CHANGE_DETECTION'

def test_representative_query_optical_sar():
    """PS §46 Query 4: Optical-SAR fusion route."""
    from satquery.controller.agent_controller import classify_task
    task = classify_task(
        'Use the optical and SAR images together to identify built-up and water-covered regions.',
        CROSS_MODAL_PAIR
    )
    assert task == 'OPTICAL_SAR_FUSION'

def test_representative_query_change_vqa():
    """PS §46 Query 5: Change VQA route."""
    from satquery.controller.agent_controller import classify_task
    task = classify_task(
        'Has the built-up area increased, decreased, or remained unchanged?',
        BITEMPORAL_PAIR
    )
    assert task == 'CHANGE_VQA'

# ─── Multi-image tests ─────────────────────────────────────────────────────────

def test_three_images_classified_multi(tmp_path):
    from satquery.controller.input_validator import classify_input_config
    from satquery.constants import MULTI_IMAGE
    metas = [make_optical_meta() for _ in range(3)]
    kind = classify_input_config(metas)
    assert kind == MULTI_IMAGE

@patch('satquery.adapters.gemini_adapter.GeminiAdapter.is_available', return_value=True)
@patch('satquery.adapters.gemini_adapter.GeminiAdapter.analyze')
def test_multi_image_controller(mock_analyze, mock_available, tmp_path):
    mock_analyze.return_value = _mock_gemini_result()
    arr = (np.random.rand(64, 64, 3) * 255).astype('uint8')
    meta = make_optical_meta()
    controller = AgentController()
    # 3 images should not crash
    result = controller.handle_query('Describe each image.', [(arr, meta)] * 3)
    assert result is not None
    assert result.answer  # should have an answer

