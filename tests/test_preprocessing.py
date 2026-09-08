"""
tests/test_preprocessing.py

Tests for the image preprocessing module.
No external model or API calls required.
"""
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from satquery.utils.image_io import ImageMeta
from satquery.utils.image_preprocessor import (
    _percentile_clip,
    _sar_log_normalize,
    normalize_bands,
    generate_rgb_visualization,
    generate_false_color,
    generate_sar_visualization,
    generate_change_map,
    preprocess_for_llm,
    resize_for_llm,
    extract_image_statistics,
    format_statistics_for_evidence,
    _compute_ndvi,
    LLM_MAX_SIZE,
)


def make_meta(**kwargs):
    defaults = dict(path="/tmp/test.png", modality="optical", width=64,
                    height=64, bands=3, format="PNG", band_names=["Red", "Green", "Blue"])
    defaults.update(kwargs)
    return ImageMeta(**defaults)


# ─── Normalization ────────────────────────────────────────────────────────────

def test_percentile_clip_range():
    arr = np.random.rand(64, 64).astype(np.float32) * 10000
    result = _percentile_clip(arr)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_percentile_clip_uniform():
    arr = np.ones((32, 32), dtype=np.float32) * 500.0
    result = _percentile_clip(arr)
    # All same value → all zeros
    assert result.max() == 0


def test_sar_log_normalize():
    arr = np.random.rand(64, 64).astype(np.float32) * 1000 + 1
    result = _sar_log_normalize(arr)
    assert result.dtype == np.uint8
    assert 0 <= result.min() <= result.max() <= 255


def test_sar_log_normalize_zeros():
    arr = np.zeros((32, 32), dtype=np.float32)
    result = _sar_log_normalize(arr)
    assert result.max() == 0


def test_normalize_bands_3d():
    arr = np.random.rand(64, 64, 3).astype(np.float32) * 255
    result = normalize_bands(arr)
    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8


def test_normalize_bands_2d():
    arr = np.random.rand(64, 64).astype(np.float32) * 255
    result = normalize_bands(arr)
    assert result.shape == (64, 64, 1)


# ─── Visualization ────────────────────────────────────────────────────────────

def test_rgb_visualization_3band():
    arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    meta = make_meta(bands=3)
    img, desc = generate_rgb_visualization(arr, meta)
    assert img.size == (64, 64)
    assert img.mode == "RGB"


def test_rgb_visualization_1band():
    arr = (np.random.rand(64, 64, 1) * 255).astype(np.uint8)
    meta = make_meta(bands=1)
    img, desc = generate_rgb_visualization(arr, meta)
    assert img.mode == "RGB"
    assert "grayscale" in desc.lower() or "single" in desc.lower()


def test_sar_visualization():
    arr = (np.random.rand(64, 64, 1) * 500).astype(np.float32)
    meta = make_meta(modality="sar", bands=1, is_sar=True, band_names=["VV"])
    img, desc = generate_sar_visualization(arr, meta)
    assert isinstance(img, Image.Image)
    assert "SAR" in desc or "sar" in desc.lower()


def test_false_color_4band():
    arr = (np.random.rand(64, 64, 4) * 255).astype(np.uint8)
    meta = make_meta(bands=4, band_names=["Blue", "Green", "Red", "NIR"])
    result = generate_false_color(arr, meta)
    assert result is not None
    img, desc = result
    assert isinstance(img, Image.Image)
    assert "false color" in desc.lower() or "composite" in desc.lower()


def test_false_color_not_available_3band():
    arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    meta = make_meta(bands=3)
    result = generate_false_color(arr, meta)
    assert result is None


def test_preprocess_for_llm():
    arr = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    meta = make_meta(width=100, height=100)
    img = preprocess_for_llm(arr, meta)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert max(img.size) <= LLM_MAX_SIZE


def test_preprocess_for_llm_large():
    arr = (np.random.rand(2000, 3000, 3) * 255).astype(np.uint8)
    meta = make_meta(width=3000, height=2000)
    img = preprocess_for_llm(arr, meta)
    assert max(img.size) <= LLM_MAX_SIZE


# ─── Change detection ─────────────────────────────────────────────────────────

def test_change_detection_identical_images():
    arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    meta = make_meta()
    result = generate_change_map(arr, arr, meta, meta)
    assert 0 <= result.change_pct <= 100
    assert 0 <= result.change_score <= 1


def test_change_detection_produces_map():
    arr_a = np.zeros((64, 64, 3), dtype=np.uint8)
    arr_b = (np.ones((64, 64, 3)) * 255).astype(np.uint8)
    meta = make_meta()
    result = generate_change_map(arr_a, arr_b, meta, meta)
    assert result.change_map_pil is not None
    assert result.change_pct > 0


def test_change_detection_evidence_strength():
    arr_a = np.zeros((64, 64, 3), dtype=np.uint8)
    arr_b = (np.ones((64, 64, 3)) * 255).astype(np.uint8)
    meta = make_meta()
    result = generate_change_map(arr_a, arr_b, meta, meta)
    assert result.evidence_strength in ("high", "moderate", "limited")


# ─── NDVI ─────────────────────────────────────────────────────────────────────

def test_ndvi_computation_4band():
    # B=0, G=1, R=2, NIR=3
    arr = np.zeros((32, 32, 4), dtype=np.float32)
    arr[:, :, 2] = 50.0   # Red
    arr[:, :, 3] = 200.0  # NIR → NDVI = (200-50)/(200+50) = 0.6
    meta = make_meta(bands=4, band_names=["Blue", "Green", "Red", "NIR"])
    result = _compute_ndvi(arr, meta)
    assert result is not None
    assert abs(result["mean_ndvi"] - 0.6) < 0.05
    assert result["vegetation_coverage_pct"] > 0


def test_ndvi_not_available_for_rgb():
    arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    meta = make_meta(bands=3)
    result = _compute_ndvi(arr, meta)
    assert result is None


# ─── Statistics formatting ────────────────────────────────────────────────────

def test_format_statistics_for_evidence():
    stats = {
        "image_size_pixels": "64 × 64",
        "band_statistics": [
            {"band": "Red", "min": 0, "max": 255, "mean": 128.0, "std": 40.0},
        ],
    }
    text = format_statistics_for_evidence(stats)
    assert "64 × 64" in text
    assert "Red" in text
    assert "mean=128" in text
