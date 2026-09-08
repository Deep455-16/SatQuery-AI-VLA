"""
satquery/utils/image_preprocessor.py

Satellite image preprocessing for SatQuery AI.

Handles:
- Band normalization (percentile clipping, SAR log-scale)
- RGB visualization (from multispectral or grayscale)
- False-color visualization (NIR/SWIR composites for Sentinel-2)
- SAR-specific visualization (log-scale, adaptive histogram equalization)
- Intelligent resize/compression for LLM submission (≤1024px)
- Change map generation using OpenCV (real pixel comparison)
- Image statistics extraction (for evidence context)

Design principle:
  Original image is NEVER modified. All outputs are new PIL Images.
  The change map uses actual computer vision (not LLM hallucination).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance

from satquery.utils.image_io import ImageMeta

logger = logging.getLogger(__name__)

# Maximum image dimension (pixels) to send to the LLM
LLM_MAX_SIZE = 1024

# ─── Change detection result ──────────────────────────────────────────────────

@dataclass
class ChangeDetectionResult:
    change_pct: float                     # percentage of changed pixels (0–100)
    change_score: float                   # normalized change magnitude (0–1)
    change_map_pil: Optional[Image.Image] # color-coded change mask
    num_changed_regions: int = 0
    changed_area_summary: str = ""
    evidence_strength: str = "limited"    # high | moderate | limited
    notes: list[str] = field(default_factory=list)


# ─── Normalization utilities ──────────────────────────────────────────────────

def _percentile_clip(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Clip array to percentile range and normalize to [0, 255] uint8."""
    arr = arr.astype(np.float32)
    p_lo = np.percentile(arr, lo)
    p_hi = np.percentile(arr, hi)
    if p_hi == p_lo:
        return np.zeros_like(arr, dtype=np.uint8)
    clipped = np.clip(arr, p_lo, p_hi)
    normalized = (clipped - p_lo) / (p_hi - p_lo) * 255.0
    return normalized.astype(np.uint8)


def _sar_log_normalize(arr: np.ndarray) -> np.ndarray:
    """
    SAR-specific normalization using log-scale (linear → dB-like scale).
    Handles the large dynamic range typical of SAR backscatter.
    """
    arr = arr.astype(np.float32)
    # Clip negatives (backscatter should be positive linear power)
    arr = np.clip(arr, 1e-6, None)
    # Log transform
    log_arr = np.log10(arr)
    # Percentile clip after log transform
    return _percentile_clip(log_arr, lo=1.0, hi=99.0)


def normalize_bands(arr: np.ndarray, is_sar: bool = False) -> np.ndarray:
    """Normalize a multi-band array to uint8."""
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    result = np.zeros((*arr.shape[:2], arr.shape[2]), dtype=np.uint8)
    for b in range(arr.shape[2]):
        band = arr[:, :, b]
        if is_sar:
            result[:, :, b] = _sar_log_normalize(band)
        else:
            result[:, :, b] = _percentile_clip(band)
    return result


# ─── Visualization generators ─────────────────────────────────────────────────

def generate_rgb_visualization(arr: np.ndarray, meta: ImageMeta) -> tuple[Image.Image, str]:
    """
    Generate an RGB visualization of the satellite image.

    Returns:
        (PIL Image, description string)
    """
    h, w = arr.shape[:2]
    bands = arr.shape[2] if arr.ndim == 3 else 1

    # ── SAR: single-channel log visualization ──
    if meta.is_sar or meta.modality == "sar":
        if bands >= 2:
            # Dual-pol SAR: show VV, VH, VV/VH ratio
            vv = _sar_log_normalize(arr[:, :, 0])
            vh = _sar_log_normalize(arr[:, :, 1])
            ratio = _percentile_clip((arr[:, :, 0].astype(np.float32) /
                                     (arr[:, :, 1].astype(np.float32) + 1e-6)))
            rgb = np.stack([vv, vh, ratio], axis=-1)
            desc = "SAR dual-polarization composite (VV, VH, VV/VH ratio)"
        else:
            gray = _sar_log_normalize(arr[:, :, 0] if bands > 1 else arr.squeeze())
            rgb = np.stack([gray, gray, gray], axis=-1)
            desc = "SAR single-polarization visualization (log-scale)"
        img = Image.fromarray(rgb.astype(np.uint8))
        img = _apply_sar_enhancement(img)
        return img, desc

    # ── Sentinel-2: use B4/B3/B2 (Red/Green/Blue) ──
    if meta.is_sentinel2 and bands >= 4:
        # Standard Sentinel-2 ordering: B2=Blue(0), B3=Green(1), B4=Red(2), B8=NIR(3)
        # But band order varies by product; try to detect
        r_idx, g_idx, b_idx = _find_rgb_bands(meta.band_names, bands)
        r = _percentile_clip(arr[:, :, r_idx].astype(np.float32))
        g = _percentile_clip(arr[:, :, g_idx].astype(np.float32))
        b_ch = _percentile_clip(arr[:, :, b_idx].astype(np.float32))
        rgb = np.stack([r, g, b_ch], axis=-1)
        desc = "Sentinel-2 true color (RGB: B4/B3/B2)"
        return Image.fromarray(rgb), desc

    # ── Standard RGB ──
    if bands >= 3:
        r = _percentile_clip(arr[:, :, 0].astype(np.float32))
        g = _percentile_clip(arr[:, :, 1].astype(np.float32))
        b_ch = _percentile_clip(arr[:, :, 2].astype(np.float32))
        rgb = np.stack([r, g, b_ch], axis=-1)
        return Image.fromarray(rgb), "RGB visualization (bands 1, 2, 3)"

    # ── Single band: grayscale ──
    band_data = arr[:, :, 0] if bands > 1 else arr.squeeze()
    gray = _percentile_clip(band_data.astype(np.float32))
    rgb = np.stack([gray, gray, gray], axis=-1)
    return Image.fromarray(rgb), "Single-band grayscale visualization"


def generate_false_color(arr: np.ndarray, meta: ImageMeta) -> Optional[tuple[Image.Image, str]]:
    """
    Generate a false-color visualization (NIR/Red/Green) if bands are available.
    Returns None if the required bands are not present.
    """
    bands = arr.shape[2] if arr.ndim == 3 else 1

    if meta.is_sentinel2 and bands >= 4:
        # Find NIR band (B8 in Sentinel-2)
        nir_idx = _find_band_index(meta.band_names, ["B8", "NIR", "nir"])
        r_idx = _find_band_index(meta.band_names, ["B4", "Red", "red", "R"])
        g_idx = _find_band_index(meta.band_names, ["B3", "Green", "green", "G"])

        if all(i is not None for i in [nir_idx, r_idx, g_idx]):
            nir = _percentile_clip(arr[:, :, nir_idx].astype(np.float32))
            red = _percentile_clip(arr[:, :, r_idx].astype(np.float32))
            grn = _percentile_clip(arr[:, :, g_idx].astype(np.float32))
            fc = np.stack([nir, red, grn], axis=-1)
            return Image.fromarray(fc), "False color composite (NIR/Red/Green — vegetation appears red)"

    # Try 4-band: assume B1=Blue, B2=Green, B3=Red, B4=NIR
    if bands == 4:
        nir = _percentile_clip(arr[:, :, 3].astype(np.float32))
        red = _percentile_clip(arr[:, :, 2].astype(np.float32))
        grn = _percentile_clip(arr[:, :, 1].astype(np.float32))
        fc = np.stack([nir, red, grn], axis=-1)
        return Image.fromarray(fc), "False color composite (band 4/3/2 as NIR/Red/Green)"

    return None


def _find_rgb_bands(band_names: list[str], band_count: int) -> tuple[int, int, int]:
    """Find R, G, B band indices from band names."""
    r = _find_band_index(band_names, ["B4", "Red", "RED", "R", "red"]) or min(2, band_count - 1)
    g = _find_band_index(band_names, ["B3", "Green", "GREEN", "G", "green"]) or min(1, band_count - 1)
    b = _find_band_index(band_names, ["B2", "Blue", "BLUE", "B", "blue"]) or 0
    return r, g, b


def _find_band_index(band_names: list[str], candidates: list[str]) -> Optional[int]:
    """Find the index of the first matching band name."""
    for candidate in candidates:
        for i, name in enumerate(band_names):
            if name and (name.upper() == candidate.upper() or candidate.upper() in name.upper()):
                return i
    return None


def _apply_sar_enhancement(img: Image.Image) -> Image.Image:
    """Apply adaptive contrast enhancement for SAR imagery."""
    try:
        from PIL import ImageOps
        # CLAHE-like: using PIL's autocontrast
        img = ImageOps.autocontrast(img, cutoff=1)
        # Slight sharpening for speckle
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.3)
    except Exception:
        pass
    return img


def generate_sar_visualization(arr: np.ndarray, meta: ImageMeta) -> tuple[Image.Image, str]:
    """Specialized SAR visualization (same as generate_rgb_visualization for SAR)."""
    return generate_rgb_visualization(arr, meta)


def resize_for_llm(img: Image.Image, max_size: int = LLM_MAX_SIZE) -> Image.Image:
    """
    Resize a PIL image to fit within max_size × max_size while preserving aspect ratio.
    Uses high-quality Lanczos resampling.
    """
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def preprocess_for_llm(arr: np.ndarray, meta: ImageMeta) -> Image.Image:
    """
    Full preprocessing pipeline: raw array → LLM-ready PIL Image.
    
    Steps:
    1. Generate best available visualization
    2. Resize to ≤ LLM_MAX_SIZE
    3. Convert to RGB JPEG quality
    """
    img, _ = generate_rgb_visualization(arr, meta)
    img = resize_for_llm(img, max_size=LLM_MAX_SIZE)
    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ─── Change detection ─────────────────────────────────────────────────────────

def generate_change_map(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    meta_a: ImageMeta,
    meta_b: ImageMeta,
    threshold_pct: float = 15.0,
) -> ChangeDetectionResult:
    """
    Real change detection using OpenCV image comparison.
    
    Algorithm:
    1. Convert both images to grayscale
    2. Resize to common dimensions if needed
    3. Normalize both images to [0, 255]
    4. Compute absolute difference
    5. Apply threshold to create binary change mask
    6. Apply morphological operations to clean noise
    7. Find contours (changed regions)
    8. Create color-coded change visualization

    This is actual pixel-level comparison, NOT LLM output.
    """
    try:
        import cv2
    except ImportError:
        return ChangeDetectionResult(
            change_pct=0.0,
            change_score=0.0,
            change_map_pil=None,
            notes=["OpenCV not available — change detection disabled. Install opencv-python-headless."],
        )

    notes = []

    # Convert to grayscale for comparison
    def to_gray(arr: np.ndarray, meta: ImageMeta) -> np.ndarray:
        """Convert to uint8 grayscale using raw min-max normalization.
        We deliberately do NOT use percentile clipping here because that
        would collapse uniform images (pure black or pure white) to zero,
        destroying the difference between them.
        """
        def raw_norm(band: np.ndarray) -> np.ndarray:
            b = band.astype(np.float32)
            mn, mx = b.min(), b.max()
            if mx == mn:
                # Uniform band: preserve its absolute value scaled 0-255
                # (e.g. pure black=0, pure white=255)
                return np.clip(b / 255.0 * 255, 0, 255).astype(np.uint8)
            return ((b - mn) / (mx - mn) * 255).astype(np.uint8)

        if arr.ndim == 2:
            return raw_norm(arr)
        if meta.is_sar:
            sar = arr[:, :, 0].astype(np.float32)
            sar = np.clip(sar, 1e-6, None)
            log_sar = np.log10(sar)
            return raw_norm(log_sar)
        # Multispectral: luminance
        bands = arr.shape[2]
        r_idx = min(0, bands - 1)
        g_idx = min(1, bands - 1)
        b_idx = min(2, bands - 1)
        r = raw_norm(arr[:, :, r_idx])
        g = raw_norm(arr[:, :, g_idx])
        b = raw_norm(arr[:, :, b_idx])
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    gray_a = to_gray(arr_a, meta_a)
    gray_b = to_gray(arr_b, meta_b)


    # Resize to match if needed
    if gray_a.shape != gray_b.shape:
        target_h = min(gray_a.shape[0], gray_b.shape[0])
        target_w = min(gray_a.shape[1], gray_b.shape[1])
        gray_a = cv2.resize(gray_a, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray_b = cv2.resize(gray_b, (target_w, target_h), interpolation=cv2.INTER_AREA)
        notes.append("Images resized to common dimensions for comparison (results are approximate).")

    # Absolute difference
    diff = cv2.absdiff(gray_a, gray_b)
    change_score = float(np.mean(diff)) / 255.0

    # Adaptive threshold: use mean + k*std
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff))

    if mean_diff >= 200:
        # Nearly entire image changed (e.g., black vs white) — use low absolute threshold
        thresh_val = 30
    else:
        thresh_val = int(mean_diff + 1.5 * std_diff)
        # Clamp: minimum 15 (noise floor), maximum 200 (preserve sensitivity)
        thresh_val = max(min(thresh_val, 200), 15)

    _, binary = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

    # Morphological cleanup: remove noise, fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # remove noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # fill gaps

    # Calculate change percentage
    total_pixels = binary.shape[0] * binary.shape[1]
    changed_pixels = int(np.sum(binary > 0))
    change_pct = (changed_pixels / total_pixels) * 100.0

    # Find changed regions (contours)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter tiny regions (< 0.01% of image)
    min_area = total_pixels * 0.0001
    significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    num_regions = len(significant_contours)

    # Build evidence strength
    if change_pct > 10.0:
        ev_strength = "high"
    elif change_pct > 2.0:
        ev_strength = "moderate"
    else:
        ev_strength = "limited"

    # Area summary
    if change_pct < 1.0:
        area_summary = f"Minimal change detected ({change_pct:.1f}% of image area)."
    elif change_pct < 5.0:
        area_summary = f"Small areas of change detected ({change_pct:.1f}% of image area) in {num_regions} region(s)."
    elif change_pct < 20.0:
        area_summary = f"Moderate change detected ({change_pct:.1f}% of image area) across {num_regions} region(s)."
    else:
        area_summary = f"Significant change detected ({change_pct:.1f}% of image area) across {num_regions} region(s)."

    # Create color-coded change map visualization
    h, w = binary.shape
    change_map_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Base: grayscale average of both images
    avg_gray = ((gray_a.astype(np.float32) + gray_b.astype(np.float32)) / 2).astype(np.uint8)
    change_map_rgb[:, :, 0] = avg_gray
    change_map_rgb[:, :, 1] = avg_gray
    change_map_rgb[:, :, 2] = avg_gray
    # Changed pixels: highlight in red/orange
    change_mask = binary > 0
    change_map_rgb[change_mask, 0] = 255  # Red channel max
    change_map_rgb[change_mask, 1] = 80   # Some green for orange
    change_map_rgb[change_mask, 2] = 0    # No blue

    # Draw contour outlines
    contour_img = change_map_rgb.copy()
    cv2.drawContours(contour_img, significant_contours, -1, (255, 255, 0), 2)
    change_map_rgb = contour_img

    change_map_pil = Image.fromarray(change_map_rgb)
    change_map_pil = resize_for_llm(change_map_pil, max_size=LLM_MAX_SIZE)

    return ChangeDetectionResult(
        change_pct=change_pct,
        change_score=change_score,
        change_map_pil=change_map_pil,
        num_changed_regions=num_regions,
        changed_area_summary=area_summary,
        evidence_strength=ev_strength,
        notes=notes,
    )


# ─── Image statistics for evidence context ────────────────────────────────────

def extract_image_statistics(arr: np.ndarray, meta: ImageMeta) -> dict:
    """
    Extract numerical image statistics to provide as evidence to the LLM.
    These are real measurements, not estimates.
    """
    stats = {}
    bands = arr.shape[2] if arr.ndim == 3 else 1
    h, w = arr.shape[:2]

    stats["image_size_pixels"] = f"{w} × {h}"
    stats["total_pixels"] = w * h

    if bands >= 1:
        # Per-band statistics
        band_stats = []
        for b in range(min(bands, 6)):  # limit to 6 bands
            band_data = arr[:, :, b] if arr.ndim == 3 else arr
            band_data_f = band_data.astype(np.float32)
            band_stats.append({
                "band": meta.band_names[b] if b < len(meta.band_names) else f"Band {b+1}",
                "min": float(np.min(band_data_f)),
                "max": float(np.max(band_data_f)),
                "mean": float(np.mean(band_data_f)),
                "std": float(np.std(band_data_f)),
            })
        stats["band_statistics"] = band_stats

    # NDVI if NIR and Red bands are available
    ndvi_result = _compute_ndvi(arr, meta)
    if ndvi_result is not None:
        stats["ndvi"] = ndvi_result

    # Brightness/darkness ratio
    if arr.ndim == 3 and bands >= 3:
        gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
        dark_pct = float(np.sum(gray < 50) / gray.size * 100)
        bright_pct = float(np.sum(gray > 200) / gray.size * 100)
        mid_pct = 100 - dark_pct - bright_pct
        stats["brightness_distribution"] = {
            "dark_pct": round(dark_pct, 1),
            "mid_pct": round(mid_pct, 1),
            "bright_pct": round(bright_pct, 1),
        }

    return stats


def _compute_ndvi(arr: np.ndarray, meta: ImageMeta) -> Optional[dict]:
    """Compute NDVI if NIR and Red bands are available."""
    if arr.ndim < 3:
        return None

    bands = arr.shape[2]
    nir_idx = _find_band_index(meta.band_names, ["B8", "NIR", "nir", "Near-Infrared"])
    r_idx = _find_band_index(meta.band_names, ["B4", "Red", "red", "R"])

    # Fallback for 4-band images (B=0, G=1, R=2, NIR=3)
    if nir_idx is None and bands >= 4:
        nir_idx = 3
    if r_idx is None and bands >= 3:
        r_idx = 2

    if nir_idx is None or r_idx is None:
        return None

    nir = arr[:, :, nir_idx].astype(np.float32)
    red = arr[:, :, r_idx].astype(np.float32)

    denom = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)

    # Classify
    veg_pct = float(np.sum(ndvi > 0.3) / ndvi.size * 100)
    sparse_veg_pct = float(np.sum((ndvi > 0.1) & (ndvi <= 0.3)) / ndvi.size * 100)
    non_veg_pct = 100 - veg_pct - sparse_veg_pct

    return {
        "mean_ndvi": round(float(np.mean(ndvi)), 3),
        "max_ndvi": round(float(np.max(ndvi)), 3),
        "vegetation_coverage_pct": round(veg_pct, 1),
        "sparse_vegetation_pct": round(sparse_veg_pct, 1),
        "non_vegetation_pct": round(non_veg_pct, 1),
        "note": "NDVI > 0.3 classified as dense vegetation; 0.1–0.3 as sparse/stressed vegetation",
    }


def format_statistics_for_evidence(stats: dict) -> str:
    """Format image statistics into a readable evidence string for the LLM."""
    lines = ["Image statistics (computed from pixel data):"]

    if "image_size_pixels" in stats:
        lines.append(f"  Size: {stats['image_size_pixels']} pixels")

    if "band_statistics" in stats:
        lines.append("  Band statistics:")
        for bs in stats["band_statistics"]:
            lines.append(
                f"    {bs['band']}: min={bs['min']:.1f}, max={bs['max']:.1f}, "
                f"mean={bs['mean']:.1f}, std={bs['std']:.1f}"
            )

    if "ndvi" in stats:
        ndvi = stats["ndvi"]
        lines.append(f"  NDVI analysis:")
        lines.append(f"    Mean NDVI: {ndvi['mean_ndvi']:.3f}")
        lines.append(f"    Dense vegetation (NDVI>0.3): {ndvi['vegetation_coverage_pct']:.1f}%")
        lines.append(f"    Sparse vegetation (NDVI 0.1–0.3): {ndvi['sparse_vegetation_pct']:.1f}%")

    if "brightness_distribution" in stats:
        bd = stats["brightness_distribution"]
        lines.append(f"  Brightness: {bd['dark_pct']:.0f}% dark, {bd['mid_pct']:.0f}% mid, {bd['bright_pct']:.0f}% bright")

    return "\n".join(lines)
