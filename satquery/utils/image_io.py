"""
satquery/utils/image_io.py

Image loading and metadata extraction utilities.
Supports GeoTIFF/TIFF (via rasterio) and PNG/JPEG.

Extracts rich geospatial metadata:
- width, height, band count, CRS, affine transform, bounds
- pixel resolution, data type, acquisition date (from TIFF tags)
- sensor information (inferred from metadata/filename)
- band names where available
- modality detection (optical/SAR/unknown)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from PIL import Image

from satquery.constants import SUPPORTED_EXTENSIONS

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False


@dataclass
class ImageMeta:
    path: str
    modality: str                       # "optical", "sar", or "unknown"
    width: int
    height: int
    bands: int
    crs: Optional[str] = None
    is_georeferenced: bool = False
    format: str = ""
    # Extended fields
    resolution_x: Optional[float] = None   # pixel width in CRS units
    resolution_y: Optional[float] = None   # pixel height in CRS units
    transform: Optional[Any] = None        # affine transform object
    bounds: Optional[Any] = None           # BoundingBox
    data_type: str = "uint8"
    acquisition_date: Optional[str] = None
    sensor: Optional[str] = None
    band_names: list[str] = field(default_factory=list)
    nodata_value: Optional[float] = None
    is_sentinel2: bool = False             # True if Sentinel-2 band structure detected
    is_sar: bool = False                   # True if SAR-specific metadata found
    extra: dict = field(default_factory=dict)

    def to_display_dict(self) -> dict:
        """Human-readable metadata dict for the UI."""
        d = {
            "Format": self.format,
            "Modality": self.modality.capitalize(),
            "Width × Height": f"{self.width} × {self.height} px",
            "Bands": self.bands,
            "Data Type": self.data_type,
        }
        if self.crs:
            d["CRS"] = str(self.crs)
        if self.resolution_x and self.resolution_y:
            d["Pixel Resolution"] = f"{abs(self.resolution_x):.4f} × {abs(self.resolution_y):.4f} (CRS units)"
        if self.acquisition_date:
            d["Acquisition Date"] = self.acquisition_date
        if self.sensor:
            d["Sensor"] = self.sensor
        if self.band_names:
            d["Band Names"] = ", ".join(self.band_names[:8])
        if self.bounds:
            b = self.bounds
            d["Geographic Bounds"] = f"({b.left:.4f}, {b.bottom:.4f}, {b.right:.4f}, {b.top:.4f})"
        return d

    def to_evidence_string(self) -> str:
        """Structured string to include in Gemini evidence context."""
        lines = [
            f"Image format: {self.format}",
            f"Modality: {self.modality}",
            f"Dimensions: {self.width} × {self.height} pixels",
            f"Bands: {self.bands}",
            f"Data type: {self.data_type}",
        ]
        if self.crs:
            lines.append(f"CRS: {self.crs}")
        if self.resolution_x:
            lines.append(f"Pixel resolution: {abs(self.resolution_x):.4f} CRS units/pixel")
        if self.acquisition_date:
            lines.append(f"Acquisition date: {self.acquisition_date}")
        if self.sensor:
            lines.append(f"Sensor: {self.sensor}")
        if self.band_names:
            lines.append(f"Band names: {', '.join(self.band_names[:8])}")
        if self.is_sentinel2:
            lines.append("Spectral bands: Sentinel-2 multispectral detected")
        if self.is_sar:
            lines.append("SAR imagery: backscatter/polarization data")
        return "\n".join(lines)


def _infer_sensor(path: str, meta_tags: dict) -> Optional[str]:
    """Infer sensor from filename or TIFF metadata tags."""
    name = os.path.basename(path).lower()
    # Check tags first
    for key in ("sensor", "instrument", "mission", "satellite"):
        for tag_key, val in meta_tags.items():
            if key in str(tag_key).lower():
                return str(val)[:60]
    # Filename heuristics
    if "s2" in name or "sentinel2" in name or "sentinel-2" in name:
        return "Sentinel-2 (ESA)"
    if "s1" in name or "sentinel1" in name or "sentinel-1" in name:
        return "Sentinel-1 SAR (ESA)"
    if "landsat" in name or "lc08" in name or "lc09" in name:
        return "Landsat (USGS)"
    if "cartosat" in name:
        return "Cartosat (ISRO)"
    if "risat" in name:
        return "RISAT SAR (ISRO)"
    if "resourcesat" in name:
        return "ResourceSat (ISRO)"
    if "modis" in name:
        return "MODIS (NASA)"
    return None


def _infer_acquisition_date(path: str, meta_tags: dict) -> Optional[str]:
    """Try to extract acquisition date from metadata or filename."""
    # Check TIFF tags
    for key, val in meta_tags.items():
        k = str(key).lower()
        if "date" in k or "time" in k or "acquisition" in k:
            date_str = str(val)[:20]
            if re.search(r"\d{4}", date_str):
                return date_str
    # Filename pattern: YYYYMMDD or YYYY-MM-DD
    name = os.path.basename(path)
    m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _detect_sentinel2_bands(band_names: list[str], band_count: int) -> bool:
    """Detect if image has Sentinel-2 band structure."""
    s2_bands = {"B2", "B3", "B4", "B8", "B8A", "B11", "B12"}
    if any(b.upper() in s2_bands for b in band_names):
        return True
    # By band count: S2 has 13 bands (all), 10m subset has 4, 20m has 6, etc.
    if band_count in (4, 6, 10, 12, 13):
        name_lower = " ".join(band_names).lower()
        if "blue" in name_lower or "nir" in name_lower or "swir" in name_lower:
            return True
    return False


def _detect_sar(path: str, meta_tags: dict, band_names: list[str], band_count: int) -> bool:
    """Detect SAR imagery."""
    name = os.path.basename(path).lower()
    if "sar" in name or "s1" in name or "risat" in name:
        return True
    # SAR specific tags
    for key, val in meta_tags.items():
        k = str(key).lower()
        if "polarization" in k or "polarimetric" in k or "backscatter" in k:
            return True
    # Single-band high-dynamic-range (typical of SAR)
    if band_count == 1:
        for bn in band_names:
            if any(p in bn.upper() for p in ("VV", "VH", "HH", "HV")):
                return True
    return False


def guess_modality(path: str, bands: int, band_names: list[str] = None,
                   meta_tags: dict = None) -> str:
    """
    Determine image modality from available metadata.
    Priority: metadata > filename > band count heuristic.
    """
    if band_names is None:
        band_names = []
    if meta_tags is None:
        meta_tags = {}

    if _detect_sar(path, meta_tags, band_names, bands):
        return "sar"
    if bands >= 3:
        return "optical"
    name = os.path.basename(path).lower()
    if "optical" in name or "s2" in name or "cartosat" in name:
        return "optical"
    if bands == 1:
        # Ambiguous: could be SAR or panchromatic
        return "sar"
    return "unknown"


def load_image(path: str) -> tuple[np.ndarray, ImageMeta]:
    """
    Load a single image file and return (array, metadata).

    For GeoTIFF: uses rasterio to extract full geospatial metadata.
    For PNG/JPEG: uses PIL with filename-based heuristics.

    Returns:
        array: shape (H, W) or (H, W, C) in uint8 or original dtype
        meta: rich ImageMeta dataclass
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext in {".tif", ".tiff"} and _HAS_RASTERIO:
        return _load_geotiff(path)

    return _load_plain_image(path)


def _load_geotiff(path: str) -> tuple[np.ndarray, ImageMeta]:
    """Load a GeoTIFF using rasterio and extract full metadata."""
    with rasterio.open(path) as src:
        arr = src.read()           # (bands, H, W)
        arr = np.moveaxis(arr, 0, -1)  # → (H, W, bands)

        tags = dict(src.tags()) if hasattr(src, "tags") else {}
        band_names = list(src.descriptions or [])
        band_names = [b if b else f"Band_{i+1}" for i, b in enumerate(band_names)]

        transform = src.transform
        resolution_x = transform.a if transform else None
        resolution_y = abs(transform.e) if transform else None

        is_s2 = _detect_sentinel2_bands(band_names, src.count)
        is_sar = _detect_sar(path, tags, band_names, src.count)
        modality = "sar" if is_sar else ("optical" if src.count >= 3 or is_s2 else "unknown")
        sensor = _infer_sensor(path, tags)
        acq_date = _infer_acquisition_date(path, tags)

        meta = ImageMeta(
            path=path,
            modality=modality,
            width=src.width,
            height=src.height,
            bands=src.count,
            crs=str(src.crs) if src.crs else None,
            is_georeferenced=src.crs is not None,
            format="GeoTIFF" if src.crs else "TIFF",
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            transform=transform,
            bounds=src.bounds if src.crs else None,
            data_type=str(src.dtypes[0]) if src.dtypes else "uint8",
            acquisition_date=acq_date,
            sensor=sensor,
            band_names=band_names,
            nodata_value=src.nodata,
            is_sentinel2=is_s2,
            is_sar=is_sar,
            extra=tags,
        )
        return arr, meta


def _load_plain_image(path: str) -> tuple[np.ndarray, ImageMeta]:
    """Load a plain image (PNG/JPEG) using PIL."""
    img = Image.open(path)
    arr = np.array(img)
    bands = 1 if arr.ndim == 2 else arr.shape[-1]
    ext = os.path.splitext(path)[1].lower().lstrip(".")

    band_names = []
    if bands == 1:
        band_names = ["Gray"]
    elif bands == 3:
        band_names = ["Red", "Green", "Blue"]
    elif bands == 4:
        band_names = ["Red", "Green", "Blue", "Alpha"]

    modality = guess_modality(path, bands, band_names)

    meta = ImageMeta(
        path=path,
        modality=modality,
        width=img.width,
        height=img.height,
        bands=bands,
        crs=None,
        is_georeferenced=False,
        format=img.format or ext.upper(),
        data_type="uint8",
        band_names=band_names,
        acquisition_date=_infer_acquisition_date(path, {}),
        sensor=_infer_sensor(path, {}),
    )
    return arr, meta


def check_pair_compatibility(meta_a: ImageMeta, meta_b: ImageMeta) -> list[str]:
    """
    Returns a list of compatibility warnings/errors for a pair of images.
    Empty list == compatible.
    """
    issues = []
    if meta_a.width != meta_b.width or meta_a.height != meta_b.height:
        issues.append(
            f"Dimension mismatch: {meta_a.width}×{meta_a.height} vs "
            f"{meta_b.width}×{meta_b.height}. "
            "Images will be resized to match for comparison, but co-registration may be imprecise."
        )
    if meta_a.is_georeferenced and meta_b.is_georeferenced and meta_a.crs != meta_b.crs:
        issues.append(
            f"CRS mismatch: '{meta_a.crs}' vs '{meta_b.crs}'. "
            "Reprojection is recommended for accurate co-registration."
        )
    if not meta_a.is_georeferenced or not meta_b.is_georeferenced:
        issues.append(
            "One or both images lack georeferencing metadata. "
            "Spatial alignment cannot be fully verified; results are approximate."
        )
    return issues
