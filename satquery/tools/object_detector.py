"""
satquery/tools/object_detector.py

Object detection for satellite imagery using YOLOv8n (primary)
with OpenCV contour-based detection as a CPU-friendly fallback.

Red bounding boxes with class labels are drawn on the image and
returned as a PIL Image for display in the frontend.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ── Detection result ─────────────────────────────────────────────────────────

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list[float]   # [x1, y1, x2, y2] in pixel coords
    source: str = "yolo"  # "yolo" or "opencv"

@dataclass
class DetectionResult:
    detections: list[Detection] = field(default_factory=list)
    annotated_image: Optional[Image.Image] = None
    backend_used: str = "none"
    error: Optional[str] = None

# ── Drawing helpers ──────────────────────────────────────────────────────────

_RED   = (220, 30, 30)
_WHITE = (255, 255, 255)
_BOX_WIDTH = 3

def _draw_boxes(img: Image.Image, detections: list[Detection]) -> Image.Image:
    """Draw red bounding boxes + labels onto a PIL image."""
    annotated = img.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        # Red bounding box
        for t in range(_BOX_WIDTH):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=_RED)

        # Label background + text
        label_text = f"{det.label} {det.confidence:.0%}"
        bbox_text = draw.textbbox((x1, y1 - 18), label_text, font=font)
        draw.rectangle(bbox_text, fill=_RED)
        draw.text((x1, y1 - 18), label_text, fill=_WHITE, font=font)

    return annotated

# ── YOLO backend ─────────────────────────────────────────────────────────────

def _detect_yolo(img: Image.Image, conf_threshold: float = 0.25) -> list[Detection]:
    """Run YOLOv8n detection. Returns [] if ultralytics not installed."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed")

    # Load nano model (CPU-friendly, downloads once, ~6MB)
    model = YOLO("yolov8n.pt")
    results = model.predict(img, conf=conf_threshold, verbose=False)

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            xyxy    = box.xyxy[0].tolist()
            label   = model.names.get(cls_id, str(cls_id))
            detections.append(Detection(
                label=label, confidence=conf, bbox=xyxy, source="yolo"
            ))
    return detections

# ── OpenCV contour fallback ───────────────────────────────────────────────────

def _detect_opencv(img: Image.Image, min_area: int = 2000) -> list[Detection]:
    """Simple OpenCV contour-based object detector — always available as fallback."""
    import cv2
    import numpy as np

    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold to highlight distinct regions
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh  = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        11, 4
    )
    # Morphological cleanup
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Aspect ratio filter (skip very thin lines)
        aspect = w / max(h, 1)
        if aspect > 10 or aspect < 0.1:
            continue
        detections.append(Detection(
            label="region",
            confidence=min(0.5 + area / (img.width * img.height), 0.95),
            bbox=[x, y, x + w, y + h],
            source="opencv",
        ))

    # Limit to top 20 by area to avoid clutter
    detections.sort(key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]), reverse=True)
    return detections[:20]

# ── Public API ───────────────────────────────────────────────────────────────

def detect(
    img: Image.Image,
    conf_threshold: float = 0.25,
    max_detections: int = 30,
) -> DetectionResult:
    """
    Run object detection on a PIL image.
    Tries YOLOv8n first; falls back to OpenCV contours if ultralytics unavailable.

    Returns a DetectionResult with:
      - detections:      list of Detection objects
      - annotated_image: PIL image with red bounding boxes drawn
      - backend_used:    "yolo" or "opencv"
    """
    if img is None:
        return DetectionResult(error="No image provided")

    # Resize large images to speed up detection (max 1024px on longest side)
    max_side = 1024
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    try:
        detections = _detect_yolo(img, conf_threshold)
        backend = "yolo"
        logger.info("YOLO detected %d objects.", len(detections))
    except ImportError:
        logger.info("ultralytics not installed — using OpenCV contour fallback.")
        try:
            detections = _detect_opencv(img)
            backend = "opencv"
            logger.info("OpenCV detected %d regions.", len(detections))
        except Exception as e:
            return DetectionResult(error=f"Detection failed: {e}")
    except Exception as e:
        logger.warning("YOLO failed (%s) — trying OpenCV fallback.", e)
        try:
            detections = _detect_opencv(img)
            backend = "opencv"
        except Exception as e2:
            return DetectionResult(error=f"Both detectors failed: {e} / {e2}")

    detections = detections[:max_detections]
    annotated = _draw_boxes(img, detections)

    return DetectionResult(
        detections=detections,
        annotated_image=annotated,
        backend_used=backend,
    )
