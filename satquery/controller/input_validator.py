"""
satquery/controller/input_validator.py

Validates image configurations and queries.
"""
from enum import Enum
from typing import List
import logging

from satquery.constants import SINGLE_IMAGE, CROSS_MODAL_PAIR, BITEMPORAL_PAIR, MULTI_IMAGE
from satquery.utils.image_io import ImageMeta

class ValidationErrorCode(str, Enum):
    UNSUPPORTED_FORMAT = 'UNSUPPORTED_FORMAT'
    FILE_TOO_LARGE = 'FILE_TOO_LARGE'
    INVALID_IMAGE_PAIR = 'INVALID_IMAGE_PAIR'
    CRS_MISMATCH = 'CRS_MISMATCH'
    NO_SPATIAL_OVERLAP = 'NO_SPATIAL_OVERLAP'
    INVALID_TEMPORAL_PAIR = 'INVALID_TEMPORAL_PAIR'
    MISSING_REQUIRED_BAND = 'MISSING_REQUIRED_BAND'
    EMPTY_QUERY = 'EMPTY_QUERY'
    INTERNAL_ERROR = 'INTERNAL_ERROR'

class ValidationError(Exception):
    def __init__(self, message: str, code: ValidationErrorCode = ValidationErrorCode.INTERNAL_ERROR):
        super().__init__(f"[{code.value}] {message}")
        self.code = code

def validate_query(query: str) -> None:
    if not query or not query.strip():
        raise ValidationError("Query cannot be empty", ValidationErrorCode.EMPTY_QUERY)

def classify_input_config(metas: List[ImageMeta]) -> str:
    if not metas:
        raise ValidationError("No images provided", ValidationErrorCode.INTERNAL_ERROR)
    
    if len(metas) == 1:
        return SINGLE_IMAGE
    
    if len(metas) > 2:
        return MULTI_IMAGE
        
    if len(metas) == 2:
        meta1, meta2 = metas[0], metas[1]
        
        # Check CRS
        if hasattr(meta1, 'crs') and hasattr(meta2, 'crs') and meta1.crs and meta2.crs and meta1.crs != meta2.crs:
            raise ValidationError("CRS mismatch between image pair", ValidationErrorCode.CRS_MISMATCH)
            
        # Dimension mismatch: log a warning only — different sensors/angles/times
        # will legitimately produce images with different pixel dimensions.
        if hasattr(meta1, 'width') and hasattr(meta2, 'width') and (meta1.width != meta2.width or
                (hasattr(meta1, 'height') and hasattr(meta2, 'height') and meta1.height != meta2.height)):
            logging.warning(
                "Image pair has different dimensions (%sx%s vs %sx%s); "
                "continuing — the pipeline will handle resampling if needed.",
                getattr(meta1, 'width', '?'), getattr(meta1, 'height', '?'),
                getattr(meta2, 'width', '?'), getattr(meta2, 'height', '?'),
            )
            
        # Check temporal order
        if hasattr(meta1, 'acquisition_date') and hasattr(meta2, 'acquisition_date'):
            if meta1.acquisition_date and meta2.acquisition_date:
                if meta1.acquisition_date > meta2.acquisition_date:
                    logging.warning("Temporal order warning: Image 1 acquisition date is later than Image 2.")
                
        # Determine cross-modal or bitemporal based on modality
        mod1 = getattr(meta1, 'modality', 'unknown')
        mod2 = getattr(meta2, 'modality', 'unknown')
        # One optical + one SAR = cross-modal pair
        modalities = {mod1, mod2}
        if 'optical' in modalities and 'sar' in modalities:
            return CROSS_MODAL_PAIR
        # Same modality (both optical or both SAR) = bitemporal pair
        return BITEMPORAL_PAIR

    raise ValidationError("Invalid number of images", ValidationErrorCode.INTERNAL_ERROR)
