"""Global constants for SatQuery AI."""

# Supported geospatial / benchmark image formats
GEOSPATIAL_FORMATS = ("GeoTIFF", "TIFF")
BENCHMARK_FORMATS = ("PNG", "JPEG")
SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Input configuration types
SINGLE_IMAGE = "single"
CROSS_MODAL_PAIR = "cross_modal_pair"
BITEMPORAL_PAIR = "bitemporal_pair"
MULTI_IMAGE = "multi_image"  # 3+ images

# Formal PS 26167 task tokens
TASK_VQA = "VQA"
TASK_CAPTION = "CAPTION"
TASK_GROUNDING = "GROUNDING"
TASK_CHANGE_DETECTION = "CHANGE_DETECTION"
TASK_CHANGE_VQA = "CHANGE_VQA"
TASK_OPTICAL_SAR_FUSION = "OPTICAL_SAR_FUSION"
TASK_COMPARISON = "COMPARISON"
TASK_UNSUPPORTED = "UNSUPPORTED"

# Legacy task tokens (mirrors GeoChat-style special task tokens)
TASK_TOKEN_VQA = "[vqa]"
TASK_TOKEN_CAPTION = "[caption]"
TASK_TOKEN_GROUNDING = "[grounding]"
TASK_TOKEN_CHANGE = "[change]"
TASK_TOKEN_FUSION = "[fusion]"

DEFAULT_CONFIDENCE_FLOOR = 0.5

# Confidence mapping from evidence strength string
CONFIDENCE_MAP = {
    "high": 0.85,
    "moderate": 0.65,
    "limited": 0.35,
}