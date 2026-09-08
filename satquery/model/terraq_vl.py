"""
satquery/model/terraq_vl.py

TerraQ-VL — LLaVA-1.5-7B local inference engine.

Priority:
  1. Run inference locally (CPU or GPU).
  2. If inference times out (>90s on CPU) → raise TerraQTimeoutError → caller falls back to Gemini.
  3. If answer is blank or too short → raise TerraQInsufficientError → caller falls back to Gemini.
"""
from __future__ import annotations

import logging
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)

# ── Custom exceptions ────────────────────────────────────────────────────────

class TerraQTimeoutError(RuntimeError):
    """Raised when local inference exceeds the CPU timeout threshold."""

class TerraQInsufficientError(RuntimeError):
    """Raised when the model output is too short/degenerate to be useful."""

# ── Result dataclass (mirrors GeminiAnalysisResult) ─────────────────────────

@dataclass
class TerraQResult:
    answer: str
    observations: list[str] = field(default_factory=list)
    inferences: list[str] = field(default_factory=list)
    evidence_strength: str = "moderate"
    limitations: list[str] = field(default_factory=list)
    model_used: str = "TerraQ-VL (LLaVA-1.5-7B)"
    raw_text: str = ""
    latency_ms: int = 0

# ── Hardware detection ───────────────────────────────────────────────────────

def _detect_hardware() -> dict:
    info = {
        'cpu': platform.processor(),
        'ram_gb': round(psutil.virtual_memory().total / 1e9, 1),
        'gpu': None,
        'cuda': False,
        'vram_gb': 0,
    }
    try:
        import torch
        info['cuda'] = torch.cuda.is_available()
        if info['cuda']:
            info['gpu'] = torch.cuda.get_device_name(0)
            info['vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except ImportError:
        pass
    return info

# ── Status constants ─────────────────────────────────────────────────────────

class TerraQVLStatus:
    READY       = 'READY'
    LOADING     = 'LOADING'
    UNAVAILABLE = 'UNAVAILABLE'
    BLOCKED     = 'BLOCKED'
    ERROR       = 'ERROR'
    DISABLED    = 'DISABLED'
    STANDBY     = 'STANDBY'

# ── LLaVA prompt helpers ─────────────────────────────────────────────────────

def _build_llava_prompt(query: str, evidence_context: str = "") -> str:
    """Build a simple prompt combining system instructions, context, and query."""
    system = (
        "You are SatQuery AI, a satellite image analysis assistant for ISRO. "
        "Answer concisely and state uncertainty explicitly."
    )
    evidence_block = f"\nEVIDENCE CONTEXT:\n{evidence_context}\n" if evidence_context else ""
    return f"{system}{evidence_block}\nUSER QUERY: {query}"

def _parse_terraq_output(raw: str) -> TerraQResult:
    """Parse free-form model output into structured TerraQResult."""
    text = raw.strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    observations: list[str] = []
    inferences: list[str] = []
    limitations: list[str] = []
    answer_lines: list[str] = []

    section = "answer"
    for line in lines:
        up = line.upper()
        if up.startswith("OBSERVATIONS:") or up.startswith("OBSERVATION:"):
            section = "obs"; continue
        if up.startswith("INFERENCES:") or up.startswith("INFERENCE:"):
            section = "inf"; continue
        if up.startswith("LIMITATIONS:") or up.startswith("LIMITATION:"):
            section = "lim"; continue

        clean = line.lstrip("-•").strip()
        if section == "answer":
            answer_lines.append(clean)
        elif section == "obs":
            if clean:
                observations.append(clean)
        elif section == "inf":
            if clean:
                inferences.append(clean)
        elif section == "lim":
            if clean:
                limitations.append(clean)

    answer = " ".join(answer_lines) if answer_lines else text

    return TerraQResult(
        answer=answer,
        observations=observations,
        inferences=inferences,
        evidence_strength="moderate",
        limitations=limitations,
        raw_text=raw,
    )

# ── Main model class ─────────────────────────────────────────────────────────

class TerraQVLModel:
    # Timeout in seconds for CPU-only inference before falling back to Gemini
    CPU_TIMEOUT_SECONDS = int(os.environ.get("VLM_CPU_TIMEOUT", "15"))
    # Minimum token count for a response to be considered "useful"
    MIN_USEFUL_TOKENS = int(os.environ.get("VLM_MIN_TOKENS", "15"))

    def __init__(self, model_path=None, lora_path=None, device='auto', dtype='auto'):
        self._status = TerraQVLStatus.DISABLED
        self._note = ''
        self._model = None
        self._processor = None

        self.enabled = os.environ.get('VLM_ENABLED', 'true').lower() == 'true'
        if not self.enabled:
            self._status = TerraQVLStatus.DISABLED
            self._note = 'VLM disabled in configuration.'
            return

        self.model_path = model_path or os.environ.get('VLM_MODEL_PATH', '')
        self.lora_path  = lora_path  or os.environ.get('VLM_LORA_PATH', '')
        self.device     = device if device != 'auto' else os.environ.get('VLM_DEVICE', 'auto')
        self.dtype      = dtype
        self.hw_info    = _detect_hardware()

        if not self.model_path:
            self._status = TerraQVLStatus.STANDBY
            self._note   = 'Local weights not configured (VLM_MODEL_PATH). Using Gemini.'
            return

        auto_load = os.environ.get('VLM_AUTO_LOAD', 'true').lower() == 'true'
        flag_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'vlm_ready.flag'
        )

        if auto_load or os.path.exists(flag_file):
            self._load_model()
        else:
            self._status = TerraQVLStatus.STANDBY
            self._note   = 'VLM_AUTO_LOAD=false. Call load_model_now() to load.'

    def _load_model(self):
        self._status = TerraQVLStatus.LOADING
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            self._status = TerraQVLStatus.UNAVAILABLE
            self._note   = 'transformers not installed. Run: pip install transformers'
            return
        try:
            import torch
            # Resolve device
            if self.device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = self.device

            logger.info("Loading TerraQ-VL (%s) on %s …", self.model_path, device)

            load_kwargs: dict = {"device_map": device, "trust_remote_code": True}
            if not torch.cuda.is_available():
                # CPU path
                load_kwargs["torch_dtype"] = torch.float32
                logger.info("CPU mode: loading in float32.")
            else:
                load_kwargs["torch_dtype"] = torch.float16

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, revision="2024-08-26", **load_kwargs
            )
            self._processor = AutoTokenizer.from_pretrained(
                self.model_path, revision="2024-08-26"
            )

            # Attach LoRA if configured
            if self.lora_path and os.path.exists(self.lora_path):
                try:
                    from peft import PeftModel
                    self._model = PeftModel.from_pretrained(self._model, self.lora_path)
                    logger.info("LoRA adapter loaded from %s", self.lora_path)
                except Exception as lora_err:
                    logger.warning("LoRA load failed (continuing without it): %s", lora_err)

            self._status = TerraQVLStatus.READY
            self._note   = f'Loaded on {device}. CPU timeout: {self.CPU_TIMEOUT_SECONDS}s.'
            logger.info("TerraQ-VL (Moondream2) ready on %s.", device)

        except Exception as e:
            self._status = TerraQVLStatus.ERROR
            self._note   = f"Failed to load model: {e}"
            logger.error("TerraQ-VL load error: %s", e)

    def load_model_now(self) -> bool:
        self._load_model()
        if self._status == TerraQVLStatus.READY:
            flag_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'vlm_ready.flag'
            )
            try:
                with open(flag_file, 'w') as f:
                    f.write('READY')
            except Exception:
                pass
        return self._status == TerraQVLStatus.READY

    @property
    def status(self) -> str:
        return self._status

    def is_available(self) -> bool:
        return self._status == TerraQVLStatus.READY

    # ── Core inference ───────────────────────────────────────────────────────

    def _do_generate(self, image_pil, prompt: str) -> str:
        """Raw generation — runs in a thread so it can be timed out."""
        import torch

        # Convert to RGB (required by Moondream)
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        with torch.no_grad():
            enc_image = self._model.encode_image(image_pil)
            answer = self._model.answer_question(enc_image, prompt, self._processor)
            
        return answer.strip()

    def run_analysis(
        self,
        image_pil,
        query: str,
        evidence_context: str = "",
    ) -> TerraQResult:
        """
        Run TerraQ-VL inference with a CPU timeout guard.

        Raises:
            RuntimeError: if model not loaded.
            TerraQTimeoutError: if inference exceeds CPU_TIMEOUT_SECONDS.
            TerraQInsufficientError: if output is too short to be useful.
        """
        if not self.is_available():
            raise RuntimeError(f"TerraQ-VL not available (status={self._status})")

        prompt = _build_llava_prompt(query, evidence_context)
        t0 = time.time()

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._do_generate, image_pil, prompt)
        try:
            raw_text = future.result(timeout=self.CPU_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            # We must not block waiting for the thread to finish
            executor.shutdown(wait=False, cancel_futures=True)
            raise TerraQTimeoutError(
                f"TerraQ-VL exceeded {self.CPU_TIMEOUT_SECONDS}s CPU timeout — switching to Gemini."
            )
        finally:
            executor.shutdown(wait=False)

        latency_ms = int((time.time() - t0) * 1000)

        # Quality gate
        word_count = len(raw_text.split())
        if word_count < self.MIN_USEFUL_TOKENS:
            raise TerraQInsufficientError(
                f"TerraQ-VL output too short ({word_count} words) — switching to Gemini."
            )

        logger.info("TerraQ-VL inference complete in %dms (%d words).", latency_ms, word_count)

        result = _parse_terraq_output(raw_text)
        result.latency_ms = latency_ms
        return result

    # ── Legacy stub methods (kept for compatibility) ─────────────────────────

    def run_vqa(self, image_pil, query: str) -> dict:
        result = self.run_analysis(image_pil, query)
        return {'answer': result.answer, 'confidence': 0.8, 'model': result.model_used}

    def run_caption(self, image_pil) -> dict:
        result = self.run_analysis(image_pil, "Describe this satellite image in detail.")
        return {'caption': result.answer, 'confidence': 0.8, 'model': result.model_used}

    def run_grounding(self, image_pil, query: str) -> dict:
        if not self.is_available():
            raise RuntimeError(f"TerraQ-VL not available. Status: {self._status}")
        result = self.run_analysis(image_pil, f"Locate and describe: {query}")
        return {'grounding_available': True, 'description': result.answer, 'bboxes': []}

    def get_status_dict(self) -> dict:
        return {
            'status':     self._status,
            'model_path': getattr(self, 'model_path', ''),
            'lora':       bool(getattr(self, 'lora_path', '')),
            'device':     getattr(self, 'device', ''),
            'note':       self._note,
            'hardware':   getattr(self, 'hw_info', {}),
            'cpu_timeout_seconds': self.CPU_TIMEOUT_SECONDS,
        }


# ── Module singleton ─────────────────────────────────────────────────────────

_terraq_vl_instance: TerraQVLModel | None = None

def get_terraq_vl() -> TerraQVLModel:
    global _terraq_vl_instance
    if _terraq_vl_instance is None:
        _terraq_vl_instance = TerraQVLModel()
    return _terraq_vl_instance
