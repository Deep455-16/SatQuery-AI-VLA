"""
satquery/adapters/gemini_adapter.py

Gemini multimodal adapter for SatQuery AI (PS 26167 / ISRO).
Uses the NEW google-genai SDK with gemini-3.5-flash.
"""
from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

logger = logging.getLogger(__name__)


class GeminiUnavailableError(Exception):
    """Raised when the Gemini API key is missing or the API is unreachable."""


class GeminiRateLimitError(Exception):
    """Raised when the free-tier rate limit is hit."""


@dataclass
class GeminiAnalysisResult:
    answer: str
    observations: list[str] = field(default_factory=list)
    inferences: list[str] = field(default_factory=list)
    evidence_strength: str = "limited"
    limitations: list[str] = field(default_factory=list)
    model_used: str = ""
    raw_text: str = ""


_SYSTEM_PROMPT = """You are SatQuery AI, a satellite image analysis assistant for ISRO/SAC (PS 26167).

CRITICAL RULES:
1. Answer ONLY from the supplied satellite image(s) and verified evidence.
2. Do NOT hallucinate objects, coordinates, measurements, or features not visible.
3. Distinguish OBSERVATIONS (directly visible) from INFERENCES (reasonably inferred).
4. State uncertainty explicitly when evidence is insufficient.
5. NEVER fabricate confident answers.

RESPONSE FORMAT:

ANSWER:
[Concise direct answer — 1-3 sentences]

OBSERVATIONS:
- [Directly visible feature]
- [Another observation]

INFERENCES:
- [Reasonably inferred conclusion]

EVIDENCE_STRENGTH: [high|moderate|limited]
[Brief justification]

LIMITATIONS:
- [What cannot be reliably determined]

If EVIDENCE CONTEXT is provided below, incorporate it into your answer."""


class GeminiAdapter:
    """Wraps the google-genai SDK for satellite image analysis."""

    def __init__(self):
        self._api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self._model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
        self._client = None
        if self._api_key:
            self._init_client()

    def _init_client(self):
        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
            logger.info("Gemini adapter initialised with model: %s", self._model_name)
        except ImportError:
            logger.error("google-genai not installed. Run: pip install google-genai")
            self._client = None
        except Exception as exc:
            logger.error("Failed to initialise Gemini client: %s", exc)
            self._client = None

    def is_available(self) -> bool:
        return bool(self._api_key and self._client is not None)

    def check_available(self):
        if not self._api_key:
            raise GeminiUnavailableError(
                "GEMINI_API_KEY not set. Add it to .env. "
                "Get a free key at: https://aistudio.google.com/apikey"
            )
        if self._client is None:
            self._init_client()
        if self._client is None:
            raise GeminiUnavailableError(
                "Gemini client failed to initialise. "
                "Run: pip install google-genai  then restart."
            )

    def analyze(
        self,
        query: str,
        images: list[Image.Image],
        evidence_context: str = "",
        max_retries: int = 3,
        timeout_seconds: int = 90,
    ) -> GeminiAnalysisResult:
        self.check_available()

        from google import genai
        from google.genai import types

        parts = []
        for i, img in enumerate(images):
            if len(images) == 2:
                label = "Before/Earlier Image:" if i == 0 else "After/Later Image:"
                parts.append(types.Part.from_text(text=label))
            buf = io.BytesIO()
            img_rgb = img.convert("RGB") if img.mode not in ("RGB", "L") else img
            img_rgb.save(buf, format="JPEG", quality=90)
            buf.seek(0)
            parts.append(
                types.Part.from_bytes(data=buf.read(), mime_type="image/jpeg")
            )

        prompt_lines = []
        if evidence_context:
            prompt_lines.append(
                f"EVIDENCE CONTEXT (from satellite image processing):\n{evidence_context}\n"
            )
        prompt_lines.append(f"USER QUERY: {query}")
        parts.append(types.Part.from_text(text="\n\n".join(prompt_lines)))

        contents = [types.Content(role="user", parts=parts)]
        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=1024,
        )

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=config,
                )
                return self._parse_response(response.text)

            except Exception as exc:
                exc_str = str(exc).lower()
                last_error = exc

                if "429" in exc_str or "quota" in exc_str or "rate" in exc_str:
                    raise GeminiRateLimitError(
                        "Gemini free-tier quota exceeded. "
                        "Limit: ~15 req/min, 1500/day. Wait a minute and retry."
                    ) from exc

                if "403" in exc_str or "api_key" in exc_str or "invalid" in exc_str:
                    raise GeminiUnavailableError(
                        "Gemini API key is invalid. Check GEMINI_API_KEY in .env."
                    ) from exc

                if "404" in exc_str or "not found" in exc_str:
                    raise GeminiUnavailableError(
                        f"Model '{self._model_name}' not found. "
                        "Set GEMINI_MODEL=gemini-2.5-flash in .env"
                    ) from exc

                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "Gemini attempt %d/%d failed: %s — retrying in %ds",
                        attempt + 1, max_retries, exc, wait,
                    )
                    time.sleep(wait)

        raise GeminiUnavailableError(
            f"Gemini API failed after {max_retries} attempts: {last_error}"
        )

    def _parse_response(self, raw_text: str) -> GeminiAnalysisResult:
        result = GeminiAnalysisResult(
            answer="", model_used=self._model_name, raw_text=raw_text
        )

        lines = raw_text.strip().split("\n")
        current_section = None
        answer_lines: list[str] = []
        obs_lines: list[str] = []
        inf_lines: list[str] = []
        lim_lines: list[str] = []
        ev_strength = "limited"

        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("ANSWER:"):
                current_section = "answer"
                rem = stripped[7:].strip()
                if rem:
                    answer_lines.append(rem)
                continue
            elif upper.startswith("OBSERVATIONS:"):
                current_section = "observations"; continue
            elif upper.startswith("INFERENCES:"):
                current_section = "inferences"; continue
            elif upper.startswith("EVIDENCE_STRENGTH:") or upper.startswith("EVIDENCE STRENGTH:"):
                current_section = "evidence_strength"
                rest = stripped.split(":", 1)[1].strip().lower()
                ev_strength = "high" if "high" in rest else "moderate" if "moderate" in rest else "limited"
                continue
            elif upper.startswith("LIMITATIONS:"):
                current_section = "limitations"; continue

            if current_section == "answer" and stripped:
                answer_lines.append(stripped)
            elif current_section == "observations":
                if stripped.startswith(("-", "•")): obs_lines.append(stripped.lstrip("-•").strip())
                elif stripped: obs_lines.append(stripped)
            elif current_section == "inferences":
                if stripped.startswith(("-", "•")): inf_lines.append(stripped.lstrip("-•").strip())
                elif stripped: inf_lines.append(stripped)
            elif current_section == "limitations":
                if stripped.startswith(("-", "•")): lim_lines.append(stripped.lstrip("-•").strip())
                elif stripped: lim_lines.append(stripped)

        result.answer = " ".join(answer_lines) if answer_lines else raw_text.strip()[:800]
        result.observations = [o for o in obs_lines if o]
        result.inferences = [i for i in inf_lines if i]
        result.limitations = [l for l in lim_lines if l]
        result.evidence_strength = ev_strength

        if not answer_lines and not obs_lines:
            result.answer = raw_text.strip()

        return result


_adapter_instance: Optional[GeminiAdapter] = None


def get_adapter() -> GeminiAdapter:
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = GeminiAdapter()
    return _adapter_instance
