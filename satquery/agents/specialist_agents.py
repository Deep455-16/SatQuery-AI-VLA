"""
satquery/agents/specialist_agents.py

Agentic specialist tools as mandated by PS 26167:
"Instead of applying a single generic VLM, the system selects and executes
suitable remote-sensing specialist models..."

Each agent encapsulates a specific remote-sensing skill.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from PIL import Image

from satquery.model.terraq_vl import get_terraq_vl, TerraQTimeoutError, TerraQInsufficientError
from satquery.adapters.gemini_adapter import get_adapter
from satquery.tools.object_detector import detect as run_object_detection
from satquery.utils.image_preprocessor import generate_change_map, generate_rgb_visualization

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Standardized output from any specialist agent."""
    answer: str
    observations: list[str] = field(default_factory=list)
    inferences: list[str] = field(default_factory=list)
    evidence_strength: str = "moderate"
    limitations: list[str] = field(default_factory=list)
    visual_evidence: Optional[Image.Image] = None  # E.g., detection boxes, change maps
    models_used: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)


class BaseAgent:
    """Base class for all specialist agents."""
    name: str = "BaseAgent"
    
    def execute(self, query: str, images: list[Image.Image], metadata: list[dict], context: str) -> AgentResult:
        raise NotImplementedError("Specialist agent must implement execute()")

    def _fallback_inference(self, query: str, images: list[Image.Image], context: str, task_name: str) -> AgentResult:
        """Standard two-tier inference (TerraQ-VL -> Gemini) used by text-based agents."""
        terraq = get_terraq_vl()
        adapter = get_adapter()
        
        result = None
        models = []
        
        # 1. Try Gemini (Primary)
        if adapter.is_available():
            try:
                gem_res = adapter.analyze(query, images, context)
                models.append(getattr(gem_res, 'model_used', 'gemini-3.5-flash'))
                result = gem_res
            except Exception as e:
                logger.warning(f"Gemini primary inference failed (e.g. rate limit): {e}. Falling back to TerraQ-VL.")
                
        # 2. Try TerraQ-VL (Fallback)
        if result is None and terraq.is_available() and images:
            try:
                tq_res = terraq.run_analysis(images[0], query, context)
                models.append("TerraQ-VL (LLaVA-1.5-7B) [Fallback]")
                result = tq_res
            except Exception as e:
                logger.warning(f"TerraQ-VL fallback failed in {task_name}: {e}.")
            
        if result is None:
            return AgentResult(
                answer="Error: Gemini API limit reached and TerraQ-VL local fallback is unavailable/failed. Please wait 60 seconds and try again.",
                limitations=["Both Gemini and TerraQ-VL fallbacks failed."],
                models_used=models,
                tools_used=[task_name]
            )
            
        return AgentResult(
            answer=getattr(result, 'answer', ''),
            observations=getattr(result, 'observations', []),
            inferences=getattr(result, 'inferences', []),
            evidence_strength=getattr(result, 'evidence_strength', 'limited'),
            limitations=getattr(result, 'limitations', []),
            models_used=models,
            tools_used=[task_name]
        )


class VQAAgent(BaseAgent):
    """Mandatory baseline: Visual Question Answering."""
    name = "VQA_Specialist"
    
    def execute(self, query: str, images: list[Image.Image], metadata: list[dict], context: str) -> AgentResult:
        logger.info("Executing VQA Agent...")
        # VQA explicitly passes the user query to the inference engine
        return self._fallback_inference(query, images, context, "vqa_reasoning")


class CaptioningAgent(BaseAgent):
    """Single-image captioning / scene description."""
    name = "Captioning_Specialist"
    
    def execute(self, query: str, images: list[Image.Image], metadata: list[dict], context: str) -> AgentResult:
        logger.info("Executing Captioning Agent...")
        caption_query = "Provide a comprehensive scene description of this satellite image, including land cover and major features."
        return self._fallback_inference(caption_query, images, context, "scene_captioning")


class GroundingAgent(BaseAgent):
    """Text-guided region grounding using object detection models."""
    name = "Grounding_Specialist"
    
    def execute(self, query: str, images: list[Image.Image], metadata: list[dict], context: str) -> AgentResult:
        logger.info("Executing Grounding Agent...")
        if not images:
            return AgentResult(answer="No image provided for grounding.")
            
        # Run YOLO / OpenCV detection
        det_result = run_object_detection(images[0])
        
        # Format the detections to feed into the VQA to answer the specific text query
        if det_result.detections:
            det_text = ", ".join([f"{d.label} (conf: {d.confidence:.0%})" for d in det_result.detections])
            extended_context = f"{context}\nGrounding System detected: {det_text}"
        else:
            extended_context = f"{context}\nGrounding System detected no explicit bounded objects."
            
        # Run text inference to answer the user's specific grounding question using the detection context
        vqa_res = self._fallback_inference(query, images, extended_context, "text_guided_grounding")
        
        # Attach the bounding-box image as visual evidence
        vqa_res.visual_evidence = det_result.annotated_image
        vqa_res.tools_used.append(f"object_detection_{det_result.backend_used}")
        
        return vqa_res


class ChangeAnalysisAgent(BaseAgent):
    """Bi-temporal pair change detection and Change-VQA."""
    name = "Change_Analysis_Specialist"
    
    def execute(self, query: str, images: list[Image.Image], metadata: list[dict], context: str) -> AgentResult:
        logger.info("Executing Change Analysis Agent...")
        if len(images) < 2:
            return AgentResult(answer="Change analysis requires exactly two images.")
            
        # 1. Generate spatial change map
        try:
            # We need raw numpy arrays, but images might be different sizes and modes
            import numpy as np
            img1 = images[0].convert("RGB")
            img2 = images[1].convert("RGB").resize(img1.size, Image.Resampling.BILINEAR)
            
            arr1, arr2 = np.array(img1), np.array(img2)
            # Simulate change map via simple absolute difference thresholding for the agent
            diff = np.abs(arr1.astype(np.float32) - arr2.astype(np.float32))
            diff_mean = np.mean(diff, axis=2) if len(diff.shape) == 3 else diff
            mask = (diff_mean > 30).astype(np.uint8) * 255
            
            from matplotlib import cm
            colored_diff = cm.hot(mask / 255.0)[:, :, :3]
            colored_diff = (colored_diff * 255).astype(np.uint8)
            change_map_pil = Image.fromarray(colored_diff).convert("RGB")
            
            change_pct = (np.sum(mask > 0) / mask.size) * 100
            context += f"\nSpatial Change Map generated. Computed change percentage: {change_pct:.1f}%."
        except Exception as e:
            logger.warning(f"Failed to generate change map: {e}")
            change_map_pil = None
            
        # 2. Run Change-VQA
        vqa_res = self._fallback_inference(query, images, context, "change_understanding")
        vqa_res.visual_evidence = change_map_pil
        if change_map_pil is not None:
            vqa_res.tools_used.append("spatial_change_mapping")
        
        return vqa_res


class OpticalSARFusionAgent(BaseAgent):
    """Cross-modal pair analysis (Optical + SAR)."""
    name = "Optical_SAR_Fusion_Specialist"
    
    def execute(self, query: str, images: list[Image.Image], metadata: list[dict], context: str) -> AgentResult:
        logger.info("Executing Optical-SAR Fusion Agent...")
        if len(images) < 2:
            return AgentResult(answer="Fusion requires both Optical and SAR images.")
            
        # Inform the VLM that this is a cross-modal task
        fusion_query = f"Cross-modal task: Fuse information from the provided Optical and SAR images to answer: {query}"
        
        vqa_res = self._fallback_inference(fusion_query, images, context, "cross_modal_fusion")
        
        # Simple visual blend for evidence
        try:
            img1 = images[0].convert("RGBA")
            img2 = images[1].resize(img1.size, Image.Resampling.BILINEAR).convert("RGBA")
            fused = Image.blend(img1, img2, alpha=0.5)
            vqa_res.visual_evidence = fused.convert("RGB")
            vqa_res.tools_used.append("alpha_blending_fusion")
        except Exception as e:
            logger.warning(f"Failed to fuse images: {e}")
            
        return vqa_res
