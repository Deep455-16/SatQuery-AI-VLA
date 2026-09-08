"""
satquery/controller/agent_controller.py

Agentic Controller / Orchestrator for SatQuery AI.

Strictly follows PS 26167 guidelines:
1. interpret the query and classify the requested task;
2. check the number, modality, format, metadata, and compatibility of the input images;
3. select one or more models or tools from a predefined registry;
4. configure only permitted task parameters and execute the selected workflow;
5. combine textual and spatial outputs, estimate confidence, and return visual evidence;
6. provide an auditable execution summary...
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from satquery.model.specialist_registry import get_registry
from satquery.agents.specialist_agents import AgentResult
from satquery.utils.image_preprocessor import generate_rgb_visualization, resize_for_llm
from satquery.controller.evidence_layer import build_evidence, EvidenceBundle

logger = logging.getLogger(__name__)

SINGLE_IMAGE     = "single_image"
BITEMPORAL_PAIR  = "bitemporal_pair"
CROSS_MODAL_PAIR = "cross_modal_pair"
MULTI_IMAGE      = "multi_image"

@dataclass
class TraceStep:
    step: str
    status: str
    detail: str = ""
    duration_ms: int = 0

@dataclass
class ExecutionTrace:
    task: str
    models: list[str]
    tools: list[str]
    parameters: dict[str, Any]
    steps: list[TraceStep]
    timing: dict[str, int]
    status: str
    images_analyzed: int
    images_failed: int

@dataclass
class ExecutionResult:
    task: str
    tools_used: list[str]
    parameters: dict[str, Any]
    answer: str
    observations: list[str] = field(default_factory=list)
    inferences: list[str] = field(default_factory=list)
    evidence_strength: str = "moderate"
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    processing_time_ms: int = 0
    change_map_pil: Optional[Any] = None
    rgb_visualization: Optional[Any] = None
    false_color_visualization: Optional[Any] = None
    false_color_description: str = ""
    detection_visualization: Optional[Any] = None
    models_used: list[str] = field(default_factory=list)
    execution_trace: Optional[ExecutionTrace] = None
    confidence: Optional[float] = None
    image_ids: list[str] = field(default_factory=list)
    images_failed: int = 0

    def to_dict(self):
        data = {
            "task": self.task,
            "answer": self.answer,
            "observations": self.observations,
            "inferences": self.inferences,
            "evidence_strength": self.evidence_strength,
            "limitations": self.limitations,
            "warnings": self.warnings,
            "processing_time_ms": self.processing_time_ms,
            "tools_used": self.tools_used,
            "models_used": self.models_used,
            "confidence": self.confidence,
            "images_failed": self.images_failed,
        }
        if self.execution_trace:
            data["execution_trace"] = {
                "task": self.execution_trace.task,
                "models": self.execution_trace.models,
                "tools": self.execution_trace.tools,
                "parameters": self.execution_trace.parameters,
                "steps": [
                    {
                        "step": s.step,
                        "status": s.status,
                        "detail": s.detail,
                        "duration_ms": s.duration_ms
                    } for s in self.execution_trace.steps
                ],
                "timing": self.execution_trace.timing,
                "status": self.execution_trace.status,
                "images_analyzed": self.execution_trace.images_analyzed,
                "images_failed": self.execution_trace.images_failed
            }
        return data


class AgentController:
    """The central Orchestrator model."""
    
    def __init__(self):
        self.registry = get_registry()

    def classify_task(self, query: str, input_kind: str) -> str:
        """1. Base Orchestrator Model: Interprets query and assigns the task."""
        from satquery.adapters.gemini_adapter import get_adapter
        
        prompt = f"""You are the Base Orchestrator Model for SatQuery AI.
Your job is to route the user's query to the correct remote-sensing specialist model.

Input Configuration: {input_kind}
User Query: "{query}"

Available Specialist Models:
- VQA (General questions about a single image)
- CAPTION (Describing, summarizing, or classifying land-cover)
- GROUNDING (Locating, highlighting, or detecting specific objects)
- CHANGE_DETECTION (Analyzing differences between two bitemporal images)
- OPTICAL_SAR_FUSION (Analyzing cross-modal optical and SAR pairs)

Reply with ONLY the exact token of the specialist model needed (e.g. VQA). No other text."""

        try:
            adapter = get_adapter()
            if adapter.is_available():
                response = adapter._client.models.generate_content(
                    model=adapter._model_name,
                    contents=prompt
                )
                token = response.text.strip().upper()
                valid_tokens = ['VQA', 'CAPTION', 'GROUNDING', 'CHANGE_DETECTION', 'OPTICAL_SAR_FUSION']
                for valid in valid_tokens:
                    if valid in token:
                        logger.info(f"Base Orchestrator Model assigned task: {valid}")
                        return valid
        except Exception as e:
            logger.warning(f"Orchestrator Model failed ({e}), falling back to heuristic routing.")

        # Fallback if Orchestrator Model is offline
        query_lower = query.lower()
        if input_kind == BITEMPORAL_PAIR: return 'CHANGE_DETECTION'
        if input_kind == CROSS_MODAL_PAIR: return 'OPTICAL_SAR_FUSION'
        if any(kw in query_lower for kw in ['highlight', 'locate', 'where']): return 'GROUNDING'
        if any(kw in query_lower for kw in ['describe', 'caption', 'classify']): return 'CAPTION'
        return 'VQA'

    def classify_input_config(self, images: list) -> str:
        """2. check the number, modality..."""
        if len(images) == 1:
            return SINGLE_IMAGE
        elif len(images) == 2:
            m1 = getattr(images[0][1], 'modality', 'optical').lower()
            m2 = getattr(images[1][1], 'modality', 'optical').lower()
            if m1 == 'sar' and m2 == 'optical' or m1 == 'optical' and m2 == 'sar':
                return CROSS_MODAL_PAIR
            return BITEMPORAL_PAIR
        return MULTI_IMAGE

    def handle_query(
        self,
        query: str,
        images: list[tuple[Any, Any]],  # [(numpy_array, ImageMeta)]
        session_id: Optional[str] = None
    ) -> ExecutionResult:
        start_time = time.time()
        steps = []
        parameters = {}
        preprocessing_notes = []
        image_ids = [getattr(m, 'path', f'img_{i}').split('/')[-1] for i, (_, m) in enumerate(images)]

        def add_step(name: str, start: float, status: str, detail: str = ""):
            steps.append(TraceStep(name, status, detail, int((time.time() - start) * 1000)))

        # ── Step 1 & 2: Input Config and Task Classification ──
        step_start = time.time()
        input_kind = self.classify_input_config(images)
        task_token = self.classify_task(query, input_kind)
        parameters['input_type'] = input_kind
        parameters['intent'] = task_token
        parameters['images_analyzed'] = len(images)
        add_step("query_interpretation_and_input_validation", step_start, "success", f"Classified as {task_token} for {input_kind}")

        # ── Step 3: Select from predefined registry ──
        step_start = time.time()
        agent = self.registry.get_agent(task_token)
        if not agent:
            # Fallback for MULTI_IMAGE comparison not mapped to a specific agent
            if input_kind == MULTI_IMAGE:
                return self._handle_multi_image(query, images, session_id, task_token, image_ids, start_time)
                
            raise RuntimeError(f"No specialist agent configured for task: {task_token}")
        add_step("agent_selection", step_start, "success", f"Routed to {agent.name}")

        # ── Step 4: Configure permitted parameters and Preprocess ──
        step_start = time.time()
        rgb_viz, llm_images = None, []
        try:
            for arr, meta in images:
                rgb, _ = generate_rgb_visualization(arr, meta)
                llm_images.append(resize_for_llm(rgb))
                if rgb_viz is None: rgb_viz = rgb  # keep first for display
            add_step("image_preprocessing", step_start, "success")
        except Exception as e:
            add_step("image_preprocessing", step_start, "failed", str(e))
            raise RuntimeError(f"Preprocessing failed: {e}")

        # Build evidence context
        evidence_context = ""
        try:
            ev = build_evidence(images, task_token, input_kind)
            evidence_context = ev.to_context_string() if ev else ""
        except Exception as e:
            preprocessing_notes.append(f"Evidence builder error: {e}")

        # ── Step 5: Execute the selected workflow (Agent execution) ──
        step_start = time.time()
        agent_result: AgentResult = agent.execute(query, llm_images, [m for _, m in images], evidence_context)
        add_step("specialist_execution", step_start, "success", f"Executed {agent.name}")

        # ── Step 6: Combine outputs, estimate confidence, build execution trace ──
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        conf_map = {"high": 0.85, "moderate": 0.65, "limited": 0.35}
        confidence = conf_map.get(agent_result.evidence_strength, 0.5)

        tools = ["task_router", "image_preprocessor", agent.name] + agent_result.tools_used

        trace = ExecutionTrace(
            task=task_token,
            models=agent_result.models_used,
            tools=tools,
            parameters=parameters,
            steps=steps,
            timing={"total_ms": elapsed_ms},
            status="success",
            images_analyzed=len(images),
            images_failed=0
        )

        return ExecutionResult(
            task=task_token,
            tools_used=tools,
            parameters=parameters,
            answer=agent_result.answer,
            observations=agent_result.observations,
            inferences=agent_result.inferences,
            evidence_strength=agent_result.evidence_strength,
            limitations=agent_result.limitations,
            warnings=preprocessing_notes,
            processing_time_ms=elapsed_ms,
            rgb_visualization=rgb_viz,
            detection_visualization=agent_result.visual_evidence if task_token == 'GROUNDING' else None,
            change_map_pil=agent_result.visual_evidence if task_token in ['CHANGE_DETECTION', 'CHANGE_VQA'] else None,
            models_used=agent_result.models_used,
            execution_trace=trace,
            confidence=confidence,
            image_ids=image_ids,
            images_failed=0
        )

    def _handle_multi_image(self, query: str, images: list, session_id: str, task_token: str, image_ids: list, start_time: float) -> ExecutionResult:
        """Fallback multi-image looper when a specific agent isn't registered for 'COMPARISON'."""
        # Simple implementation that routes each image to the VQA agent
        agent = self.registry.get_agent('VQA')
        answers = []
        models_used = []
        for idx, (arr, meta) in enumerate(images):
            rgb, _ = generate_rgb_visualization(arr, meta)
            llm_img = resize_for_llm(rgb)
            res = agent.execute(query, [llm_img], [meta], "")
            answers.append(f"Image {idx+1}: {res.answer}")
            models_used.extend([m for m in res.models_used if m not in models_used])
            
        final_answer = "\n".join(answers)
        elapsed_ms = int((time.time() - start_time) * 1000)
        trace = ExecutionTrace(
            task=task_token, models=models_used, tools=["multi_image_looper", "VQA_Specialist"],
            parameters={"intent": task_token, "images_analyzed": len(images)}, steps=[],
            timing={"total_ms": elapsed_ms}, status="success", images_analyzed=len(images), images_failed=0
        )
        return ExecutionResult(
            task=task_token, tools_used=trace.tools, parameters=trace.parameters,
            answer=final_answer, processing_time_ms=elapsed_ms, models_used=models_used,
            execution_trace=trace, image_ids=image_ids
        )
