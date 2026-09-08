"""
satquery/model/specialist_registry.py

Predefined registry of specialist models/tools as mandated by PS 26167.
Provides instances of agentic components based on task classification.
"""
from __future__ import annotations

import logging
from typing import Optional
from satquery.agents.specialist_agents import (
    BaseAgent,
    VQAAgent,
    CaptioningAgent,
    GroundingAgent,
    ChangeAnalysisAgent,
    OpticalSARFusionAgent
)

logger = logging.getLogger(__name__)

class SpecialistRegistry:
    def __init__(self):
        self._agents: dict[str, BaseAgent] = {
            'VQA': VQAAgent(),
            'CAPTION': CaptioningAgent(),
            'GROUNDING': GroundingAgent(),
            'CHANGE_DETECTION': ChangeAnalysisAgent(),
            'CHANGE_VQA': ChangeAnalysisAgent(),
            'OPTICAL_SAR_FUSION': OpticalSARFusionAgent(),
        }

    def get_agent(self, task_token: str) -> Optional[BaseAgent]:
        """Retrieve the appropriate specialist agent for a given task."""
        agent = self._agents.get(task_token)
        if agent:
            logger.info(f"Registry selected: {agent.name} for task {task_token}")
        else:
            logger.warning(f"No specialist agent found in registry for task {task_token}")
        return agent

    def list_agents(self) -> list[str]:
        return [agent.name for agent in self._agents.values()]

# Module singleton
_registry = None
def get_registry() -> SpecialistRegistry:
    global _registry
    if _registry is None:
        _registry = SpecialistRegistry()
    return _registry
