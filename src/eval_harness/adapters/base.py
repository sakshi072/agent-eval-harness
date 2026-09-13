from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
@dataclass
class TargetResponse:
    """Generic shape every target adapter must produce, regardless of
    what the underlying system actually is."""

    answer: str
    retrieved_contexts: List[str] = field(default_factory=list)
    tools_called: List[str] = field(default_factory=list)
    thread_id: Optional[str] = None
    raw_response: Optional[dict] = None  # full original response, for debugging

class TargetAdapter(ABC):
    """Implement this once per target system. The rest of the harness
    (runner, RAGAS/DeepEval scorers, Langfuse reporting) never imports a
    concrete adapter directly - only this interface."""

    @abstractmethod
    async def ask(self, question:str, thread_id:Optional[str]=None) -> TargetResponse:
        """Send one question to the target system, return its response in
        the generic shape above."""
        raise NotImplementedError