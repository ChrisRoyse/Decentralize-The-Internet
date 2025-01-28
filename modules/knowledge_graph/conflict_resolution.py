import logging
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ResolutionResult:
    action: str           # "keep_existing", "overwrite", "store_disputed"
    final_confidence: float
    reason: str = ""

class ConflictResolver:
    """Uses LLM ResolutionAgent to resolve knowledge graph conflicts."""
    
    def __init__(self, config, llm_resolution_agent=None):
        self.config = config
        self.llm_resolution_agent = llm_resolution_agent
        
    def resolve_conflict(self, existing_edge: Dict[str, Any], new_edge: Dict[str, Any]) -> ResolutionResult:
        """Use LLM to decide how to handle conflicting facts."""
        if not self.llm_resolution_agent:
            logger.error("No LLM resolution agent provided!")
            return ResolutionResult(
                action="keep_existing", 
                final_confidence=existing_edge.get("confidence", 0.5)
            )

        # Get LLM's decision
        result = self.llm_resolution_agent.resolve_conflict(existing_edge, new_edge)
        
        return ResolutionResult(
            action=result.get("action", "keep_existing"),
            final_confidence=result.get("final_confidence", 0.5),
            reason=result.get("reason", "No reason given")
        ) 