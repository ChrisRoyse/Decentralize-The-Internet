import logging
from typing import Optional
from .llm_agent import PlannerAgent, ComparisonAgent, ResolutionAgent
from .entity_extraction_agent import EntityExtractionAgent

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    """Coordinates specialized LLM agents using local quantum-accelerated inference."""
    
    def __init__(self, model_path: Optional[str] = None):
        # Use a single model path for all agents to save memory
        self.model_path = model_path or "mistralai/Mistral-7B-v0.1"
        
        logger.info(f"Initializing LLM agents with model: {self.model_path}")
        
        # Initialize agents with same model
        self.planner = PlannerAgent(model_path=self.model_path)
        self.comparer = ComparisonAgent(model_path=self.model_path)
        self.resolver = ResolutionAgent(model_path=self.model_path)
        self.extractor = EntityExtractionAgent(model_path=self.model_path)
        
        logger.info("LLM agents initialized successfully")

    def plan_crawl(self, candidate_urls):
        return self.planner.rank_urls(candidate_urls)

    def deduplicate(self, new_text, existing_text):
        return self.comparer.is_duplicate(new_text, existing_text)

    def resolve_facts(self, existing_fact, new_fact):
        return self.resolver.resolve_conflict(existing_fact, new_fact)

    def extract_entities(self, text: str, source: str = "unknown"):
        return self.extractor.extract_entities(text, source) 