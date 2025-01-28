import logging
from typing import List, Dict, Any
from ..agents.entity_extraction_agent import EntityExtractionAgent

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Extracts entities and relationships from text using LLM-based extraction.
    """
    
    def __init__(self, model_path: str = None):
        self.extraction_agent = EntityExtractionAgent(model_path=model_path)
        logger.info("EntityExtractor initialized with LLM agent")

    def extract_entities(self, text: str, source: str = "unknown") -> List[Dict[str, Any]]:
        """
        Extract entity relationships from text using LLM.
        Returns list of dicts with {entity1, relation, entity2, confidence, source}
        """
        try:
            facts = self.extraction_agent.extract_entities(text, source)
            logger.debug(f"Extracted {len(facts)} facts from text")
            return facts
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return [] 