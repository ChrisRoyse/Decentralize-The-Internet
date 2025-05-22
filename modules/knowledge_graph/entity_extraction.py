import logging
from typing import List, Dict, Any
from ..agents.entity_extraction_agent import EntityExtractionAgent
from modules.knowledge_graph.ontology_manager import OntologyManager # Added import

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Extracts entities and relationships from text using LLM-based extraction,
    leveraging an ontology for schema-awareness.
    """
    
    def __init__(self, model_path: str = None):
        # It's generally better to pass an existing OntologyManager if one is managed globally,
        # but for this task, we'll instantiate one here.
        # This assumes OntologyManager's default path "config/schema.yaml" is correct
        # relative to the project's execution root.
        self.ontology_manager = OntologyManager() 
        self.extraction_agent = EntityExtractionAgent(model_path=model_path, ontology_manager=self.ontology_manager)
        logger.info("EntityExtractor initialized with LLM agent and OntologyManager")

    def extract_entities(self, text: str, source: str = "unknown") -> List[Dict[str, Any]]:
        """
        Extract entity relationships from text using LLM.
        Returns list of dicts with {
            entity1, entity1_type, 
            relation, 
            entity2, entity2_type, 
            confidence, source
        }
        """
        try:
            facts = self.extraction_agent.extract_entities(text, source)
            logger.debug(f"Extracted {len(facts)} facts from text")
            return facts
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return [] 