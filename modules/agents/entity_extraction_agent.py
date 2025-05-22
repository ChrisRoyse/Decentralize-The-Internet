import logging
from typing import List, Dict, Any, Optional
from .llm_agent import BaseLLMAgent
from modules.knowledge_graph.ontology_manager import OntologyManager # Added import

logger = logging.getLogger(__name__)

class EntityExtractionAgent(BaseLLMAgent):
    """Uses DeepSeek to extract entities and relationships using LSF framework, with schema awareness."""
    
    def __init__(self, model_path: str, ontology_manager: Optional[OntologyManager] = None): # Added ontology_manager
        super().__init__(model_path=model_path) # Assuming BaseLLMAgent takes model_path
        self.ontology_manager: Optional[OntologyManager] = ontology_manager

    def extract_entities(self, text: str, source: str = "unknown") -> List[Dict[str, Any]]:
        """Extract entity relationships using Long Structured Framework."""

        entity_types_prompt_str = ""
        relationship_types_prompt_str = ""

        if self.ontology_manager:
            entity_types = self.ontology_manager.get_entity_types()
            if entity_types:
                entity_types_prompt_str = "\nAllowed Entity Types:\n" + "\n".join(f"- {etype}" for etype in entity_types)
            
            relationship_types_details_prompt_list = []
            relationship_types = self.ontology_manager.get_relationship_types()
            for rel_type in relationship_types:
                schema = self.ontology_manager.get_relationship_schema(rel_type)
                if schema and schema.get('domain') and schema.get('range'):
                    relationship_types_details_prompt_list.append(f"- {rel_type}: {schema['domain']} -> {schema['range']}")
            if relationship_types_details_prompt_list:
                relationship_types_prompt_str = "\nAllowed Relationship Types (and their expected entity types for domain->range):\n" + "\n".join(relationship_types_details_prompt_list)

        instructions = f"""Analyze this text and extract entity relationships:

{text}
{entity_types_prompt_str}
{relationship_types_prompt_str}

For each relationship found:
1. Identify two concrete entities.
2. Determine the type for each entity (must be one of the Allowed Entity Types).
3. Define their relationship type (must be one of the Allowed Relationship Types, and match the expected domain/range entity types).
4. Assign a confidence score based on:
   - Statement clarity (direct vs implied)
   - Source reliability
   - Information completeness

Output Format:
Return a JSON array of relationships, where each has:
- entity1: The first entity name
- entity1_type: The type of the first entity (must be one of the Allowed Entity Types)
- relation: The relationship type (must be one of the Allowed Relationship Types)
- entity2: The second entity name
- entity2_type: The type of the second entity (must be one of the Allowed Entity Types)
- confidence: Float 0.0-1.0
- source: "{source}"

Example:
[{{"entity1": "Microsoft", "entity1_type": "Organization", "relation": "acquired", "entity2": "Activision Blizzard", "entity2_type": "Organization", "confidence": 0.98}}]

Rules:
- Only extract explicitly stated relationships.
- Use normalized entity names (full company names, etc.).
- Ensure extracted entity types and relationship types strictly adhere to the 'Allowed Entity Types' and 'Allowed Relationship Types' sections if provided.
- Ensure the types of entity1 and entity2 are consistent with the domain and range specified for the chosen relationship type.
- Assign high confidence (>0.8) only to clear, direct statements.
- Include source attribution."""

        prompt = self._format_long_prompt(
            role="You are an expert knowledge graph entity extractor.",
            objective="Extract structured entity relationships from unstructured text according to a defined schema.",
            context="""This extraction will be used to build a knowledge graph of real-world facts.
Focus on concrete, verifiable relationships between entities. Adhere strictly to the provided schema for entity and relationship types.""",
            instructions=instructions,
            examples=[
                {
                    "input": "Microsoft announced today it has acquired Activision Blizzard for $69 billion.",
                    "output": '[{"entity1": "Microsoft", "entity1_type": "Organization", "relation": "acquired", "entity2": "Activision Blizzard", "entity2_type": "Organization", "confidence": 0.98}]'
                }
            ]
        )
        
        response = self._call_llm(prompt, max_length=2048, temperature=0.3) # Consider increasing max_length if schema is large
        
        try:
            import json
            facts = json.loads(response)
            
            # Validate and clean facts
            validated_facts = []
            for fact in facts:
                if all(k in fact for k in ["entity1", "entity1_type", "relation", "entity2", "entity2_type", "confidence"]):
                    # Optional: Add further validation using self.ontology_manager
                    # if self.ontology_manager:
                    #     if not self.ontology_manager.is_valid_entity_type(fact["entity1_type"]):
                    #         logger.warning(f"LLM returned invalid entity1_type: {fact}")
                    #         continue
                    #     if not self.ontology_manager.is_valid_entity_type(fact["entity2_type"]):
                    #         logger.warning(f"LLM returned invalid entity2_type: {fact}")
                    #         continue
                    #     if not self.ontology_manager.is_valid_relationship(fact["entity1_type"], fact["relation"], fact["entity2_type"]):
                    #         logger.warning(f"LLM returned invalid relationship based on types: {fact}")
                    #         continue
                    fact["source"] = source
                    validated_facts.append(fact)
            
            return validated_facts
            
        except Exception as e:
            logger.error(f"Error parsing entity extraction response: {e}")
            return []

    def _format_long_prompt(self, role: str, objective: str, context: str, 
                          instructions: str, examples: List[Dict[str, str]]) -> str:
        """Format prompt using Long Structured Framework from paper."""
        example_text = ""
        if examples:
            example_text = "\n\n# Examples\n"
            for i, example in enumerate(examples, 1):
                example_text += f"""Example {i}:
Input: {example['input']}
Output: {example['output']}\n"""

        return f"""# Role
{role}

# Objective
{objective}

# Context
{context}

# Instructions
{instructions}
{example_text}
Response:""" 