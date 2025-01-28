import logging
from typing import List, Dict, Any
from .llm_agent import BaseLLMAgent

logger = logging.getLogger(__name__)

class EntityExtractionAgent(BaseLLMAgent):
    """Uses DeepSeek to extract entities and relationships using LSF framework."""
    
    def extract_entities(self, text: str, source: str = "unknown") -> List[Dict[str, Any]]:
        """Extract entity relationships using Long Structured Framework."""
        prompt = self._format_long_prompt(
            role="You are an expert knowledge graph entity extractor.",
            objective="Extract structured entity relationships from unstructured text.",
            context="""This extraction will be used to build a knowledge graph of real-world facts.
Focus on concrete, verifiable relationships between entities. Avoid speculation or uncertain claims.""",
            instructions=f"""Analyze this text and extract entity relationships:

{text}

For each relationship found:
1. Identify two concrete entities (companies, people, products, etc.)
2. Define their relationship type (acquired, partnered, launched, etc.)
3. Assign a confidence score based on:
   - Statement clarity (direct vs implied)
   - Source reliability
   - Information completeness

Output Format:
Return a JSON array of relationships, where each has:
- entity1: The first entity name
- relation: The relationship type
- entity2: The second entity name
- confidence: Float 0.0-1.0
- source: "{source}"

Example:
[{{"entity1": "TechCorp", "relation": "acquired", "entity2": "StartupX", "confidence": 0.95}}]

Rules:
- Only extract explicitly stated relationships
- Use normalized entity names (full company names, etc.)
- Assign high confidence (>0.8) only to clear, direct statements
- Include source attribution""",
            examples=[
                {
                    "input": "Microsoft announced today it has acquired Activision Blizzard for $69 billion.",
                    "output": '[{"entity1": "Microsoft", "relation": "acquired", "entity2": "Activision Blizzard", "confidence": 0.98}]'
                }
            ]
        )
        
        response = self._call_llm(prompt, max_length=2048, temperature=0.3)
        
        try:
            import json
            facts = json.loads(response)
            
            # Validate and clean facts
            validated_facts = []
            for fact in facts:
                if all(k in fact for k in ["entity1", "relation", "entity2", "confidence"]):
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