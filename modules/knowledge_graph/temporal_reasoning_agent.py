from typing import Dict, Any, List
from datetime import datetime
from ..agents.llm_agent import BaseLLMAgent
import logging

logger = logging.getLogger(__name__)

class TemporalReasoningAgent(BaseLLMAgent):
    """Agent for temporal reasoning and high-confidence edge case inference."""
    
    def infer_temporal_updates(self, fact: Dict[str, Any], current_time: datetime = None) -> List[Dict[str, Any]]:
        """Infer temporal updates (e.g., age changes) from existing facts."""
        if current_time is None:
            current_time = datetime.now()

        prompt = self._format_long_prompt(
            role="""You are a Temporal Reasoning Expert with the ability to infer 
high-confidence updates based on the passage of time.""",
            objective="""Analyze facts and identify any high-confidence temporal updates 
that should be made based on the current time.""",
            context=f"""Current time: {current_time.isoformat()}
We need to identify any facts that can be updated with extremely high confidence 
based on the passage of time alone. Only suggest updates when confidence exceeds 97%.""",
            instructions=f"""Analyze this fact for temporal implications:

Fact: {fact}

Consider:
1. Age Progression
   - Birth dates -> current age
   - Time-based statuses (e.g., employment duration)
   - Project/event completion status

2. Time-Based State Changes
   - Event transitions (scheduled -> ongoing -> completed)
   - Status changes based on known deadlines
   - Lifecycle progressions

3. Temporal Relationships
   - Duration calculations
   - Sequential dependencies
   - Periodic updates

Rules:
- Only suggest updates with >97% confidence
- Must be based purely on temporal logic
- Must preserve fact provenance
- Include confidence calculation explanation

Output Format:
Return a JSON array of suggested updates, each containing:
- original_fact: The source fact
- updated_fact: The new version
- confidence: Float (must be >0.97)
- reasoning: Detailed explanation
- temporal_logic: Mathematical/logical proof""",
            examples=[
                {
                    "input": """Fact: {
                        "entity1": "John Smith",
                        "relation": "hasAge",
                        "entity2": "25",
                        "metadata": {
                            "birth_date": "1998-03-15"
                        }
                    }""",
                    "output": """{
                        "updates": [{
                            "original_fact": {"entity1": "John Smith", "relation": "hasAge", "entity2": "25"},
                            "updated_fact": {"entity1": "John Smith", "relation": "hasAge", "entity2": "26"},
                            "confidence": 0.9999,
                            "reasoning": "Based on birth date and current date, age can be calculated exactly",
                            "temporal_logic": "2024 - 1998 = 26 years (accounting for birth month)"
                        }]
                    }"""
                }
            ]
        )
        
        response = self._call_llm(prompt, temperature=0.1)  # Very low temp for precision
        
        try:
            import json
            result = json.loads(response)
            updates = result.get("updates", [])
            
            # Filter for only very high confidence updates
            high_confidence_updates = [
                update for update in updates 
                if update.get("confidence", 0) > 0.97
            ]
            
            return high_confidence_updates
            
        except Exception as e:
            logger.error(f"Error processing temporal updates: {e}")
            return []

    def validate_temporal_inference(self, original_fact: Dict[str, Any], 
                                  inferred_fact: Dict[str, Any]) -> bool:
        """Double-check temporal inference with strict logical validation."""
        prompt = self._format_long_prompt(
            role="You are a Temporal Logic Validator.",
            objective="Verify that a temporal inference is logically sound.",
            context="""We must validate temporal inferences with absolute logical certainty.
Only approve inferences that can be mathematically proven.""",
            instructions=f"""Validate this temporal inference:

Original Fact:
{original_fact}

Inferred Fact:
{inferred_fact}

Verification Steps:
1. Check temporal logic
   - Verify all calculations
   - Confirm logical steps
   - Validate assumptions

2. Edge Case Analysis
   - Check for leap years
   - Verify timezone implications
   - Consider calendar edge cases

3. Logical Proof
   - Provide step-by-step proof
   - Identify any assumptions
   - Verify completeness

Respond with:
- valid: boolean (true only if mathematically certain)
- proof: string (mathematical proof)
- edge_cases_considered: string[]""",
            examples=[{
                "input": "Age inference: 2024-01-01 birth date -> age 1 on 2025-01-01",
                "output": '{"valid": true, "proof": "1. Current date - birth date = 1 year exactly", "edge_cases_considered": ["leap year 2024", "timezone rollover"]}'
            }]
        )
        
        response = self._call_llm(prompt, temperature=0.1)
        
        try:
            result = json.loads(response)
            return result.get("valid", False)
        except Exception as e:
            logger.error(f"Error validating temporal inference: {e}")
            return False 