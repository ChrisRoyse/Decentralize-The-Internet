from typing import Dict, Any, List
from ..agents.llm_agent import BaseLLMAgent
import logging

logger = logging.getLogger(__name__)

class GroundTruthAgent(BaseLLMAgent):
    """Specialized agent for managing ground truth in knowledge graph."""
    
    def evaluate_ground_truth_candidate(self, fact: Dict[str, Any], 
                                      evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate if a fact should be marked as ground truth."""
        prompt = self._format_long_prompt(
            role="""You are the Ground Truth Authority responsible for maintaining 
absolute truth in the knowledge graph.""",
            objective="""Evaluate whether a proposed fact meets the strict criteria 
to be marked as ground truth.""",
            context="""Ground truth represents facts that are considered absolutely true 
and take precedence over conflicting information. This designation requires 
extensive evidence and careful consideration.""",
            instructions=f"""Evaluate this fact for ground truth status:

Proposed Fact:
{fact}

Supporting Evidence:
{evidence}

Evaluate based on:
1. Evidence Quality
   - Source reliability
   - Verification status
   - Corroboration level

2. Fact Properties
   - Clarity and specificity
   - Temporal stability
   - Logical consistency

3. Confidence Assessment
   - Evidence strength
   - Verification level
   - Expert consensus

Respond in JSON format with:
- mark_as_ground_truth: boolean
- confidence_score: float
- reasoning: string
- verification_notes: string[]
- required_additional_evidence: string[]""",
            examples=[{
                "input": "Fact: Google acquired YouTube in 2006",
                "output": '{"mark_as_ground_truth": true, "confidence_score": 1.0, "reasoning": "Multiple reliable sources, official documentation, consistent historical record"}'
            }]
        )
        
        response = self._call_llm(prompt, temperature=0.1)  # Very low temp for consistency
        
        try:
            import json
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error evaluating ground truth: {e}")
            return {
                "mark_as_ground_truth": False,
                "confidence_score": 0.0,
                "reasoning": "Failed to evaluate due to error",
                "verification_notes": ["Evaluation failed"],
                "required_additional_evidence": ["Complete re-evaluation required"]
            } 