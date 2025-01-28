import logging
import torch
from typing import List, Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)

logger = logging.getLogger(__name__)

class BaseLLMAgent:
    """Base class for LLM Agents using local DeepSeek inference."""
    
    def __init__(self, 
                 model_path: str = "deepseek-ai/deepseek-coder-33b-instruct",
                 device: str = "cuda",
                 quantization: bool = True):
        self.device = device
        
        # Configure quantization for quantum hardware
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ) if quantization else None
        
        logger.info(f"Loading DeepSeek model {model_path} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        # Enable model parallelism for quantum acceleration
        if hasattr(self.model, "enable_model_parallel"):
            self.model.enable_model_parallel()
        
        logger.info("DeepSeek model loaded successfully")

    def _format_prompt(self, role: str, objective: str, instructions: str) -> str:
        """Format prompt using the Short Structured Framework."""
        return f"""# Role
{role}

# Objective
{objective}

# Instructions
{instructions}

Response:"""

    def _call_llm(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate response using local DeepSeek model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,  # DeepSeek recommended
                    top_k=50     # DeepSeek recommended
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.replace(prompt, "").strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""

class PlannerAgent(BaseLLMAgent):
    """Uses DeepSeek to prioritize URLs for crawling."""
    
    def rank_urls(self, candidate_urls: List[str]) -> List[str]:
        prompt = self._format_prompt(
            role="You are an expert web crawler prioritization agent.",
            objective="Rank the provided URLs based on their potential information value.",
            instructions=f"""Given these URLs, provide a priority order based on:
1. Likely information richness
2. Source credibility
3. Content freshness

URLs to rank:
{candidate_urls}

Output the URLs in priority order, one per line.
Do not include any additional text or explanations."""
        )
        
        response = self._call_llm(prompt, temperature=0.3)  # Lower temp for consistency
        
        # Parse response into list of URLs
        ranked_urls = [url.strip() for url in response.split("\n") if url.strip()]
        
        # Ensure we don't lose any URLs
        remaining = set(candidate_urls) - set(ranked_urls)
        ranked_urls.extend(list(remaining))
        
        return ranked_urls

class ComparisonAgent(BaseLLMAgent):
    """Uses DeepSeek to detect duplicate content."""
    
    def is_duplicate(self, new_text: str, existing_text: str) -> bool:
        prompt = self._format_prompt(
            role="You are a precise content comparison specialist.",
            objective="Determine if two texts contain essentially the same information.",
            instructions=f"""Compare these texts for semantic similarity:

Text 1:
{existing_text}

Text 2:
{new_text}

Respond with EXACTLY one word: either "DUPLICATE" or "UNIQUE".
Consider texts duplicate if they convey the same core information, even if worded differently."""
        )
        
        response = self._call_llm(prompt, max_length=256, temperature=0.1)
        return "duplicate" in response.lower()

class ResolutionAgent(BaseLLMAgent):
    """Uses DeepSeek to resolve knowledge graph conflicts."""
    
    def resolve_conflict(self, existing_fact: Dict[str, Any], new_fact: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._format_prompt(
            role="You are a knowledge graph conflict resolution expert.",
            objective="Resolve conflicts between competing facts in a knowledge graph.",
            instructions=f"""Analyze these potentially conflicting facts:

Existing fact: {existing_fact}
New fact: {new_fact}

Respond in strict JSON format with these fields:
- action: "overwrite", "keep_existing", or "store_disputed"
- reason: Brief explanation of your decision
- final_confidence: Float between 0.0 and 1.0

Base your decision on:
1. Source credibility
2. Information recency
3. Fact specificity
4. Confidence scores"""
        )
        
        response = self._call_llm(prompt, temperature=0.2)
        
        try:
            import json
            result = json.loads(response)
            return {
                "action": result.get("action", "keep_existing"),
                "reason": result.get("reason", "Failed to parse response"),
                "final_confidence": float(result.get("final_confidence", 0.5))
            }
        except Exception as e:
            logger.error(f"Error parsing resolution response: {e}")
            return {
                "action": "keep_existing",
                "reason": "Error parsing response",
                "final_confidence": 0.5
            } 