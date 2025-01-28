from typing import Dict, Literal, Tuple
import logging
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

ResolutionType = Literal["keep_existing", "overwrite", "store_disputed"]

@dataclass
class ResolutionResult:
    action: ResolutionType
    final_confidence: float
    reason: str = ""

class ConflictResolver:
    """
    Handles conflicting edges in the knowledge graph based on confidence scores,
    domain trust, and explicit bias configuration.
    """
    
    def __init__(self, config):
        self.config = config
        self.conflict_policy = config["knowledge_graph"].get("conflict_policy", "store_disputed")
        
        # Load trust scores and bias overrides
        kg_config = config["knowledge_graph"]
        self.domain_trust_scores = kg_config.get("domain_trust_scores", {})
        self.bias_overrides = kg_config.get("bias_overrides", {})
        
        # Extract official sources and domain boosts
        self.official_sources = self.bias_overrides.get("official_sources", [])
        self.domain_boost = self.bias_overrides.get("domain_boost", {})
        
        logger.info(f"Initialized ConflictResolver with {len(self.official_sources)} official sources")

    def resolve_conflict(self, existing_edge: Dict, new_edge: Dict) -> ResolutionResult:
        """
        Determine how to handle conflicting edges based on confidence, trust, and bias.
        Returns a ResolutionResult with action and final confidence.
        """
        if self.conflict_policy == "store_disputed":
            return ResolutionResult(
                action="store_disputed",
                final_confidence=self._calculate_final_confidence(new_edge),
                reason="Policy is set to store all disputed facts"
            )

        # Calculate final confidence scores
        existing_final_conf = self._calculate_final_confidence(existing_edge)
        new_final_conf = self._calculate_final_confidence(new_edge)

        # Check for official truth override
        if existing_edge.get("official_truth"):
            if not self._is_official_source(new_edge.get("source", "")):
                return ResolutionResult(
                    action="keep_existing",
                    final_confidence=existing_final_conf,
                    reason="Existing edge is marked as official truth"
                )

        # Compare final confidence scores with threshold
        threshold = 1.2  # Require 20% higher confidence to overwrite
        if new_final_conf > existing_final_conf * threshold:
            return ResolutionResult(
                action="overwrite",
                final_confidence=new_final_conf,
                reason=f"New confidence ({new_final_conf:.2f}) significantly higher than existing ({existing_final_conf:.2f})"
            )
        else:
            return ResolutionResult(
                action="keep_existing",
                final_confidence=existing_final_conf,
                reason=f"New confidence ({new_final_conf:.2f}) not significantly higher than existing ({existing_final_conf:.2f})"
            )

    def _calculate_final_confidence(self, edge: Dict) -> float:
        """Calculate final confidence score incorporating all factors"""
        # Start with base confidence
        confidence = edge.get("confidence", 0.5)
        source = edge.get("source", "").lower()

        # Apply domain trust
        confidence *= self._get_domain_trust(source)

        # Apply bias overrides
        confidence = self._apply_bias_overrides(source, confidence)

        # Cap at 1.0 unless it's an official source
        if not self._is_official_source(source):
            confidence = min(confidence, 1.0)

        return confidence

    def _get_domain_trust(self, source: str) -> float:
        """Get trust score for a domain"""
        try:
            domain = self._extract_domain(source)
            
            # Check each known domain pattern
            for known_domain, trust_score in self.domain_trust_scores.items():
                if known_domain in domain:
                    return trust_score
                    
            return 0.5  # Default trust score
            
        except Exception as e:
            logger.warning(f"Error processing domain trust for {source}: {e}")
            return 0.5

    def _apply_bias_overrides(self, source: str, confidence: float) -> float:
        """Apply additional bias multipliers from configuration"""
        try:
            domain = self._extract_domain(source)
            original_confidence = confidence

            # Apply domain-specific boost
            for boost_domain, multiplier in self.domain_boost.items():
                if boost_domain in domain:
                    confidence *= multiplier
                    logger.info(
                        f"Applied domain boost: {boost_domain} ({multiplier}x) "
                        f"to {domain} ({original_confidence:.2f} -> {confidence:.2f})"
                    )
                    break  # Only apply highest boost

            # Apply official source multiplier
            if self._is_official_source(domain):
                original_confidence = confidence
                confidence *= 2.0
                logger.info(
                    f"Applied official source boost to {domain} "
                    f"({original_confidence:.2f} -> {confidence:.2f})"
                )

            return confidence

        except Exception as e:
            logger.warning(f"Error applying bias overrides for {source}: {e}")
            return confidence

    def _is_official_source(self, source: str) -> bool:
        """Check if source is in the official sources list"""
        domain = self._extract_domain(source)
        return any(off_source in domain for off_source in self.official_sources)

    def _extract_domain(self, source: str) -> str:
        """Extract domain from source URL or return source if not a URL"""
        try:
            parsed = urlparse(source)
            if parsed.netloc:
                return parsed.netloc.lower()
            return source.lower()
        except Exception:
            return source.lower() 