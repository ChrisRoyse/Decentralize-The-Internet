import spacy
from typing import List, Dict, Any
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ExtractedFact:
    entity1: str
    relation: str
    entity2: str
    confidence: float
    source: str

class EntityExtractor:
    """
    Extracts entities and relationships from text using spaCy NLP.
    """
    
    def __init__(self):
        # Load English language model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model successfully")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise
            
        # Define relationship patterns
        self.relation_patterns = [
            {
                "pattern": [{"POS": "PROPN"}, {"LEMMA": "acquire"}, {"POS": "PROPN"}],
                "relation": "acquired"
            },
            {
                "pattern": [{"POS": "PROPN"}, {"LEMMA": "invest"}, {"LOWER": "in"}, {"POS": "PROPN"}],
                "relation": "invested_in"
            },
            {
                "pattern": [{"POS": "PROPN"}, {"LEMMA": "be"}, {"POS": "DET"}, {"POS": "NOUN"}, {"LOWER": "of"}, {"POS": "PROPN"}],
                "relation": "part_of"
            }
            # Add more patterns as needed
        ]
        
        # Add patterns to matcher
        self.matcher = spacy.matcher.Matcher(self.nlp.vocab)
        for i, pattern in enumerate(self.relation_patterns):
            self.matcher.add(f"pattern_{i}", [pattern["pattern"]])

    def extract_entities(self, text: str, source: str = "unknown") -> List[Dict[str, Any]]:
        """
        Extract entity relationships from text.
        Returns list of dicts with {entity1, relation, entity2, confidence, source}
        """
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract facts using different methods
            facts = []
            facts.extend(self._extract_from_patterns(doc, source))
            facts.extend(self._extract_from_dependencies(doc, source))
            
            return facts
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []

    def _extract_from_patterns(self, doc, source: str) -> List[Dict[str, Any]]:
        """Extract facts using predefined patterns"""
        facts = []
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            pattern_name = self.nlp.vocab.strings[match_id]
            pattern_index = int(pattern_name.split("_")[1])
            relation = self.relation_patterns[pattern_index]["relation"]
            
            span = doc[start:end]
            entities = [token for token in span if token.pos_ == "PROPN"]
            
            if len(entities) >= 2:
                facts.append({
                    "entity1": entities[0].text,
                    "relation": relation,
                    "entity2": entities[-1].text,
                    "confidence": 0.8,  # Pattern-based confidence
                    "source": source
                })
        
        return facts

    def _extract_from_dependencies(self, doc, source: str) -> List[Dict[str, Any]]:
        """Extract facts using dependency parsing"""
        facts = []
        
        for token in doc:
            # Look for subject-verb-object patterns
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                subject = token
                verb = token.head
                
                # Find object
                for child in verb.children:
                    if child.dep_ in ["dobj", "pobj"]:
                        # Only consider named entities
                        if subject.ent_type_ and child.ent_type_:
                            facts.append({
                                "entity1": subject.text,
                                "relation": verb.lemma_,
                                "entity2": child.text,
                                "confidence": 0.6,  # Dependency-based confidence
                                "source": source
                            })
        
        return facts

    def _clean_entity(self, text: str) -> str:
        """Clean and normalize entity text"""
        return text.strip()

    def _get_relation_confidence(self, relation: str, context_size: int) -> float:
        """
        Calculate confidence score for a relation based on various factors
        """
        # Base confidence
        confidence = 0.5
        
        # Adjust based on context size
        if context_size > 100:
            confidence += 0.2
            
        # Adjust based on relation type
        if relation in ["acquired", "invested_in", "part_of"]:
            confidence += 0.2
            
        return min(confidence, 1.0) 