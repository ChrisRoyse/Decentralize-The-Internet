import numpy as np
import hashlib
import uuid
from typing import Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChunkInfo:
    chunk_id: str
    embedding: np.ndarray
    signature: str

class DeduplicationManager:
    """Handles document deduplication using embeddings and LSH"""
    
    def __init__(self, config, embedding_model):
        self.config = config
        self.embedding_model = embedding_model
        self.similarity_threshold = config["dedup"]["similarity_threshold"]
        
        # Store chunk info for deduplication
        self.chunks: dict[str, ChunkInfo] = {}  # signature -> ChunkInfo
        
    def process_chunk(self, text: str) -> Tuple[bool, Optional[str], Optional[bytes]]:
        """
        Process a text chunk and determine if it's new.
        Returns: (is_new, chunk_id, compressed_data)
        """
        try:
            # Generate embedding
            embedding = self.embedding_model.get_embedding(text)
            
            # Generate signature from embedding
            signature = self._generate_signature(embedding)
            
            # Check if we've seen this signature
            if signature in self.chunks:
                logger.debug(f"Duplicate chunk found with signature {signature}")
                return False, self.chunks[signature].chunk_id, None
                
            # New chunk - store it
            chunk_id = str(uuid.uuid4())
            self.chunks[signature] = ChunkInfo(
                chunk_id=chunk_id,
                embedding=embedding,
                signature=signature
            )
            
            # Compress the text (could be more sophisticated)
            compressed = text.encode('utf-8')
            
            logger.debug(f"New chunk stored with ID {chunk_id}")
            return True, chunk_id, compressed
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return False, None, None
            
    def _generate_signature(self, embedding: np.ndarray) -> str:
        """Generate a signature from an embedding for quick comparison"""
        # Simple approach: round embedding values and hash
        rounded = np.round(embedding, decimals=2)
        signature_input = ",".join(str(x) for x in rounded)
        return hashlib.md5(signature_input.encode('utf-8')).hexdigest()
        
    def check_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        ) 