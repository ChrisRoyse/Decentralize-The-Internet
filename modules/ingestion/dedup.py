import logging
import uuid
from typing import Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChunkInfo:
    chunk_id: str
    text: str

class DeduplicationManager:
    """Uses LLM-based comparison for deduplication."""
    
    def __init__(self, config):
        self.config = config
        self.chunks = {}  # chunk_id -> ChunkInfo

    def process_chunk_llm(self, text: str, llm_comparer) -> bool:
        """Return True if chunk is new, False if duplicate."""
        # Check against existing chunks
        for chunk_info in self.chunks.values():
            if llm_comparer.is_duplicate(new_text=text, existing_text=chunk_info.text):
                logger.debug(f"Duplicate chunk found by LLM: {chunk_info.chunk_id}")
                return False

        # Store new chunk
        chunk_id = str(uuid.uuid4())
        self.chunks[chunk_id] = ChunkInfo(chunk_id=chunk_id, text=text)
        logger.debug(f"New chunk stored with ID {chunk_id}")
        return True 