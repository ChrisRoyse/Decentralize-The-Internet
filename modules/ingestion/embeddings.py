import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Union
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Generates embeddings for text using a transformer model"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Initializing embedding model: {model_name}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Cache for storing embeddings
        self.cache = {}
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string"""
        # Check cache first
        if text in self.cache:
            return self.cache[text]
            
        try:
            # Tokenize and prepare input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Convert to numpy and normalize
            embedding = embedding[0].cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding)
            
            # Cache result
            self.cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.model.config.hidden_size) 