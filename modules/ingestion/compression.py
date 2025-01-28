import zlib
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

class Compressor:
    """
    Handles data compression with multiple strategies.
    Could be extended to use GPU acceleration or specialized compression.
    """
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
        
    def compress(self, data: bytes) -> bytes:
        """Compress data using zlib"""
        try:
            return zlib.compress(data, level=self.compression_level)
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return data
            
    def decompress(self, data: bytes) -> Optional[bytes]:
        """Decompress zlib-compressed data"""
        try:
            return zlib.decompress(data)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return None
            
    def compress_text(self, text: str) -> bytes:
        """Compress text data"""
        return self.compress(text.encode('utf-8'))
        
    def decompress_text(self, data: bytes) -> Optional[str]:
        """Decompress to text"""
        decompressed = self.decompress(data)
        if decompressed is not None:
            try:
                return decompressed.decode('utf-8')
            except UnicodeDecodeError as e:
                logger.error(f"Failed to decode decompressed data: {e}")
        return None

class GPUCompressor(Compressor):
    """
    Placeholder for GPU-accelerated compression.
    Would use CUDA or similar in production.
    """
    
    def __init__(self):
        super().__init__()
        self.gpu_available = False
        try:
            # Check for GPU availability
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                logger.info("GPU compression enabled")
        except ImportError:
            logger.warning("torch not available, falling back to CPU compression")
            
    def compress(self, data: bytes) -> bytes:
        if self.gpu_available:
            # Placeholder for GPU compression
            # In reality, would use CUDA kernels or GPU libraries
            return super().compress(data)
        return super().compress(data) 