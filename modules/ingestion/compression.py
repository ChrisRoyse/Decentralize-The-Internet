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
    Attempts GPU-accelerated compression using CuPy if available, 
    otherwise falls back to CPU-based compression.
    """
    
    def __init__(self):
        super().__init__()
        self.torch_gpu_available = False
        self.cupy_available = False
        
        try:
            import torch
            if torch.cuda.is_available():
                self.torch_gpu_available = True
            else:
                logger.info("torch is installed, but CUDA is not available. GPU compression via torch not possible.")
        except ImportError:
            logger.warning("torch not found. CUDA GPU availability check skipped. GPU compression via torch not possible.")
            
        try:
            import cupy
            # Perform a simple CuPy operation to ensure it's working with the current CUDA environment
            cupy.array([1]) 
            self.cupy_available = True
        except ImportError:
            logger.warning("cupy not found. CuPy GPU acceleration for compression unavailable.")
        except cupy.cuda.runtime.CUDARuntimeError as e:
            logger.warning(f"cupy found, but CUDA runtime error occurred: {e}. CuPy GPU acceleration for compression likely unavailable.")
            self.cupy_available = False # Ensure it's false if cupy init fails
        except Exception as e: # Catch any other cupy import/init errors
            logger.warning(f"cupy import or initialization failed with an unexpected error: {e}. CuPy GPU acceleration for compression unavailable.")
            self.cupy_available = False


        if self.torch_gpu_available and self.cupy_available:
            logger.info("GPU compression potentially available (CUDA detected by torch, CuPy imported and initialized successfully).")
        elif self.torch_gpu_available and not self.cupy_available:
            logger.info("CUDA detected by torch, but CuPy is not available or not functional. GPU compression with CuPy disabled. Consider installing/configuring CuPy.")
        elif not self.torch_gpu_available and self.cupy_available:
            # This case implies cupy might work if a CUDA toolkit is present but torch didn't see the GPU.
            logger.info("CuPy imported and initialized, but CUDA not detected by torch. GPU compression with CuPy might still be attempted if CUDA environment is correctly set up for CuPy.")
        else: # Neither torch_gpu_available nor cupy_available
            logger.warning("Neither CUDA (via torch) nor CuPy seem available/functional. Falling back to CPU compression.")
            
    def compress(self, data: bytes) -> bytes:
        # Check both torch_gpu_available (as a general CUDA health check) and cupy_available
        if self.torch_gpu_available and self.cupy_available:
            try:
                logger.debug("Attempting GPU compression path with CuPy.")
                
                # Convert bytes to CuPy array.
                # np.frombuffer creates a view, then cupy.array copies it to GPU memory.
                cupy_array = cupy.array(np.frombuffer(data, dtype=np.uint8))
                
                # --- Placeholder for actual CuPy-based compression ---
                # CuPy itself does not provide a direct zlib.compress equivalent.
                # For true GPU-accelerated compression, one would typically use:
                # 1. NVIDIA's nvCOMP library (e.g., through Python bindings or custom C++/CUDA code).
                # 2. Other GPU-accelerated compression libraries (e.g., for Zstandard, LZ4) 
                #    that can operate on CuPy arrays or raw GPU pointers.
                # Since implementing these is beyond the current scope, we log the limitation
                # and use zlib on CPU-side data obtained from GPU (via cupy_array.get()) 
                # as a simulation of the data path.
                logger.info("Simulating CuPy compression data path: data successfully moved to GPU (CuPy array created). "
                            "No direct CuPy zlib equivalent integrated. Using CPU zlib on data fetched back from GPU as a placeholder for a true GPU compression routine.")
                
                # Simulate: Get data back from GPU and compress using CPU zlib
                # In a real scenario, this would be replaced by a GPU compression call:
                # e.g., compressed_cupy_array = some_gpu_compress_lib.compress(cupy_array, level=self.compression_level)
                # compressed_data_bytes = compressed_cupy_array.tobytes()
                
                data_to_compress_on_cpu = cupy_array.get() # Moves data from GPU to CPU memory
                compressed_data_on_cpu = zlib.compress(data_to_compress_on_cpu, level=self.compression_level)
                
                logger.debug("GPU (simulated with CuPy path using CPU zlib) compression step completed.")
                return compressed_data_on_cpu
            except ImportError:
                # This handles a very unlikely case where cupy was unimported between __init__ and compress
                logger.error("CuPy import error during compression attempt. This should not happen if __init__ checks passed. Falling back to CPU.")
                return super().compress(data)
            except cupy.cuda.runtime.CUDARuntimeError as e:
                logger.error(f"CuPy CUDA runtime error during compression: {e}. Falling back to CPU compression.")
                # Optionally, disable cupy for future attempts in this session:
                # self.cupy_available = False 
                return super().compress(data)
            except Exception as e:
                logger.error(f"GPU compression with CuPy data path failed: {e}. Falling back to CPU compression.")
                return super().compress(data)
        else:
            if not self.torch_gpu_available:
                logger.info("CUDA (via torch) not available. Using CPU compression.")
            elif not self.cupy_available:
                logger.info("CuPy not available or not functional. Using CPU compression.")
            # No need for a generic else here, the specific log indicates the fallback.
            return super().compress(data)

    def decompress(self, data: bytes) -> Optional[bytes]:
        # For now, decompression remains on CPU.
        # If a CuPy-based compression was used, its corresponding decompressor would be needed here.
        logger.debug("Using CPU zlib for decompression.")
        return super().decompress(data)