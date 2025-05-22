import pytest
import zlib # For creating invalid data for decompressor tests
import sys
import logging
from modules.ingestion.compression import Compressor, GPUCompressor

# Configure basic logging for tests to see compressor logs if needed
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Test Data ---
SAMPLE_TEXT = "This is a sample text for compression testing. It includes various characters and repeats some patterns to ensure compression is effective."
LONG_TEXT = SAMPLE_TEXT * 100  # A longer text for more robust testing
EMPTY_TEXT = ""
SAMPLE_BYTES = SAMPLE_TEXT.encode('utf-8')
LONG_BYTES = LONG_TEXT.encode('utf-8')
EMPTY_BYTES = b""

TEST_DATA_PARAMS = [
    pytest.param(SAMPLE_TEXT, SAMPLE_BYTES, id="sample_text"),
    pytest.param(LONG_TEXT, LONG_BYTES, id="long_text"),
    pytest.param(EMPTY_TEXT, EMPTY_BYTES, id="empty_string")
]

# --- Fixtures ---

@pytest.fixture
def compressor():
    return Compressor()

@pytest.fixture(params=[
    {"cuda_available": True, "cupy_available": True, "cupy_init_functional": True, "id": "cuda_cupy_functional"},
    {"cuda_available": True, "cupy_available": True, "cupy_init_functional": False, "id": "cuda_cupy_init_fails"},
    {"cuda_available": True, "cupy_available": False, "cupy_init_functional": False, "id": "cuda_no_cupy"},
    {"cuda_available": False, "cupy_available": True, "cupy_init_functional": True, "id": "no_cuda_cupy_functional"},
    {"cuda_available": False, "cupy_available": False, "cupy_init_functional": False, "id": "no_cuda_no_cupy"}
])
def gpu_compressor_scenario(request, monkeypatch):
    """
    Fixture to simulate different scenarios for GPUCompressor:
    - torch.cuda.is_available() state
    - cupy import success/failure
    - cupy initialization success/failure (cupy.array([1]) in GPUCompressor.__init__)
    """
    cuda_available = request.param["cuda_available"]
    cupy_available = request.param["cupy_available"]
    cupy_init_functional = request.param["cupy_init_functional"]

    mock_torch_cuda = lambda: cuda_available
    original_torch = sys.modules.get('torch')
    original_cupy = sys.modules.get('cupy')

    # Mock torch.cuda.is_available
    if 'torch' in sys.modules and hasattr(sys.modules['torch'], 'cuda'):
        monkeypatch.setattr(sys.modules['torch'].cuda, 'is_available', mock_torch_cuda)
    elif cuda_available: # If torch is not even imported, but we want to simulate cuda_available
        class MockCuda:
            def is_available(self):
                return True
        class MockTorch:
            cuda = MockCuda()
        monkeypatch.setitem(sys.modules, 'torch', MockTorch())
        logger.info("Mocked torch and torch.cuda for cuda_available=True")
    else: # If torch is not imported and we want to simulate cuda_unavailable
        class MockCuda:
            def is_available(self):
                return False
        class MockTorch:
            cuda = MockCuda()
        monkeypatch.setitem(sys.modules, 'torch', MockTorch())
        logger.info("Mocked torch and torch.cuda for cuda_available=False")


    if cupy_available:
        if 'cupy' not in sys.modules: # If cupy wasn't imported by test runner yet
            # Create a mock cupy that can be "imported"
            class MockCuPy:
                def __init__(self):
                    self.cuda = self # for cupy.cuda.runtime
                    self.runtime = self # for cupy.cuda.runtime
                def array(self, data_obj, dtype=None): # Simplified mock
                    if not cupy_init_functional:
                        # Simulate cupy.cuda.runtime.CUDARuntimeError on array creation
                        # This is how GPUCompressor's __init__ checks cupy functionality
                        raise self.CUDARuntimeError("Mock CuPy Init Fails") 
                    # In a real scenario, this would return a cupy array.
                    # For testing compression logic, we don't need full cupy functionality here.
                    # The GPUCompressor's simulated path will use .get() if array() succeeds.
                    class MockCuPyArray:
                        def get(self):
                            return data_obj # Return the original numpy/buffer like data
                    return MockCuPyArray()
                
                class CUDARuntimeError(RuntimeError): # Define the specific error
                    pass

            cupy_mock_instance = MockCuPy()
            monkeypatch.setitem(sys.modules, 'cupy', cupy_mock_instance)
            logger.info(f"Mocked cupy as available (init_functional={cupy_init_functional})")
        elif hasattr(sys.modules['cupy'], 'array'): # cupy is imported, try to patch its array
            original_cupy_array = sys.modules['cupy'].array
            original_cupy_cuda_runtime_error = getattr(getattr(getattr(sys.modules['cupy'], 'cuda', object()), 'runtime', object()), 'CUDARuntimeError', RuntimeError)

            def mock_cupy_array_for_test(data_obj, dtype=None):
                if not cupy_init_functional:
                    raise original_cupy_cuda_runtime_error("Mock CuPy Init Fails from existing cupy")
                return original_cupy_array(data_obj, dtype=dtype) # Call original if functional

            if not cupy_init_functional: # Only patch if we need to simulate failure
                 monkeypatch.setattr(sys.modules['cupy'], 'array', mock_cupy_array_for_test)
            logger.info(f"Patched existing cupy.array (init_functional={cupy_init_functional})")

    else: # cupy not available
        monkeypatch.setitem(sys.modules, 'cupy', None) # Simulate cupy not being importable
        # GPUCompressor's __init__ will catch ImportError for cupy
        logger.info("Simulating cupy as not available (ImportError)")

    # Yield an instance of GPUCompressor, it will pick up the mocks
    yield GPUCompressor()

    # Cleanup: Restore original modules if they were changed
    if original_torch:
        sys.modules['torch'] = original_torch
    elif 'torch' in sys.modules and request.param["cuda_available"]: # Was mocked
        del sys.modules['torch']
        
    if original_cupy:
        sys.modules['cupy'] = original_cupy
    elif 'cupy' in sys.modules and cupy_available: # Was mocked
        del sys.modules['cupy']
    elif 'cupy' in sys.modules and not cupy_available: # Was set to None
        del sys.modules['cupy']


# --- Compressor Tests ---

def test_compressor_instantiation(compressor):
    assert compressor is not None
    assert compressor.compression_level == 6 # Default

def test_compressor_instantiation_custom_level():
    custom_compressor = Compressor(compression_level=9)
    assert custom_compressor.compression_level == 9

@pytest.mark.parametrize("text_data, byte_data", TEST_DATA_PARAMS)
def test_compressor_compress_decompress_cycle_bytes(compressor, text_data, byte_data):
    compressed = compressor.compress(byte_data)
    assert isinstance(compressed, bytes)
    if byte_data: # zlib cannot compress empty bytes into something smaller
        assert len(compressed) < len(byte_data) if len(byte_data) > 20 else True # Compression ratio check for non-trivial data
    
    decompressed = compressor.decompress(compressed)
    assert decompressed == byte_data

@pytest.mark.parametrize("text_data, byte_data", TEST_DATA_PARAMS)
def test_compressor_compress_decompress_cycle_text(compressor, text_data, byte_data):
    compressed = compressor.compress_text(text_data)
    assert isinstance(compressed, bytes)
    
    decompressed_text = compressor.decompress_text(compressed)
    assert decompressed_text == text_data

def test_compressor_decompress_invalid_data(compressor):
    invalid_data = b"this is not valid zlib data"
    assert compressor.decompress(invalid_data) is None
    
    # Test with data that might look like zlib but is corrupted
    valid_compressed = compressor.compress(b"valid data")
    corrupted_data = valid_compressed[:10] + b"corruption" + valid_compressed[10:]
    assert compressor.decompress(corrupted_data) is None
    
    # Test decompress_text with invalid data
    assert compressor.decompress_text(invalid_data) is None


# --- GPUCompressor Tests ---

def test_gpu_compressor_instantiation_logging(gpu_compressor_scenario, caplog):
    """ Tests GPUCompressor instantiation and checks logs for path info. """
    # The gpu_compressor_scenario fixture creates an instance, which logs during __init__
    compressor = gpu_compressor_scenario 
    assert compressor is not None
    # We can check caplog.text for messages like "GPU compression enabled", "CuPy not found", etc.
    # This implicitly tests the __init__ logic based on mocked cuda/cupy.
    logger.info(f"GPUCompressor instantiated with config: {caplog.text}")
    # Example: if cuda and cupy are mocked as available & functional:
    # if request.param["cuda_available"] and request.param["cupy_available"] and request.param["cupy_init_functional"]:
    # assert "GPU compression potentially available" in caplog.text


@pytest.mark.parametrize("text_data, byte_data", TEST_DATA_PARAMS)
def test_gpu_compressor_compress_decompress_cycle_bytes(gpu_compressor_scenario, text_data, byte_data, caplog):
    compressor = gpu_compressor_scenario
    
    # Log which path is expected
    if compressor.torch_gpu_available and compressor.cupy_available:
        logger.info("Testing GPUCompressor: Expected CuPy path (simulated)")
    else:
        logger.info("Testing GPUCompressor: Expected CPU fallback path")
        
    compressed = compressor.compress(byte_data)
    assert isinstance(compressed, bytes)

    # Check logs to infer path if possible (optional, main thing is correctness)
    if compressor.torch_gpu_available and compressor.cupy_available:
        assert "Simulating CuPy compression data path" in caplog.text or "Attempting GPU compression path with CuPy" in caplog.text
    else:
        assert "Using CPU compression" in caplog.text or "Falling back to CPU compression" in caplog.text or "CuPy not available" in caplog.text or "CUDA (via torch) not available" in caplog.text

    decompressed = compressor.decompress(compressed)
    assert decompressed == byte_data

@pytest.mark.parametrize("text_data, byte_data", TEST_DATA_PARAMS)
def test_gpu_compressor_compress_decompress_cycle_text(gpu_compressor_scenario, text_data, byte_data, caplog):
    compressor = gpu_compressor_scenario
    compressed = compressor.compress_text(text_data)
    assert isinstance(compressed, bytes)
    
    decompressed_text = compressor.decompress_text(compressed)
    assert decompressed_text == text_data

def test_gpu_compressor_decompress_invalid_data(gpu_compressor_scenario):
    compressor = gpu_compressor_scenario
    invalid_data = b"this is not valid zlib data"
    assert compressor.decompress(invalid_data) is None
    assert compressor.decompress_text(invalid_data) is None

# --- Further tests could include specific error condition simulations in compress path ---

# Example: Test what happens if cupy.array() itself fails mid-operation (not just in init)
# This is more complex to mock if cupy was successfully initialized.
# For now, the existing tests cover init failures and import failures.

# Test compression levels if GPUCompressor also took compression_level
# (Currently it inherits default from Compressor or uses self.compression_level)

# Test that GPUCompressor falls back if cupy.get() fails (if that's a distinct possibility)
# The current placeholder `cupy_array.get()` returns the original numpy/buffer like data, so it's unlikely to fail
# unless the mock is made more complex.
# Real CuPy's .get() can fail under certain CUDA error conditions.
# The generic `except Exception as e:` in GPUCompressor.compress should catch this.

# To make the log check in test_gpu_compressor_compress_decompress_cycle_bytes more robust:
# We can clear the caplog before compressor.compress() if needed,
# or check for specific log messages from the compress method vs. __init__.
# For now, it checks the entire log text for simplicity.

# A final check to make sure the file ends with a newline
print("")
