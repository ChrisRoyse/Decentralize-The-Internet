import pytest
import asyncio
import yaml
from pathlib import Path
from modules.security.access_control import AccessControl
from modules.security.encryption import EncryptionManager, SecureMessage
from modules.messaging.message_bus import MessageBus
from modules.crawler.frontier_manager import FrontierManager

@pytest.fixture
def config():
    """Load test configuration"""
    config_path = Path(__file__).parent / "test_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def access_control(config):
    """Initialize access control"""
    return AccessControl(config)

@pytest.fixture
def encryption_manager(config):
    """Initialize encryption manager"""
    return EncryptionManager(config)

@pytest.fixture
def message_bus(config):
    """Initialize message bus"""
    bus = MessageBus(config)
    yield bus
    bus.close()

class TestSecurity:
    def test_user_roles(self, access_control):
        """Test user role management"""
        # Add test user
        result = access_control.add_user(
            user_id="test1",
            username="testuser",
            roles={"crawler"}
        )
        assert result is True
        
        # Check permissions
        assert access_control.check_permission("test1", "read:urls") is True
        assert access_control.check_permission("test1", "manage:users") is False

    def test_token_lifecycle(self, access_control):
        """Test JWT token generation and validation"""
        # Add test user
        access_control.add_user(
            user_id="test2",
            username="testuser2",
            roles={"analyst"}
        )
        
        # Generate token
        token = access_control.generate_token("test2")
        assert token is not None
        
        # Validate token
        payload = access_control.validate_token(token)
        assert payload is not None
        assert payload["user_id"] == "test2"
        assert "analyst" in payload["roles"]

class TestMessaging:
    @pytest.mark.asyncio
    async def test_pub_sub(self, message_bus):
        """Test basic publish/subscribe functionality"""
        received_messages = []
        
        def callback(message):
            received_messages.append(message)
        
        # Subscribe to test topic
        message_bus.subscribe("test_topic", callback)
        
        # Publish message
        test_data = {"key": "value"}
        message_bus.publish("test_topic", test_data)
        
        # Wait briefly for message processing
        await asyncio.sleep(0.1)
        
        assert len(received_messages) == 1
        assert received_messages[0]["key"] == "value"

class TestCrawler:
    def test_frontier_management(self, config, message_bus):
        """Test URL frontier functionality"""
        frontier = FrontierManager(config, message_bus)
        
        # Add test URLs
        test_urls = [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3"
        ]
        frontier.add_urls(test_urls)
        
        # Get batch
        batch = frontier.get_next_batch()
        assert len(batch) > 0
        assert all(url in test_urls for url in batch)

@pytest.mark.asyncio
async def test_system_startup(config):
    """Test that the system can start up and shut down cleanly"""
    from main import DecentralizedPipeline
    
    pipeline = DecentralizedPipeline()
    await pipeline.initialize()
    
    # Start pipeline in background task
    task = asyncio.create_task(pipeline.start())
    
    # Let it run briefly
    await asyncio.sleep(2)
    
    # Signal shutdown
    pipeline.running = False
    
    # Wait for shutdown
    await task 