import asyncio
import logging
import yaml
from pathlib import Path
import signal
import sys
from typing import Optional
import structlog

# Import our modules
from modules.crawler.crawler_agent import CrawlerAgent
from modules.crawler.frontier_manager import FrontierManager
from modules.ingestion.dedup import DeduplicationManager
from modules.ingestion.embeddings import EmbeddingModel
from modules.knowledge_graph.graph_manager import KnowledgeGraphManager
from modules.knowledge_graph.entity_extraction import EntityExtractor
from modules.knowledge_graph.conflict_resolution import ConflictResolver
from modules.messaging.message_bus import MessageBus
from modules.messaging.p2p_node import P2PNode
from modules.security.encryption import EncryptionManager, SecureMessage
from modules.security.access_control import AccessControl
from modules.config.config_loader import ConfigLoader
from modules.monitoring.metrics_server import MetricsServer
from modules.logging.log_config import configure_logging

# Configure logging
logger = configure_logging()

class DecentralizedPipeline:
    """Main application class that coordinates all components"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.running = False
        self.components = {}
        self.logger = structlog.get_logger()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    async def initialize(self):
        """Initialize all system components"""
        try:
            self.logger.info("initializing_pipeline",
                            node_id=self.config["node"]["id"],
                            role=self.config["node"]["role"])
            
            # Initialize security components first
            self.components["encryption"] = EncryptionManager(self.config)
            self.components["access_control"] = AccessControl(self.config)
            
            # Initialize messaging
            self.components["message_bus"] = MessageBus(self.config)
            self.components["p2p_node"] = P2PNode(
                self.config,
                self.components["message_bus"]
            )
            
            # Initialize knowledge graph components
            self.components["graph_manager"] = KnowledgeGraphManager(self.config)
            self.components["entity_extractor"] = EntityExtractor()
            self.components["conflict_resolver"] = ConflictResolver(self.config)
            
            # Initialize ingestion components
            self.components["embedding_model"] = EmbeddingModel()
            self.components["dedup_manager"] = DeduplicationManager(
                self.config,
                self.components["embedding_model"]
            )
            
            # Initialize crawler components
            self.components["frontier_manager"] = FrontierManager(
                self.config,
                self.components["message_bus"]
            )
            
            self.components["crawler"] = CrawlerAgent(
                config=self.config,
                frontier_manager=self.components["frontier_manager"],
                deduper=self.components["dedup_manager"],
                compressor=None,  # Optional compression
                entity_extractor=self.components["entity_extractor"],
                kg_manager=self.components["graph_manager"],
                conflict_resolver=self.components["conflict_resolver"],
                message_bus=self.components["message_bus"],
                access_control=self.components["access_control"]
            )
            
            # Start metrics server
            self.metrics_server = MetricsServer()
            await self.metrics_server.start()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    async def start(self):
        """Start the pipeline"""
        try:
            self.running = True
            logger.info("Starting decentralized pipeline...")
            
            # Register signal handlers
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, self._signal_handler)
            
            # Main processing loop
            while self.running:
                try:
                    # Run crawler cycle
                    await self.components["crawler"].crawl_cycle()
                    
                    # Sleep briefly between cycles
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error in main processing loop: {e}")
                    await asyncio.sleep(5)  # Back off on error
                    
        except Exception as e:
            logger.error(f"Error starting pipeline: {e}")
            raise
        finally:
            await self.shutdown()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.running = False

    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("Shutting down pipeline...")
        
        try:
            # Close knowledge graph connection
            if "graph_manager" in self.components:
                self.components["graph_manager"].close()
            
            # Close message bus
            if "message_bus" in self.components:
                self.components["message_bus"].close()
            
            # Stop metrics server
            await self.metrics_server.stop()
            
            logger.info("Shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

async def main():
    """Application entry point"""
    # Load config with secrets
    config = ConfigLoader.load_config()
    
    pipeline = DecentralizedPipeline()
    
    try:
        await pipeline.initialize()
        await pipeline.start()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 