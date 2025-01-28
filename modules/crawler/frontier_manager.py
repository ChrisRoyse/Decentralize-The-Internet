import random
from typing import List, Set
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class FrontierManager:
    """Manages the queue of URLs to be crawled"""
    
    def __init__(self, config, message_bus):
        self.config = config
        self.message_bus = message_bus
        self.local_frontier: Set[str] = set()
        self.visited_urls: Set[str] = set()
        self.batch_size = config["crawler"].get("max_pages_per_cycle", 10)
        
        # Subscribe to peer messages about new URLs
        self.message_bus.subscribe("new_urls", self._handle_peer_urls)
        
    def add_urls(self, urls: List[str]) -> None:
        """Add new URLs to the frontier"""
        new_urls = []
        
        for url in urls:
            # Basic URL validation and normalization
            try:
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    continue
                
                normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if parsed.query:
                    normalized_url += f"?{parsed.query}"
                    
                # Skip if already visited or in frontier
                if normalized_url in self.visited_urls or normalized_url in self.local_frontier:
                    continue
                    
                self.local_frontier.add(normalized_url)
                new_urls.append(normalized_url)
                
            except Exception as e:
                logger.warning(f"Error processing URL {url}: {e}")
                continue
        
        # Share new URLs with peers if any were added
        if new_urls:
            self.message_bus.publish("new_urls", new_urls)
            logger.info(f"Added {len(new_urls)} new URLs to frontier")
            
    def get_next_batch(self) -> List[str]:
        """Get the next batch of URLs to crawl"""
        if not self.local_frontier:
            return []
            
        # Select random sample to avoid getting stuck in one domain
        batch_size = min(self.batch_size, len(self.local_frontier))
        batch = random.sample(list(self.local_frontier), batch_size)
        
        # Remove selected URLs from frontier and mark as visited
        for url in batch:
            self.local_frontier.remove(url)
            self.visited_urls.add(url)
            
        return batch
        
    def _handle_peer_urls(self, message: dict) -> None:
        """Handle new URLs received from peers"""
        if "urls" in message:
            self.add_urls(message["urls"]) 