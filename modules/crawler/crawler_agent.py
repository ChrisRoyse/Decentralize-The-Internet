import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from ..monitoring.metrics import (
    PAGES_CRAWLED, CRAWL_ERRORS, URLS_EXTRACTED, 
    CHUNKS_PROCESSED, MetricsMiddleware
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrawlerAgent:
    def __init__(self, config, frontier_manager, deduper, compressor, 
                 entity_extractor, kg_manager, conflict_resolver, 
                 message_bus, access_control):
        self.config = config
        self.frontier_manager = frontier_manager
        self.deduper = deduper
        self.compressor = compressor
        self.entity_extractor = entity_extractor
        self.kg_manager = kg_manager
        self.conflict_resolver = conflict_resolver
        self.message_bus = message_bus
        self.access_control = access_control
        
        self.user_agent = config["crawler"]["user_agent"]
        self.max_concurrent = config["crawler"].get("max_concurrent_requests", 5)
        self.politeness_delay = config["crawler"].get("politeness_delay", 1.0)
        self.timeout = config["crawler"].get("timeout", 30)
        self.max_retries = config["crawler"].get("max_retries", 3)
        
        # Initialize with start URLs if provided
        if "start_urls" in config["crawler"]:
            self.frontier_manager.add_urls(config["crawler"]["start_urls"])

        # Rate limiting per domain
        self.domain_delays = {}
        self.domain_last_access = {}

    @MetricsMiddleware.track_crawl_time
    async def crawl_cycle(self):
        """Run one crawl cycle with parallel requests"""
        urls = self.frontier_manager.get_next_batch()
        if not urls:
            logger.info("No URLs in frontier, waiting...")
            return

        PAGES_CRAWLED.inc(len(urls))

        # Group URLs by domain for politeness
        domain_groups = self._group_urls_by_domain(urls)
        
        # Create tasks for each domain group
        tasks = []
        async with aiohttp.ClientSession() as session:
            for domain, domain_urls in domain_groups.items():
                task = asyncio.create_task(
                    self._process_domain_urls(session, domain, domain_urls)
                )
                tasks.append(task)
            
            # Wait for all domain tasks to complete
            await asyncio.gather(*tasks)

    async def _process_domain_urls(self, session: aiohttp.ClientSession, 
                                 domain: str, urls: List[str]) -> None:
        """Process all URLs for a single domain with politeness delay"""
        for url in urls:
            # Respect politeness delay for this domain
            await self._wait_for_politeness(domain)
            
            try:
                content = await self._fetch_url(session, url)
                if content:
                    await self._process_content(url, content)
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")

            # Update last access time for this domain
            self.domain_last_access[domain] = asyncio.get_event_loop().time()

    async def _fetch_url(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch a URL with retries"""
        headers = {"User-Agent": self.user_agent}
        
        for attempt in range(self.max_retries):
            try:
                async with session.get(url, headers=headers, 
                                     timeout=self.timeout) as response:
                    if response.status == 200:
                        PAGES_CRAWLED.inc()
                        return await response.text()
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {url}, attempt {attempt + 1}")
            except Exception as e:
                CRAWL_ERRORS.labels(error_type=type(e).__name__).inc()
                return None
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None

    async def _process_content(self, url: str, content: str) -> None:
        """Process fetched content"""
        # Extract text chunks
        soup = BeautifulSoup(content, "html.parser")
        chunks = self._extract_chunks(soup)
        
        # Process each chunk
        for chunk in chunks:
            # Check for duplicates
            is_new, chunk_id, compressed = self.deduper.process_chunk(chunk)
            if is_new:
                # Extract entities and relationships
                facts = self.entity_extractor.extract_entities(chunk)
                
                # Update knowledge graph
                if facts:
                    self.kg_manager.add_facts(facts, self.conflict_resolver)
        
        # Extract and queue new URLs
        new_urls = self._extract_urls(soup, url)
        if new_urls:
            self.frontier_manager.add_urls(new_urls)
            URLS_EXTRACTED.inc(len(new_urls))
        
        CHUNKS_PROCESSED.inc(len(chunks))

    def _extract_chunks(self, soup: BeautifulSoup) -> List[str]:
        """Extract text chunks from HTML"""
        chunks = []
        for p in soup.find_all(["p", "article", "section"]):
            text = p.get_text(separator=" ", strip=True)
            if len(text) > 50:  # Minimum length threshold
                chunks.append(text)
        return chunks

    def _extract_urls(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract and normalize URLs from HTML"""
        urls = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Normalize URL
            url = urljoin(base_url, href)
            # Basic URL filtering
            if url.startswith(("http://", "https://")):
                urls.append(url)
        return urls

    def _group_urls_by_domain(self, urls: List[str]) -> Dict[str, List[str]]:
        """Group URLs by domain for polite crawling"""
        from urllib.parse import urlparse
        groups = {}
        for url in urls:
            try:
                domain = urlparse(url).netloc
                if domain not in groups:
                    groups[domain] = []
                groups[domain].append(url)
            except Exception:
                logger.warning(f"Invalid URL: {url}")
        return groups

    async def _wait_for_politeness(self, domain: str) -> None:
        """Wait appropriate time for politeness"""
        now = asyncio.get_event_loop().time()
        last_access = self.domain_last_access.get(domain, 0)
        delay = self.domain_delays.get(domain, self.politeness_delay)
        
        wait_time = last_access + delay - now
        if wait_time > 0:
            await asyncio.sleep(wait_time) 