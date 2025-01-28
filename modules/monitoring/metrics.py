from prometheus_client import Counter, Gauge, Histogram
import time

# Crawler metrics
PAGES_CRAWLED = Counter('pages_crawled_total', 'Number of pages crawled')
CRAWL_ERRORS = Counter('crawl_errors_total', 'Number of crawl errors', ['error_type'])
CRAWL_DURATION = Histogram('crawl_duration_seconds', 'Time spent crawling pages')
FRONTIER_SIZE = Gauge('frontier_size', 'Number of URLs in frontier')
URLS_EXTRACTED = Counter('urls_extracted_total', 'Number of URLs extracted from pages')

# Content processing metrics
CHUNKS_PROCESSED = Counter('chunks_processed_total', 'Number of text chunks processed')
DUPLICATE_CHUNKS = Counter('duplicate_chunks_total', 'Number of duplicate chunks detected')
COMPRESSION_RATIO = Histogram('compression_ratio', 'Compression ratio of processed chunks')

# Knowledge graph metrics
FACTS_EXTRACTED = Counter('facts_extracted_total', 'Number of facts extracted')
RELATIONSHIPS_CREATED = Counter('relationships_created_total', 'Number of relationships created')
CONFLICTS_RESOLVED = Counter('conflicts_resolved_total', 'Number of conflicts resolved', ['resolution_type'])
TRUTH_CONFIDENCE = Histogram('truth_confidence', 'Distribution of truth confidence scores')

# P2P metrics
MESSAGES_SENT = Counter('messages_sent_total', 'Number of P2P messages sent', ['message_type'])
MESSAGES_RECEIVED = Counter('messages_received_total', 'Number of P2P messages received', ['message_type'])
PEER_COUNT = Gauge('peer_count', 'Number of connected peers')

# System metrics
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')

class MetricsMiddleware:
    """Middleware to track timing of operations"""
    
    @staticmethod
    async def track_crawl_time(func):
        """Decorator to track crawl time"""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                CRAWL_DURATION.observe(time.time() - start_time)
                return result
            except Exception as e:
                CRAWL_ERRORS.labels(error_type=type(e).__name__).inc()
                raise
        return wrapper

    @staticmethod
    def track_fact_extraction(func):
        """Decorator to track fact extraction"""
        def wrapper(*args, **kwargs):
            try:
                facts = func(*args, **kwargs)
                if facts:
                    FACTS_EXTRACTED.inc(len(facts))
                return facts
            except Exception as e:
                CRAWL_ERRORS.labels(error_type=type(e).__name__).inc()
                raise
        return wrapper 