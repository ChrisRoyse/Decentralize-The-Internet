from prometheus_client import Counter, Gauge, Histogram, Summary
import time

# Crawler Metrics
PAGES_CRAWLED = Counter('pages_crawled_total', 'Number of pages crawled')
CRAWL_ERRORS = Counter('crawl_errors_total', 'Number of crawl errors', ['error_type'])
CRAWL_DURATION = Histogram('crawl_duration_seconds', 'Time spent crawling pages')
URL_QUEUE_SIZE = Gauge('url_queue_size', 'Number of URLs in the frontier')
BYTES_DOWNLOADED = Counter('bytes_downloaded_total', 'Total bytes downloaded')

# Content Processing Metrics
CHUNKS_PROCESSED = Counter('chunks_processed_total', 'Number of text chunks processed')
DUPLICATE_CHUNKS = Counter('duplicate_chunks_total', 'Number of duplicate chunks detected')
COMPRESSION_RATIO = Histogram('compression_ratio', 'Compression ratio of processed chunks')
ENTITY_EXTRACTION_TIME = Summary('entity_extraction_seconds', 'Time spent extracting entities')

# Knowledge Graph Metrics
RELATIONSHIPS_CREATED = Counter('relationships_created_total', 'Number of relationships created')
CONFLICTS_RESOLVED = Counter('conflicts_resolved_total', 'Number of conflicts resolved', ['resolution_type'])
TRUTH_CONFIDENCE = Histogram('truth_confidence', 'Distribution of truth confidence scores')
GRAPH_SIZE = Gauge('graph_size', 'Total number of relationships in graph')
SHARD_OPERATIONS = Counter('shard_operations_total', 'Number of cross-shard operations', ['operation_type'])

# P2P Network Metrics
MESSAGES_SENT = Counter('messages_sent_total', 'Number of P2P messages sent', ['message_type'])
MESSAGES_RECEIVED = Counter('messages_received_total', 'Number of P2P messages received', ['message_type'])
PEER_COUNT = Gauge('peer_count', 'Number of connected peers')
MESSAGE_SIZE = Histogram('message_size_bytes', 'Size of P2P messages')

# System Metrics
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')
DISK_IO = Counter('disk_io_bytes_total', 'Disk I/O bytes', ['operation'])
GC_TIME = Histogram('gc_duration_seconds', 'Garbage Collection duration')

class MetricsMiddleware:
    """Middleware to track timing and metrics of operations"""
    
    @staticmethod
    def track_operation(metric):
        """Decorator to track operation timing"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    metric.observe(duration)
                    return result
                except Exception as e:
                    CRAWL_ERRORS.labels(error_type=type(e).__name__).inc()
                    raise
            return wrapper
        return decorator

    @staticmethod
    def track_size(metric):
        """Decorator to track data size"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                if result:
                    try:
                        size = len(str(result).encode('utf-8'))
                        metric.observe(size)
                    except:
                        pass
                return result
            return wrapper
        return decorator 