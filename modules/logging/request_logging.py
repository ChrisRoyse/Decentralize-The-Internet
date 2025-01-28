import structlog
import time
from typing import Callable, Awaitable
from aiohttp import web

logger = structlog.get_logger()

@web.middleware
async def request_logging_middleware(request: web.Request, 
                                   handler: Callable[[web.Request], Awaitable[web.Response]]) -> web.Response:
    """Log request details and timing"""
    start_time = time.time()
    
    request_id = request.headers.get('X-Request-ID', '-')
    
    log = logger.bind(
        request_id=request_id,
        method=request.method,
        path=request.path,
        remote=request.remote
    )
    
    log.info("request_started")
    
    try:
        response = await handler(request)
        duration = time.time() - start_time
        
        log.info("request_completed",
                 status=response.status,
                 duration=duration)
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        log.error("request_failed",
                  error_type=type(e).__name__,
                  error=str(e),
                  duration=duration)
        raise 