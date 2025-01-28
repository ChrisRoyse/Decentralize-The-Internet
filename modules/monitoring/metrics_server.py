from aiohttp import web
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import psutil
import asyncio
from .metrics import MEMORY_USAGE, CPU_USAGE
from ..logging.request_logging import request_logging_middleware

class MetricsServer:
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        self.app = web.Application(middlewares=[request_logging_middleware])
        self.app.router.add_get('/metrics', self.metrics_handler)
        self.runner = None

    async def metrics_handler(self, request):
        """Handle /metrics requests from Prometheus"""
        # Update system metrics
        MEMORY_USAGE.set(psutil.Process().memory_info().rss)
        CPU_USAGE.set(psutil.Process().cpu_percent())
        
        # Generate metrics response
        metrics_data = generate_latest()
        return web.Response(
            body=metrics_data,
            content_type=CONTENT_TYPE_LATEST
        )

    async def start(self):
        """Start the metrics server"""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        
    async def stop(self):
        """Stop the metrics server"""
        if self.runner:
            await self.runner.cleanup() 