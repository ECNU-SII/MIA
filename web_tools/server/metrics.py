from prometheus_client import Counter, Histogram

REQUESTS = Counter("search_requests_total", "Total search requests", ["status", "tool"])
CACHE_HITS = Counter("search_cache_hits_total", "Cache hits", ["tool"])
LATENCY = Histogram("search_latency_seconds", "Search latency", ["tool"])