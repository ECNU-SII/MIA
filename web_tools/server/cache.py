import redis
import json
import hashlib
import os
import logging

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "604800"))  # 1周 默认

# 尝试连接Redis，如果失败则禁用缓存
try:
    r = redis.Redis.from_url(REDIS_URL)
    # 测试连接
    r.ping()
    cache_enabled = True
except:
    logger.warning("Redis连接失败，缓存功能已禁用")
    cache_enabled = False
    r = None

def _make_key(query: str, tool: str, version: str) -> str:
    h = hashlib.sha256(f"{tool}:{version}:{query}".encode()).hexdigest()
    return f"cache:{tool}:{version}:{h}"

def get_cache(query: str, tool: str, version: str):
    if not cache_enabled or r is None:
        return None
        
    try:
        key = _make_key(query, tool, version)
        data = r.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        logger.error(f"获取缓存失败: {e}")
        return None

def set_cache(query: str, tool: str, version: str, value: dict):
    if not cache_enabled or r is None:
        return
        
    try:
        key = _make_key(query, tool, version)
        r.setex(key, CACHE_TTL, json.dumps(value))
    except Exception as e:
        logger.error(f"设置缓存失败: {e}")