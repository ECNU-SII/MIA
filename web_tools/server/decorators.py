"""
装饰器模块，用于简化工具接口的实现
"""
import uuid
import time
import logging
import json
import hashlib
from functools import wraps
from typing import Dict, Any, Callable
from fastapi.responses import JSONResponse

from .cache import get_cache, set_cache
from .metrics import REQUESTS, CACHE_HITS, LATENCY

logger = logging.getLogger("uvicorn")


def tool_endpoint(
    tool_name: str,
    tool_version: str = "1.0.0",
    cacheable: bool = True
):
    """
    工具端点装饰器，用于简化工具接口的实现
    
    Args:
        tool_name: 工具名称
        tool_version: 工具版本
        cacheable: 是否支持缓存
    
    Returns:
        装饰器函数
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 获取请求上下文
            request_id = str(uuid.uuid4())
            
            try:
                # 构建缓存作用域，只包含可序列化的参数
                scope = {}
                for k, v in kwargs.items():
                    if k not in ['deterministic']:
                        # 尝试序列化参数，如果失败则跳过
                        try:
                            json.dumps(v)
                            scope[k] = v
                        except (TypeError, ValueError):
                            # 如果参数不可序列化（如Pydantic模型），则跳过
                            pass
                
                # 处理缓存逻辑
                if cacheable:
                    cached_result = await _handle_cache_logic(
                        tool_name, tool_version, scope, 
                        kwargs.get('deterministic', False)
                    )
                    
                    if cached_result:
                        if cached_result == "CACHE_MISS":
                            return JSONResponse({
                                "error": "Cache miss in deterministic mode",
                                "request_id": request_id
                            }, status_code=404)
                        else:
                            return {
                                "request_id": request_id,
                                "tool": {"name": tool_name, "version": tool_version},
                                "result": cached_result,
                                "stat": {"cache": "HIT"}
                            }
                
                # 执行工具逻辑
                t0 = time.time()
                result = await func(*args, **kwargs)
                LATENCY.labels(tool_name).observe((time.time() - t0))
                REQUESTS.labels("200", tool_name).inc()
                
                # 缓存结果
                if cacheable and result:
                    suffix = _cache_key_suffix(scope)
                    set_cache(suffix, tool_name, tool_version, result)
                
                # 返回结果
                return {
                    "request_id": request_id,
                    "tool": {"name": tool_name, "version": tool_version},
                    "result": result,
                    "stat": {"cache": "MISS" if cacheable else "DISABLED"}
                }
                
            except Exception as e:
                logger.error(f"[{tool_name}] {request_id} failed: {e}", exc_info=True)
                REQUESTS.labels("500", tool_name).inc()
                return JSONResponse({
                    "request_id": request_id,
                    "error": "Internal server error",
                    "retry_after": 5000
                }, status_code=500)
                
        return wrapper
    return decorator


def _cache_key_suffix(params: dict) -> str:
    """
    生成缓存键的后缀
    
    Args:
        params: 参数字典
        
    Returns:
        str: 参数的SHA256哈希值
    """
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


async def _handle_cache_logic(tool: str, version: str, scope: dict, deterministic: bool = False):
    """
    统一处理缓存逻辑
    
    Args:
        tool: 工具名称
        version: 工具版本
        scope: 影响结果的参数
        deterministic: 是否启用确定性模式
        
    Returns:
        缓存数据或None
    """
    suffix = _cache_key_suffix(scope)
    
    cached = get_cache(suffix, tool, version)
    if cached:
        CACHE_HITS.labels(tool).inc()
        REQUESTS.labels("200_cache", tool).inc()
        return cached
    
    if deterministic:
        REQUESTS.labels("404_cache_miss", tool).inc()
        return "CACHE_MISS"
    
    return None