from fastapi import FastAPI, Query, Request, Body
from fastapi.responses import JSONResponse, PlainTextResponse
import uuid
import logging
import time
import json
import base64
from io import BytesIO
from typing import List, Union, Optional
from pydantic import BaseModel
from .utils.text_search import call_web_text_search
from .cache import get_cache, set_cache
from .metrics import REQUESTS, CACHE_HITS, LATENCY
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import hashlib
from .utils.img_search import call_ddgs_image_search
from .decorators import tool_endpoint
from .utils.web_scrape import call_web_scrape
from crawl4ai import AsyncWebCrawler
from contextlib import asynccontextmanager
from .utils.img_len_search import image_len_search

logger = logging.getLogger("uvicorn")

# # 定义图像反向搜索请求模型
# class ImageReverseSearchRequest(BaseModel):
#     base64_image: str
#     max_results: int = 3
#     resize_width: int = 256
#     resize_height: int = 256

# 全局爬虫实例
crawler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global crawler
    # 启动时初始化爬虫
    logger.info("Initializing web crawler...")
    crawler = AsyncWebCrawler()
    await crawler.__aenter__()
    logger.info("Web crawler initialized successfully")
    
    yield
    
    # 关闭时清理爬虫
    if crawler:
        logger.info("Cleaning up web crawler...")
        await crawler.__aexit__(None, None, None)
        logger.info("Web crawler cleaned up successfully")

app = FastAPI(title="Tool Gateway Server", version="1.1.0", lifespan=lifespan)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """
    为每个HTTP请求添加唯一请求ID的中间件
    
    输入:
        request: FastAPI Request对象，包含HTTP请求信息
        call_next: 异步可调用对象，用于继续处理请求链
    
    输出:
        response: FastAPI Response对象，在响应头中添加X-Request-ID
    """
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response



@app.get("/metrics")
async def metrics():
    """
    提供Prometheus监控指标的端点
    
    输入:
        无（HTTP GET请求）
    
    输出:
        PlainTextResponse: 包含Prometheus格式的监控指标数据
    """
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

from typing import Any

def _cache_key_suffix(params: dict[str, Any]) -> str:
    """
    把影响结果的参数做成稳定哈希，避免 key 爆长
    
    输入:
        params: dict类型，包含影响搜索结果的所有参数
    
    输出:
        str: 参数集的SHA256哈希值（64位十六进制字符串）
    """
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


async def _handle_cache_logic(tool: str, version: str, scope: dict[str, Any], deterministic: bool = False):
    """
    统一处理缓存逻辑
    
    输入:
        tool: 工具名称
        version: 工具版本
        scope: 影响结果的参数
        deterministic: 是否启用确定性模式
    
    输出:
        tuple: (cached_data, suffix) 或 (None, suffix)
    """
    suffix = _cache_key_suffix(scope)
    cached = get_cache(suffix, tool, version)
    if cached:
        CACHE_HITS.labels(tool).inc()
        REQUESTS.labels("200_cache", tool).inc()
        return cached, suffix
    
    if deterministic:
        REQUESTS.labels("404_cache_miss", tool).inc()
        return "CACHE_MISS", suffix
    
    return None, suffix

@app.get("/search")
@tool_endpoint(tool_name="text_search", tool_version="1.0.0")
async def search(q: Union[str, List[str]] = Query(..., max_length=400, min_length=1),
                 max_results: int = Query(3, ge=1, le=10),
                 deterministic: bool = False,
                 engine: str = Query("ddgs", description="搜索引擎，可选 'ddgs' 或 'serper'")):
    """
    文本搜索接口
    
    输入:
        q: str或List[str]类型，搜索查询字符串或字符串列表，最大长度400字符，最小长度1字符（必填参数）
        max_results: int类型，每个查询返回结果的最大数量，默认值为3，范围1-10
        deterministic: bool类型，是否启用确定性模式，默认值为False
        engine: str类型，选择搜索引擎
    输出:
        dict: 包含以下字段的响应字典
            - request_id: 唯一请求标识符
            - tool: 工具信息（名称和版本）
            - result: 搜索结果数据或错误信息
                -success: bool类型，搜索是否成功
                -engine": str类型，使用的搜索引擎
                -results": str类型，这次搜索结果，包括title和网页内容
                -latency_ms: int类型，搜索延迟
            - stat: 统计信息（缓存命中状态）
    """
    # 直接调用异步函数
    from .utils.text_search import call_web_text_search
    return await call_web_text_search(q, max_results, engine)


@app.get("/image_search")
@tool_endpoint(tool_name="image_search", tool_version="1.0.0")
async def image_search(q: str = Query(..., min_length=1),
                       max_results: int = Query(5, ge=1, le=10),
                       deterministic: bool = False,
                       engine: str = Query("ddgs", description="搜索引擎，可选 'ddgs' 或 'serper'")):
    """
    图片搜索接口
    
    输入:
        q: str类型，搜索查询字符串，最小长度1字符（必填参数）
        max_results: int类型，返回图片的最大数量，默认值为5，范围1-10
        deterministic: bool类型，是否启用确定性模式，默认值为False
        engine: str类型，搜索引擎，可选 "ddgs" 或 "serper"，默认值为 "ddgs"
    
    输出:
        dict: 包含以下字段的响应字典
            - request_id: 唯一请求标识符
            - tool: 工具信息（名称和版本）
            - result: 包含以下字段的搜索结果数据
                - text: 文本搜索结果,图片的title
                - images: List[str]类型，base64编码的图片数据列表
                - stat: 统计信息
            - stat: 缓存状态统计信息
    """
    # 直接调用异步函数（图片resize和编码已在img_search中处理）
    from .utils.img_search import image_search_with_resize
    return await image_search_with_resize(q, max_results, engine=engine)
    


@app.get("/web_scrape")
@tool_endpoint(tool_name="web_scrape", tool_version="1.0.0")
async def web_scrape(url: str = Query(..., min_length=1, max_length=2048),
                     goal: Optional[str] = None):
    """
    网页抓取接口
    
    输入:
        url: str类型，要抓取的网页URL，最小长度1字符，最大长度2048字符（必填参数）
        goal: 可选的字符串，用于指定需要从 Markdown 内容中提取的信息。
    
    输出:
        dict: 包含以下字段的响应字典
            - request_id: 唯一请求标识符
            - tool: 工具信息（名称和版本）
            - result: 包含以下字段的抓取结果数据
                - success: bool类型，是否成功
                - url: str类型，原始URL
                - markdown_content: str类型，抓取到的Markdown内容
                - extracted_info: 可选的字符串，根据 info_to_extract 提取的信息
                - latency_ms: int类型，延迟毫秒数
            - stat: 缓存状态统计信息
    """
    global crawler
    if crawler is None:
        return JSONResponse(
            status_code=500,
            content={
                "request_id": str(uuid.uuid4()),
                "error": "Web crawler is not initialized"
            }
        )
    return await call_web_scrape(crawler, url, goal)

