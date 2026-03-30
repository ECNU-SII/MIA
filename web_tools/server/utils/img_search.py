import os
import time
import logging
import urllib.parse
import base64
import json
import http.client
from io import BytesIO
from typing import Any
from contextlib import closing

import asyncio
import httpx
from PIL import Image, UnidentifiedImageError
from imghdr import what
from ddgs import DDGS

# ========================
# 日志配置
# ========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ddgs_image_search")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

# 从环境变量读取 API Key
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# 并发与大小限制
MAX_CONCURRENT_DOWNLOADS = 5
MAX_IMAGE_BYTES = 10_000_000  # 10MB


def contains_chinese(text: str) -> bool:
    """检查文本是否包含中文字符"""
    return any("\u4E00" <= c <= "\u9FFF" for c in text)


def _strip_to_origin(url: str) -> str:
    """提取 scheme://host"""
    try:
        p = urllib.parse.urlparse(url)
        if not p.scheme or not p.netloc:
            return ""
        return f"{p.scheme}://{p.netloc}"
    except Exception:
        return ""


def _open_image_bytes(data: bytes) -> Image.Image | None:
    """以最高兼容性打开图片字节"""
    if not data or what(None, h=data) not in ("jpeg", "png", "webp", "gif"):
        return None
    try:
        img = Image.open(BytesIO(data))
        img.load()
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, OSError, SyntaxError):
        return None


async def _fetch_image_once(
    client: httpx.AsyncClient,
    url: str,
    referer: str | None,
    timeout: float,
) -> Image.Image | None:
    """单次抓取"""
    headers = DEFAULT_HEADERS.copy()
    if referer:
        headers["Referer"] = referer
    try:
        async with client.stream("GET", url, headers=headers, timeout=timeout) as resp:
            resp.raise_for_status()
            data = await resp.aread()
            return _open_image_bytes(data)
    except Exception:
        return None


async def _fetch_image_robust(
    client: httpx.AsyncClient,
    image_url: str | None,
    thumb_url: str | None,
    source_page: str | None,
    timeout: float = 15.0,
) -> Image.Image | None:
    """多策略稳健抓取"""
    if not image_url and not thumb_url:
        return None

    referer = source_page or _strip_to_origin(image_url or "") or _strip_to_origin(thumb_url or "")

    for url, ref in [
        (image_url, referer),
        (image_url, None),
        (thumb_url, referer),
        (thumb_url, None),
    ]:
        if not url:
            continue
        img = await _fetch_image_once(client, url, ref, timeout)
        if img:
            return img
    return None


# ---------- 占位图 ----------
_PLACEHOLDER_IMG = Image.new("RGB", (224, 224), color=(200, 200, 200))
setattr(_PLACEHOLDER_IMG, "is_placeholder", True)


def _placeholder(w: int = 224, h: int = 224) -> Image.Image:
    """返回占位图副本"""
    img = _PLACEHOLDER_IMG.copy()
    setattr(img, "is_placeholder", True)
    return img


# ---------- 图像工具 ----------
def resize_and_encode_image(img: Image.Image, size: tuple[int, int] = (256, 256)) -> str:
    """调整大小并转 Base64"""
    img = img.resize(size)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


# ---------- Serper 搜索 ----------
def _sync_serper_image_search(query: str, max_results: int) -> dict[str, Any]:
    """同步执行 Serper 图片搜索"""
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY environment variable is not set")

    with closing(http.client.HTTPSConnection("google.serper.dev")) as conn:
        payload = json.dumps({
            "q": query,
            "location": "China" if contains_chinese(query) else "United States",
            "gl": "cn" if contains_chinese(query) else "us",
            "hl": "zh-cn" if contains_chinese(query) else "en",
            "num": max_results,
        })
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

        for attempt in range(5):
            try:
                conn.request("POST", "/images", payload, headers)
                res = conn.getresponse()
                data = res.read()
                return json.loads(data.decode("utf-8"))
            except Exception as e:
                if attempt == 4:
                    raise RuntimeError(f"Serper search failed after retries: {e}")
                time.sleep(0.1 * (2 ** attempt))
    return {}


# ---------- 统一结果处理 ----------
async def _process_search_results(
    results: list[dict[str, Any]],
    source: str,
    max_results: int,
) -> tuple[str, list[Image.Image], dict[str, Any]]:
    """下载并处理搜索结果"""
    logger.info(f"Processing {len(results)} {source} search results")
    t0 = time.time()
    results = results[:max_results]
    images: list[Image.Image] = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    async def limited_fetch(r: dict[str, Any]) -> Image.Image:
        async with sem:
            if source == "serper":
                img_url = r.get("imageUrl") or r.get("thumbnailUrl")
                thumb = r.get("thumbnailUrl")
                page = r.get("sourceUrl")
            else:
                img_url = r.get("image")
                thumb = r.get("thumbnail")
                page = r.get("source") or r.get("url")
            img = await _fetch_image_robust(client, img_url, thumb, page)
            return img or _placeholder()

    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        tasks = [limited_fetch(r) for r in results]
        image_results = await asyncio.gather(*tasks)

    lines = ["[Image Search Results]"]
    for i, (r, img) in enumerate(zip(results, image_results), start=1):
        title = (r.get("title") or "").strip() or "No title"
        suffix = "" if not getattr(img, "is_placeholder", False) else " (Failed to load)"
        lines.append(f"{i}. {title}{suffix}")
        images.append(img)

    total_ms = int((time.time() - t0) * 1000)
    stat = {
        "success": True,
        "engine": f"{source}-images",
        "num_results": len(results),
        "latency_ms": total_ms,
    }
    return "\n".join(lines), images, stat


# ---------- 搜索函数 ----------
async def call_serper_image_search(
    query: str,
    max_results: int = 3,
) -> tuple[str, list[Image.Image], dict[str, Any]]:
    logger.info(f"Serper search for: {query[:50]}...")
    loop = asyncio.get_running_loop()
    raw = await loop.run_in_executor(None, _sync_serper_image_search, query, max_results)
    results = raw.get("images", [])
    return await _process_search_results(results, "serper", max_results)


async def call_ddgs_image_search(
    query: str,
    max_results: int = 3,
    region: str = "cn-en",
    safesearch: str = "off",
    ddgs_timeout: float = 20.0,
) -> tuple[str, list[Image.Image], dict[str, Any]]:
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        None,
        lambda: list(DDGS(timeout=int(ddgs_timeout)).images(query, max_results=max_results, region=region, safesearch=safesearch)),
    )
    return await _process_search_results(results, "ddgs", max_results)


async def image_search_with_resize(
    query: str,
    max_results: int = 3,
    resize_size: tuple[int, int] = (256, 256),
    engine: str = "ddgs",
    **kwargs,
) -> dict[str, Any]:
    """搜索图片并返回包含文本、base64 图片列表和统计信息的字典"""
    if engine == "serper":
        text, imgs, stat = await call_serper_image_search(query, max_results)
    else:
        text, imgs, stat = await call_ddgs_image_search(query, max_results, **kwargs)

    loop = asyncio.get_running_loop()
    base64_imgs: list[str] = []
    for img in imgs:
        try:
            img_str = await loop.run_in_executor(None, resize_and_encode_image, img, resize_size)
        except Exception:
            placeholder = _placeholder(*resize_size)
            img_str = resize_and_encode_image(placeholder, resize_size)
        base64_imgs.append(img_str)

    return {
        "text": text,
        "images": base64_imgs,
        "stat": stat,
    }
