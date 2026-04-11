"""
image_len_search工具模块

该模块提供了一个图像反向搜索功能，可以：
1. 接受base64编码的图像数据
2. 将图像上传到托管服务器
3. 使用Serper API进行反向图像搜索
4. 爬取搜索结果中的图像
5. 调整图像大小并返回base64编码

使用说明：
1. 配置SERPER_API_KEY环境变量或直接在代码中设置
2. 实现upload_image_to_hosting函数以使用真实的图床服务
3. 调用image_len_search函数进行图像搜索

配置说明：
- SERPER_API_KEY: Serper API密钥，需要在https://serper.dev/注册获取
- IMAGE_HOSTING_URL: 图床服务URL，示例中使用imgbb，实际使用时请替换为真实服务
- DEFAULT_HEADERS: HTTP请求头，模拟浏览器访问
"""

import time
import logging
import urllib.parse
import base64
import json
import http.client
import requests
import os
from io import BytesIO
from typing import List, Tuple, Dict, Optional

import asyncio
import httpx
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv

# ========================
# 配置日志
# ========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image_len_search")

# 加载环境变量
load_dotenv()

# 通用请求头（模拟常见浏览器）
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

# Serper API Key
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "your_key")

# 图床服务器URL（这里使用一个示例服务，实际使用时应替换为真实的图床服务）
IMAGE_HOSTING_URL = os.getenv("IMAGE_HOSTING_URL", "https://api.imgbb.com/1/upload")


def contains_chinese(text: str) -> bool:
    """检查文本是否包含中文字符"""
    return any('\u4E00' <= char <= '\u9FFF' for char in text)


def _strip_to_origin(url: str) -> str:
    """把 URL 归一到 scheme://host[:port] 作为 Referer 兜底。"""
    try:
        p = urllib.parse.urlparse(url)
        if not p.scheme or not p.netloc:
            return ""
        return f"{p.scheme}://{p.netloc}"
    except Exception:
        return ""


def _open_image_bytes(data: bytes) -> Image.Image:
    """
    以最高兼容性打开图片字节。先 load 再必要时转 RGB。
    """
    img = Image.open(BytesIO(data))
    img.load()  # 避免懒加载导致后续异常
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


async def _fetch_image_once(
    client: httpx.AsyncClient,
    url: str,
    referer: Optional[str],
    timeout: float,
    allow_redirects: bool = True,
) -> Optional[Image.Image]:
    """
    单次抓取尝试。带 UA，若提供 referer 则加入；成功返回 PIL.Image，失败返回 None。
    """
    headers = DEFAULT_HEADERS.copy()
    if referer:
        headers["Referer"] = referer

    try:
        resp = await client.get(url, headers=headers, timeout=timeout, follow_redirects=allow_redirects)
        resp.raise_for_status()
        return _open_image_bytes(resp.content)
    except Exception:
        return None


async def _fetch_image_robust(
    client: httpx.AsyncClient,
    image_url: Optional[str],
    thumb_url: Optional[str],
    source_page: Optional[str],
    timeout: float = 15.0,
) -> Optional[Image.Image]:
    """
    稳健抓取策略：
    1) 原图 + Referer(来源页) → 2) 原图 + 仅 UA → 3) 缩略图 + Referer → 4) 缩略图 + 仅 UA
    """
    if not image_url and not thumb_url:
        return None

    # 先选定一条用于 Referer 的可能值
    referer = source_page or _strip_to_origin(image_url or "") or _strip_to_origin(thumb_url or "")

    # 尝试 1：原图 + Referer
    if image_url:
        img = await _fetch_image_once(client, image_url, referer=referer, timeout=timeout)
        if img:
            return img

    # 尝试 2：原图 + 仅 UA
    if image_url:
        img = await _fetch_image_once(client, image_url, referer=None, timeout=timeout)
        if img:
            return img

    # 尝试 3：缩略图 + Referer
    if thumb_url:
        img = await _fetch_image_once(client, thumb_url, referer=referer, timeout=timeout)
        if img:
            return img

    # 尝试 4：缩略图 + 仅 UA
    if thumb_url:
        img = await _fetch_image_once(client, thumb_url, referer=None, timeout=timeout)
        if img:
            return img

    return None


def _placeholder(w: int = 224, h: int = 224, color=(200, 200, 200)) -> Image.Image:
    """失败兜底：返回占位图。"""
    return Image.new("RGB", (w, h), color=color)


def resize_and_encode_image(img: Image.Image, size: Tuple[int, int] = (256, 256)) -> str:
    """
    调整图片大小并将其编码为base64字符串
    
    输入:
        img: PIL.Image对象
        size: 调整后的图片尺寸，默认为(256, 256)
    
    输出:
        str: base64编码的图片数据
    """
    # Resize image
    img = img.resize(size)
    
    # Convert to base64 string
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str


def upload_image_to_hosting(base64_image: str) -> str:
    """
    将base64图像上传到图床服务并返回URL
    
    输入:
        base64_image: base64编码的图像数据
    
    输出:
        str: 上传后的图像URL
    """
    # 注意：这里使用imgbb作为示例，实际使用时需要替换为真实的图床API密钥
    # 在实际应用中，你可能需要使用自己的图床服务
    
    # 这里我们模拟上传过程，实际使用时需要替换为真实的API调用
    logger.info("Uploading image to hosting service...")
    
    # 示例实现（需要替换为真实的图床服务）：
    # payload = {
    #     "key": "YOUR_IMGBB_API_KEY",
    #     "image": base64_image
    # }
    # response = requests.post(IMAGE_HOSTING_URL, data=payload)
    # result = response.json()
    # return result["data"]["url"]
    
    # 为了演示目的，我们返回一个模拟的URL
    # 在实际应用中，请替换为真实的图床服务
    return f"https://example.com/uploaded_image_{int(time.time())}.jpg"


def _sync_serper_reverse_image_search(image_url: str, max_results: int) -> dict:
    """同步执行Serper反向图像搜索的包装函数"""
    if not SERPER_API_KEY:
        raise Exception("SERPER_API_KEY environment variable is not set")
    
    conn = http.client.HTTPSConnection("google.serper.dev")
    
    payload = json.dumps({
        "url": image_url,  # 使用图像URL进行反向搜索
        "num": max_results
    })

    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    # 尝试连接并获取结果
    for i in range(5):
        try:
            conn.request("POST", "/lens", payload, headers)
            res = conn.getresponse()
            break
        except Exception as e:
            if i == 4:  # 最后一次尝试
                raise Exception(f"Search failed after 5 attempts: {e}")
            time.sleep(0.1 * (2 ** i))  # 指数退避
    
    data = res.read()
    results = json.loads(data.decode("utf-8"))
    return results


async def _process_search_results(
    results: List[Dict],
    source: str,
    max_results: int
) -> Tuple[str, List[Image.Image], Dict]:
    """
    通用的搜索结果处理函数，用于下载和处理图片
    
    输入:
        results: 搜索结果列表
        source: 搜索引擎来源 ("serper" 或 "ddgs")
        max_results: 最大返回结果数
    
    输出:
        Tuple[str, List[Image.Image], Dict]: (格式化文本, PIL.Image 列表, 工具统计信息)
    """
    logger.info(f"Processing {len(results)} {source} search results")
    t0 = time.time()
    images: List[Image.Image] = []

    try:
        # 截取所需数量的结果
        results = results[:max_results]
        
        search_latency_ms = int((time.time() - t0) * 1000)
        logger.info(f"Image search completed in {search_latency_ms}ms with {len(results)} results")

        # 使用异步HTTP客户端下载图片
        async with httpx.AsyncClient() as client:
            lines = []
            if results:
                lines.append("[Image Search Results] Below are the images related to your query, ranked in descending order of relevance:")
                
                # 创建异步任务并发下载图片
                download_tasks = []
                for i, r in enumerate(results):
                    if source == "serper":
                        image_url = r.get("imageUrl") or r.get("thumbnailUrl") or None
                        thumb_url = r.get("thumbnailUrl") or None
                        source_page = r.get("sourceUrl") or None
                    else:  # ddgs
                        image_url = r.get("image") or None
                        thumb_url = r.get("thumbnail") or None
                        source_page = r.get("source") or r.get("url") or None
                    
                    task = _fetch_image_robust(
                        client,
                        image_url=image_url,
                        thumb_url=thumb_url,
                        source_page=source_page,
                        timeout=15.0,
                    )
                    download_tasks.append(task)
                
                # 并发执行所有下载任务
                image_results = await asyncio.gather(*download_tasks, return_exceptions=True)
                
                # 处理结果
                for i, result in enumerate(image_results):
                    if isinstance(result, Exception) or result is None:
                        logger.warning(f"Failed to download or process image {i+1}: {str(result)}")
                        image_results[i] = _placeholder()
                
                # 组装结果
                for i, r in enumerate(results, start=1):
                    title = (r.get("title") or "").strip() or "No title"
                    images.append(image_results[i-1])
                    if image_results[i-1] is not None and image_results[i-1] != _placeholder():
                        lines.append(f"{i}. title: {title}")
                    else:
                        lines.append(f"{i}. title: {title} (Failed to load image)")

                tool_returned_str = "\n".join(lines)
            else:
                tool_returned_str = "[Image Search Results] No results were found for your query."
                logger.info("No image results found for query")

        total_latency_ms = int((time.time() - t0) * 1000)
        tool_stat = {
            "success": True,
            "engine": f"{source}-images",
            "num_results": len(results),
            "search_latency_ms": search_latency_ms,
            "total_latency_ms": total_latency_ms,
        }
        return tool_returned_str, images, tool_stat

    except Exception as e:
        total_latency_ms = int((time.time() - t0) * 1000)
        logger.error(f"Error during image search: {str(e)}", exc_info=True)

        tool_returned_str = (
            "[Image Search Results] There was an error performing the search. "
            "Please reason with your own capabilities or try again later."
        )
        tool_stat = {
            "success": False,
            "engine": f"{source}-images",
            "error": str(e),
            "total_latency_ms": total_latency_ms,
        }
        return tool_returned_str, [], tool_stat


async def call_serper_reverse_image_search(
    image_url: str,
    max_results: int = 3,
) -> Tuple[str, List[Image.Image], Dict]:
    """
    使用 Serper 进行反向图像搜索，并稳健抓取图片。
    返回(格式化文本, PIL.Image 列表, 工具统计信息)。
    
    输入:
        image_url: 要搜索的图像URL
        max_results: 最大返回结果数
    
    输出:
        Tuple[str, List[Image.Image], Dict]: (格式化文本, PIL.Image 列表, 工具统计信息)
    """
    logger.info(f"Performing Serper reverse image search for image URL: {image_url[:50]}{'...' if len(image_url) > 50 else ''}")
    t0 = time.time()
    
    try:
        # 执行 Serper 反向图像搜索获取结果
        raw_results = await asyncio.get_event_loop().run_in_executor(
            None, 
            _sync_serper_reverse_image_search,
            image_url,
            max_results
        )
        
        # 处理搜索结果
        results = []
        if "images" in raw_results:
            results = raw_results["images"]
        
        # 使用通用处理函数处理结果
        return await _process_search_results(results, "serper", max_results)
        
    except Exception as e:
        total_latency_ms = int((time.time() - t0) * 1000)
        logger.error(f"Error during reverse image search: {str(e)}", exc_info=True)

        tool_returned_str = (
            "[Image Search Results] There was an error performing the reverse image search. "
            "Please reason with your own capabilities or try again later."
        )
        tool_stat = {
            "success": False,
            "engine": "serper-reverse-images",
            "error": str(e),
            "total_latency_ms": total_latency_ms,
        }
        return tool_returned_str, [], tool_stat


async def image_len_search(
    base64_image: str,
    max_results: int = 3,
    resize_size: Tuple[int, int] = (256, 256),
) -> Tuple[str, List[str], Dict]:
    """
    图像长度搜索工具：接受base64图像，上传到托管服务器，使用Serper的图搜图服务，
    爬取结果图像并调整大小后返回。
    
    输入:
        base64_image: base64编码的图像数据
        max_results: 最大返回结果数（默认3）
        resize_size: 图片调整大小的尺寸（默认256x256）
    
    输出:
        Tuple[str, List[str], Dict]: (文本结果, base64图片列表, 工具统计信息)
    """
    logger.info("Starting image length search...")
    t0 = time.time()
    
    try:
        # 1. 将base64图像上传到图床服务
        image_url = upload_image_to_hosting(base64_image)
        logger.info(f"Image uploaded to: {image_url}")
        
        # 2. 使用Serper进行反向图像搜索
        text_result, images, search_stat = await call_serper_reverse_image_search(
            image_url, max_results
        )
        
        # 3. 调整图片大小并转换为base64编码
        image_data = []
        for img in images:
            try:
                # Resize image and convert to base64
                img_str = resize_and_encode_image(img, resize_size)
                image_data.append(img_str)
            except Exception as e:
                logger.warning(f"Failed to resize and encode image: {str(e)}")
                # 添加一个默认的占位图
                placeholder = _placeholder(resize_size[0], resize_size[1])
                img_str = resize_and_encode_image(placeholder, resize_size)
                image_data.append(img_str)
        
        # 更新统计信息
        total_latency_ms = int((time.time() - t0) * 1000)
        tool_stat = search_stat.copy()
        tool_stat["total_latency_ms"] = total_latency_ms
        tool_stat["engine"] = "serper-reverse-images-with-resize"
        
        return text_result, image_data, tool_stat
        
    except Exception as e:
        total_latency_ms = int((time.time() - t0) * 1000)
        logger.error(f"Error during image length search: {str(e)}", exc_info=True)

        tool_returned_str = (
            "[Image Search Results] There was an error performing the image length search. "
            "Please reason with your own capabilities or try again later."
        )
        tool_stat = {
            "success": False,
            "engine": "serper-reverse-images-with-resize",
            "error": str(e),
            "total_latency_ms": total_latency_ms,
        }
        return tool_returned_str, [], tool_stat
