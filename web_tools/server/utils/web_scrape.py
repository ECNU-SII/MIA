# utils/web_scrape.py

import logging
import time
import asyncio
from typing import Dict, Optional
import urllib.parse

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from .jina_scrape_llm_summary import scrape_and_extract_info

logger = logging.getLogger(__name__)

# 定义爬虫配置，这可以被多个调用复用
CRAWLER_CONFIG = CrawlerRunConfig(
    cache_mode=CacheMode.BYPASS,
    excluded_tags=["nav", "footer", "aside", "script", "style"],
    remove_overlay_elements=True,
    markdown_generator=DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=0.48, threshold_type="fixed", min_word_threshold=0
        ),
        options={"ignore_links": True},
    ),
)

async def call_web_scrape(
    crawler: AsyncWebCrawler,
    url: str,
    info_to_extract: Optional[str] = None 
) -> Dict:
    """
    使用 crawl4ai 抓取单个 URL 并返回其 Markdown 内容。

    输入:
        crawler: 共享的 AsyncWebCrawler 实例，用于并发抓取。
        url: 要抓取的 URL。
        info_to_extract: 可选的字符串，用于指定需要从 Markdown 内容中提取的信息。

    输出:
        一个包含抓取结果的字典。
    """
    t0 = time.time()
    unquoted_url = urllib.parse.unquote(url)
    
    try:
        logger.info(f"Scraping URL: {unquoted_url}")
        
        # 使用共享的 crawler 实例执行抓取
        crawl_result = await crawler.arun(url=unquoted_url, config=CRAWLER_CONFIG)
        
        # 关键步骤：截断内容，防止内容过长撑爆 Agent 的上下文窗口
        # 8000 tokens 大约对应 15000-20000 个英文字符
        markdown_content = crawl_result.markdown[:15000]

        if not markdown_content.strip():
             raise ValueError("Scraped content is empty after cleaning.")

        latency_ms = int((time.time() - t0) * 1000)
        logger.info(f"Successfully scraped {unquoted_url} in {latency_ms}ms")

        if info_to_extract:
            extracted_info = await scrape_and_extract_info(markdown_content, info_to_extract)

        return {
            "success": True,
            "url": unquoted_url,
            "markdown_content": markdown_content,
            "extracted_info": extracted_info,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        logger.error(f"Failed to scrape URL {unquoted_url}: {e}", exc_info=True)
        return {
            "success": False,
            "url": unquoted_url,
            "error": f"An error occurred: {type(e).__name__} - {str(e)}",
            "latency_ms": latency_ms,
        }