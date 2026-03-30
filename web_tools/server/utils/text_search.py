# utils/text_search.py

import asyncio
import httpx
import logging
import os
import urllib.parse
from typing import Any

from ddgs import DDGS

logger = logging.getLogger(__name__)

SERPER_API_KEY = os.getenv("SERPER_KEY_ID")


async def _ddgs_search(query: str, max_results: int = 5, region: str = "us-en", backend: str = "brave") -> dict[str, Any]:
    """
    使用 DuckDuckGo Search (DDGS) 进行异步搜索。
    返回统一结构的结果字典。
    """
    try:
        results: list[dict[str, Any]] = []
        with DDGS() as ddgs:
            raw = list(ddgs.text(
                query,
                max_results=max_results,
                region=region,
                safesearch="moderate",
                backend=backend
            ))

        for idx, item in enumerate(raw, start=1):
            title = item.get("title", "")
            url = item.get("href", "")
            snippet = item.get("body", "")
            formatted = f"{idx}. [{title}]({url})\n{snippet}".replace(
                "Your browser can't play this video.", ""
            )
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "formatted": formatted
            })

        formatted_text = (
            f"A search for '{query}' found {len(results)} results:\n\n## Web Results\n" +
            "\n\n".join(r["formatted"] for r in results)
        )

        return {"success": True, "engine": "ddgs", "results": results, "formatted": formatted_text}

    except Exception as e:
        logger.exception("DDGS search failed", extra={"query": query})
        return {"success": False, "engine": "ddgs", "error": str(e), "results": [], "formatted": ""}


async def _serper_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    使用 Serper API 进行异步搜索。
    返回统一结构的结果字典。
    """
    if not SERPER_API_KEY:
        return {
            "success": False,
            "engine": "serper",
            "error": "SERPER_API_KEY not set",
            "results": [],
            "formatted": ""
        }

    contains_chinese = any('\u4E00' <= c <= '\u9FFF' for c in query)
    payload = {
        "q": query,
        "num": max_results,
        "location": "China" if contains_chinese else "United States",
        "gl": "cn" if contains_chinese else "us",
        "hl": "zh-cn" if contains_chinese else "en"
    }

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post("https://google.serper.dev/search", json=payload, headers=headers)
            res.raise_for_status()
            data = res.json()
    except Exception as e:
        logger.exception("Serper search failed", extra={"query": query})
        return {"success": False, "engine": "serper", "error": str(e), "results": [], "formatted": ""}

    organic = data.get("organic", [])
    if not organic:
        return {"success": True, "engine": "serper", "results": [], "formatted": f"No results found for '{query}'."}

    results: list[dict[str, Any]] = []
    for idx, page in enumerate(organic, start=1):
        title = page.get("title", "")
        url = page.get("link", "")
        snippet = page.get("snippet", "")
        date = f"\nDate published: {page['date']}" if "date" in page else ""
        source = f"\nSource: {page['source']}" if "source" in page else ""

        formatted = (
            f"{idx}. [{title}]({url}){date}{source}\n{snippet}"
        ).replace("Your browser can't play this video.", "")

        results.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "source": page.get("source", ""),
            "date": page.get("date", ""),
            "formatted": formatted
        })

    formatted_text = (
        f"A search for '{query}' found {len(results)} results:\n\n## Web Results\n" +
        "\n\n".join(r["formatted"] for r in results)
    )

    return {"success": True, "engine": "serper", "results": results, "formatted": formatted_text}


async def call_web_text_search(
    query: str | list[str],
    max_results: int = 5,
    engine: str = "ddgs"
) -> dict[str, Any]:
    """
    执行文本搜索（支持单个或多个查询）。
    自动选择引擎（ddgs 或 serper）。
    """
    t0 = asyncio.get_running_loop().time()

    if isinstance(query, list):
        tasks = [call_web_text_search(q, max_results, engine) for q in query]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = [r for r in results if isinstance(r, dict) and r.get("success")]
        formatted_all = "\n=======\n".join(r.get("formatted", "") for r in successful)
        latency = int((asyncio.get_running_loop().time() - t0) * 1000)

        return {
            "success": True,
            "engine": engine,
            "results": formatted_all,
            "latency_ms": latency
        }

    query = urllib.parse.unquote(query.strip())
    try:
        if engine == "ddgs":
            result = await _ddgs_search(query, max_results)
        elif engine == "serper":
            result = await _serper_search(query, max_results)
        else:
            raise ValueError(f"Unsupported search engine: {engine}")
    except Exception as e:
        logger.exception("Search error", extra={"query": query, "engine": engine})
        result = {"success": False, "engine": engine, "error": str(e), "results": [], "formatted": ""}

    result["latency_ms"] = str(int((asyncio.get_running_loop().time() - t0) * 1000))
    return result