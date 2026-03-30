import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import aiohttp
from qwen_agent.tools.base import BaseTool, register_tool
import asyncio
from typing import Dict, List, Optional, Union
import uuid
import http.client
import json

import os

SERVICE_URL=os.environ.get('SERVICE_URL')   
@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        # Service configuration
        self.service_url = SERVICE_URL
        self.timeout = 30
        self.max_results = 5

    async def _call_search_service(self, query: str) -> dict:
        """Call the search service endpoint to get search results.
        
        Args:
            query: The search query string
            
        Returns:
            Dictionary containing search results
        """
        search_url = f"{self.service_url}/search"
        params = {"q": query, "engine": "serper"}
        if self.max_results is not None and isinstance(self.max_results, int):
            params["max_results"] = self.max_results
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params, timeout=self.timeout) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        return {
                            "error": f"Search service error: {response.status}",
                            "status_code": response.status
                        }
        except Exception as e:
            return {
                "error": f"Failed to call search service: {str(e)}",
                "exception": str(e)
            }

    def _format_search_results(self, search_result: dict) -> str:
        """Format search results to include title and extracted_info.
        
        Args:
            search_result: The raw search result from the service
            
        Returns:
            Formatted string with title and extracted_info
        """
        if "error" in search_result:
            return f"Error: {search_result['error']}"
            
        try:
            result_data = search_result.get("result", {})
            results = result_data.get("results", [])
            
            if not results:
                return "No search results found."
                   
            return results
        except Exception as e:
            return f"Error formatting search results: {str(e)}"
    
    def search_with_server(self, query: str):
        # 使用asyncio.run来运行异步函数
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            search_result = loop.run_until_complete(self._call_search_service(query))
            formatted_result = self._format_search_results(search_result)
            return formatted_result
        except Exception as e:
            return f"Error executing search: {str(e)}"

    def call(self, params: Union[str, dict], **kwargs) -> str:
        # 允许的查询字段名
        query_keys = ["query", "queries", "query_list"]

        if not isinstance(params, dict):
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        # 从多个可能字段中找到第一个存在的
        query = None
        for key in query_keys:
            if key in params:
                query = params[key]
                break

        if query is None:
            return "[Search] Invalid request format: Missing query field (accepted: query, queries, query_list)"

        # 单个字符串查询
        if isinstance(query, str):
            return self.search_with_server(query)

        # 多个查询
        if isinstance(query, list):
            responses = []
            for q in query:
                responses.append(self.search_with_server(q))
            return "\n=======\n".join(responses)

        # 类型不符合
        return "[Search] Invalid format: 'query' must be a string or list"



