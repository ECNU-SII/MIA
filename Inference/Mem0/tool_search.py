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
SERPER_KEY=os.environ.get('SERPER_KEY_ID')


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
        params = {"q": query}
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
                
            formatted_results = []
            for i, item in enumerate(results, start=1):
                title = item.get("title", "No title")
                url = item.get("href", "No URL")
                extracted_info = item.get("body", "No extracted information")
                formatted_results.append(f"{i}. Title: {title}\n Body: {extracted_info}\n WebURL: {url}")
                
            return "\n".join(formatted_results)
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
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            # 单个查询
            response = self.search_with_server(query)
        else:
            # 多个查询
            assert isinstance(query, List)
            responses = []
            for q in query:
                responses.append(self.search_with_server(q))
            response = "\n=======\n".join(responses)
            
        return response

