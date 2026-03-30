import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
from qwen_agent.tools.base import BaseTool, register_tool
import asyncio
from typing import Dict, List, Optional, Union
import uuid
import http.client
import json
import requests
import os


SERVICE_URL=os.environ.get('SERVICE_URL')   

def _passages2string(retrieval_result):
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference


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
        
        

        
    def search_with_server(self, query: str):
        # 使用asyncio.run来运行异步函数
        payload = {
                "queries": [query],
                "topk": 3,
                "return_scores": True
            }
        results = requests.post(f"{self.service_url}/retrieve", json=payload).json()['result']
    
        return _passages2string(results[0])





    def call(self, params: Union[str, dict], **kwargs) -> str:
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

