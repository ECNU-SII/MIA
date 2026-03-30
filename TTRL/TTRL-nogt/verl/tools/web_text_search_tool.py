# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import aiohttp
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class WebTextSearchTool(BaseTool):
    """Web text search tool that calls the deployed search service.
    
    This tool takes a search query as input and calls the search endpoint
    in the deployed service to get search results.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize WebTextSearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
        # Service configuration
        self.service_url = config.get("service_url", "http://localhost:8000")
        self.timeout = config.get("timeout", 30)
        self.max_results = config.get("max_results")
        
        logger.info(f"Initialized WebTextSearchTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        if self.tool_schema is not None:
            return self.tool_schema
            
        # Default schema if none provided
        return OpenAIFunctionToolSchema(
            type="function",
            function={
                "name": "web_text_search",
                "description": "Searches for relevant information based on a query using web search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

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
                        logger.error(f"Search service returned status {response.status}: {error_text}")
                        return {
                            "error": f"Search service error: {response.status}",
                            "status_code": response.status
                        }
        except Exception as e:
            logger.error(f"Error calling search service: {e}")
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
            logger.error(f"Error formatting text search results: {e}")
            return f"Error formatting search results: {str(e)}"

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the web text search tool.

        Args:
            instance_id: The instance id of the tool.
            parameters: The parameters containing the search query.

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing search results.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        # Validate parameters
        query = parameters.get("query")
        if not query or not isinstance(query, str):
            error_msg = "Error: 'query' is missing or not a string in parameters."
            logger.error(f"[WebTextSearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=error_msg), 0.0, {"error": "invalid_parameters"}

        try:
            # Call the search service
            search_result = await self._call_search_service(query)
            
            # Format the results
            formatted_result = self._format_search_results(search_result)
            
            # Store results in instance dictionary
            if instance_id in self._instance_dict:
                self._instance_dict[instance_id]["reward"].append(formatted_result.strip())
            
            # Create metrics
            metrics = {
                "status": "success" if "error" not in search_result else "error",
                "query": query
            }
            
            if "error" not in search_result:
                result_data = search_result.get("result", {})
                metrics["result_count"] = len(result_data.get("results", []))
            
            return ToolResponse(text=formatted_result), 0.0, metrics

        except Exception as e:
            error_result = f"Web search execution failed: {str(e)}"
            logger.error(f"[WebTextSearchTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e), "status": "exception"}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool.
        """
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]