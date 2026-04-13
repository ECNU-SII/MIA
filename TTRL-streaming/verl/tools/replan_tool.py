# Modified version of WebImageToImageSearchTool
# Title priority: if title's index corresponds to HTML or missing image, return text only (no placeholder image)

import os
import json
from typing import Any, Optional, Tuple
import numpy as np
from uuid import uuid4
import re
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
import requests


class ReplanTool(BaseTool):
    def __init__(self, config: dict = None, tool_schema: OpenAIFunctionToolSchema = None):
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema(
                type="function",
                function={
                    "name": "replan",
                    "description": "**IMPORTANT: This tool can only be called once. When the current plan fails to produce a verifiable answer—due to missing information, logical inconsistency, or ambiguous results—invoke this tool to obtain a revised, actionable plan from an expert strategist.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reflexion": {
                                "type": "string",
                                "description": "A concise self-reflection summarizing what went wrong in the current execution to inform the revised plan."
                            },
                        },
                        "required": ["reflexion"]
                    }
                }
            )
        self.memory_service_url = config.get("memory_service_url")
        self.timeout = config.get("timeout", 30)
        super().__init__(config or {}, tool_schema)


    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)
        data_id = create_kwargs.get("data_id")
        question = create_kwargs.get("question")
        plan = create_kwargs.get("plan")
        image_caption = create_kwargs.get("image_caption", [])
        if question is None:
            raise ValueError("Missing required 'data_id' or 'question' parameter in create_kwargs")
        if not hasattr(self, '_instance_dict'):
            self._instance_dict = {}
        self._instance_dict[instance_id] = {
            "data_id": data_id,
            "question": question,
            "plan": plan,
            "image_caption": image_caption,
            "response": "",
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, dict]:
        instance_data = {}
        if hasattr(self, '_instance_dict') and instance_id in self._instance_dict:
            instance_data = self._instance_dict[instance_id]
        data_id = instance_data.get("data_id")
        question = instance_data.get("question")
        plan = instance_data.get("plan")
        image_caption = instance_data.get("image_caption")
        workflow = kwargs.get("workflow", "")
        reflexion = parameters.get("reflexion")
        if not workflow:
            workflow = "No Tool Call."
        try:
            tool_returned_str, tool_stat = self.call_replan(data_id, question, plan, image_caption, reflexion, workflow)
            response = ToolResponse(
                text=tool_returned_str,
            )
            reward = 0.0
            return response, reward, tool_stat
        except Exception as e:
            error_msg = f"[Replan Results] Error executing replan: {str(e)}"
            return (
                ToolResponse(text=error_msg),
                -0.1,
                {"success": False, "error": str(e)}
            )



    def call_replan(self, data_id, question, plan, image_caption, reflexion, workflow):
        guide_url = self.memory_service_url
        tool_returned_str = ""
        tool_success = False

        try:
            data = [{"data_id": data_id, "question": question, "plan": plan, "image_caption": image_caption, "reflexion": reflexion, "workflow": workflow}]
            response = requests.post(guide_url, json=data)
            guides = response.json()
            tool_returned_str = guides[0]
            tool_success = True
        except Exception as e:
            tool_returned_str = "[Replan Results] Error encountered."
            tool_success = False
        tool_stat = {
            "success": tool_success,
            "guide": tool_returned_str,
        }
        return tool_returned_str, tool_stat

    async def release(self, instance_id: str, **kwargs) -> None:
        if hasattr(self, '_instance_dict') and instance_id in self._instance_dict:
            del self._instance_dict[instance_id]