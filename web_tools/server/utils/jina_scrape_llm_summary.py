# Copyright 2025 MiroMind Team
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
import time
from typing import Any, Dict
from openai import AsyncOpenAI   
import aiohttp
import tempfile
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm_base_url = ""
llm_name = "qwen25"

async def scrape_and_extract_info(
    extracted_info: str, info_to_extract: str, custom_headers: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Scrape content from a URL and extract specific types of information using LLM.

    Args:
        url (str): The URL to scrape content from
        info_to_extract (str): The specific types of information to extract (usually a question)
        custom_headers (Dict[str, str]): Additional headers to include in the scraping request

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - url (str): The original URL
            - extracted_info (str): The extracted information
            - error (str): Error message if the operation failed
            - scrape_stats (Dict): Statistics about the scraped content
            - model_used (str): The model used for summarization
            - tokens_used (int): Number of tokens used (if available)
    """
    extracted_result = await extract_info_with_llm(
        content=extracted_info,
        info_to_extract=info_to_extract,
        max_tokens=8192,
    )

    # Combine results
    return {
        "success": extracted_result["success"],
        "extracted_info": extracted_result["extracted_info"],
        "error": extracted_result["error"],
        "model_used": extracted_result["model_used"],
        "tokens_used": extracted_result["tokens_used"],
    }

EXTRACT_INFO_PROMPT = """You are given a piece of content and the requirement of information to extract. Your task is to extract the information specifically requested. Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.
6. Do not include url of webpages.

CONTENT TO ANALYZE:
{}"""


def get_prompt_with_truncation(
    info_to_extract: str, content: str, truncate_last_num_chars: int = -1
) -> str:
    if truncate_last_num_chars > 0:
        content = content[:-truncate_last_num_chars] + "[...truncated]"

    # Prepare the prompt
    prompt = EXTRACT_INFO_PROMPT.format(info_to_extract, content)
    return prompt


async def extract_info_with_llm(
    content: str,
    info_to_extract: str,
    max_tokens: int = 8192,
) -> Dict[str, Any]:
    """
    Summarize content using an LLM API.

    Args:
        content (str): The content to summarize
        info_to_extract (str): The specific types of information to extract (usually a question)
        max_tokens (int): Maximum tokens for the response

    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Whether the operation was successful
            - extracted_info (str): The extracted information
            - error (str): Error message if the operation failed
            - model_used (str): The model used for summarization
            - tokens_used (int): Number of tokens used (if available)
    """

    # Get summary llm name from environment
    summary_llm_name = llm_name
    if not summary_llm_name:
        return {
            "success": False,
            "extracted_info": "",
            "error": "SUMMARY_LLM_NAME environment variable is not set",
            "model_used": summary_llm_name,
            "tokens_used": 0
        }
    
    # Get summary llm url from environment
    summary_llm_url = llm_base_url
    if not summary_llm_url:
        return {
            "success": False,
            "extracted_info": "",
            "error": "SUMMARY_LLM_URL environment variable is not set",
            "model_used": summary_llm_name,
            "tokens_used": 0
        }

    # Validate input
    if not content or not content.strip():
        return {
            "success": False,
            "extracted_info": "",
            "error": "Content cannot be empty",
            "model_used": summary_llm_name,
            "tokens_used": 0,
        }
    prompt = get_prompt_with_truncation(info_to_extract, content)
    # Prepare the payload
    payload = {
        "model": summary_llm_name,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        # "temperature": 0.7,
        # "top_p": 0.8,
        # "top_k": 20,
    }
    client = AsyncOpenAI(api_key="EMPTY",
    base_url= summary_llm_url,
    )
    # Prepare headers

    try:
        # Retry configuration
        connect_retry_delays = [1, 2, 4, 8]

        for attempt, delay in enumerate(connect_retry_delays, 1):
            try:
                # Make the API request using async client
                completion = await client.chat.completions.create(
                    model=summary_llm_name,
                    messages=payload["messages"],
                )

                # Success, exit retry loop
                break

            except aiohttp.ClientConnectorError as e:
                # connection error, retry
                if attempt < len(connect_retry_delays):
                    delay = connect_retry_delays[attempt]
                    logger.warning(
                        f"Jina Scrape and Extract Info: Connection error: {e}, {delay}s before next attempt"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        "Jina Scrape and Extract Info: Connection retry attempts exhausted"
                    )
                    raise e

            except aiohttp.ClientError as e:
                logger.error(
                    f"Jina Scrape and Extract Info: HTTP error for LLM API: {e}"
                )
                raise

            except Exception as e:
                logger.error(
                    f"Jina Scrape and Extract Info: Unknown request exception: {e}"
                )
                raise e

    except Exception as e:
        error_msg = f"Jina Scrape and Extract Info: Unexpected error during LLM API call: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": summary_llm_name,
            "tokens_used": 0,
        }

    # Parse the response
    try:
        response_data = completion.choices[0].message
    except json.JSONDecodeError as e:
        error_msg = (
            f"Jina Scrape and Extract Info: Failed to parse LLM API response: {str(e)}"
        )
        logger.error(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": summary_llm_name,
            "tokens_used": 0,
        }

    logger.info(
        f"Jina Scrape and Extract Info: Info to extract: {info_to_extract}, LLM Response data: {response_data}"
    )

    try:
        # 尝试获取摘要内容
        summary = response_data.content
        # 提取 token 使用量（如果可用）
        tokens_used = 0
        if hasattr(completion, 'usage') and completion.usage:
            tokens_used = getattr(completion.usage, 'total_tokens', 0)
        return {
            "success": True,
            "extracted_info": summary,
            "error": "",
            "model_used": summary_llm_name,
            "tokens_used": tokens_used,
        }
    except Exception as e:
        # 若获取摘要失败，返回错误处理
        error_msg = f"Jina Scrape and Extract Info: Failed to get summary from LLM API response: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "extracted_info": "",
            "error": error_msg,
            "model_used": summary_llm_name,
            "tokens_used": 0,
        }


if __name__ == "__main__":
    # Example usage and testing
    pass
