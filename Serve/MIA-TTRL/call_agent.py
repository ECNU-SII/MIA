import json
import json5
import os
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer 
from datetime import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
from prompt import *
import time
import asyncio
import base64
from PIL import Image
import io
import requests
from tool_search_local import *
# from tool_serper import *
from tool_image import *



PROMPT_IMAGE = USER_PROMPT_IMAGE_SHORT
PROMPT_TEXT = USER_PROMPT_TEXT_NEW
SYSTEM_PROMPT_IMAGE = SYSTEM_PROMPT_IMAGE
SYSTEM_PROMPT_TEXT = SYSTEM_PROMPT_TEXT

OBS_START = '<tool_call>'
OBS_END = '\n<tool_call>'

MAX_LLM_CALL_PER_RUN = int(os.environ.get('MAX_LLM_CALL_PER_RUN', 20))

TOOL_CLASS = [
    Search(),
    ImageSearch(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

import random
import datetime

openai_api_key = os.environ.get('OPENAI_API_KEY', 'EMPTY')
vllm_url = os.environ.get('AGENT_URL', 'http://127.0.0.1:8000/v1')
agent_client = OpenAI(
    api_key=openai_api_key,
    base_url=vllm_url,
    timeout=600.0,
)


import time
import random
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError


def normalize_image_url_for_vllm(raw_input):
    if raw_input is None:
        return None
    if not isinstance(raw_input, str):
        raw_input = str(raw_input)
    cleaned = ''.join(raw_input.split())
    if not cleaned:
        return None
    valid_prefixes = ["http://", "https://", "file://", "data:"]
    cleaned_lower = cleaned.lower()
    for prefix in valid_prefixes:
        if cleaned_lower.startswith(prefix):
            return raw_input
    if cleaned.startswith('/9j/'):
        mime_type = "image/jpeg"
    elif cleaned.startswith('iVBORw0KG'):
        mime_type = "image/png"
    elif cleaned.startswith('UklGR'):
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"
    return f"data:{mime_type};base64,{cleaned}"


def call_vllm_server(
    messages,
    model="qwen",
    max_tries=10,
    temperature=0,
    top_p=1.0,
    max_tokens=10000,
    presence_penalty=1.1,
):
    base_sleep_time = 1
    for attempt in range(max_tries):
        try:
            print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
            chat_response = agent_client.chat.completions.create(
                model=model,
                messages=messages,
                stop=["\n<tool_response>", "<tool_response>"],
                temperature=temperature,
                top_p=top_p,
                logprobs=True,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
            )
            content = chat_response.choices[0].message.content
            if content and content.strip():
                print("--- Service call successful, received a valid response ---")
                return content.strip()
            else:
                print(f"Warning: Attempt {attempt + 1} received an empty response.")

        except (APIError, APIConnectionError, APITimeoutError) as e:
            print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
        except Exception as e:
            print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")
        if attempt < max_tries - 1:
            sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
            sleep_time = min(sleep_time, 30)
            print(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        else:
            print("Error: All retry attempts have been exhausted. The call has failed.")
    return "vllm server error!!!"


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")

def encode_image_to_base64(image_path_or_pil):
    """将图像编码为base64字符串"""
    if isinstance(image_path_or_pil, str):
        # 如果是文件路径
        with open(image_path_or_pil, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image_path_or_pil, Image.Image):
        # 如果是PIL图像对象
        buffer = io.BytesIO()
        image_path_or_pil.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        raise ValueError("Unsupported image type")
        
def create_multimodal_message(role, text_content, images=None):
    """创建多模态消息"""
    content = []
    images = images or []
    if not isinstance(images, list):
        images = [images]
    image_index = 0
    if text_content:
        # 按<image>分割文本
        text_parts = text_content.split('<image>')
        for i, part in enumerate(text_parts):
            if part:
                content.append({
                    "type": "text",
                    "text": part
                })
            # 除了最后一个分割部分，其他部分后面都可能有图像
            if i < len(text_parts) - 1 and image_index < len(images):
                image = images[image_index]
                if isinstance(image, str):
                    # 假设是base64编码的字符串
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": normalize_image_url_for_vllm(image)
                        }
                    })
                elif isinstance(image, Image.Image):
                    # PIL图像对象
                    base64_image = encode_image_to_base64(image)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image;base64,{base64_image}"
                        }
                    })
                elif isinstance(image, dict) and "url" in image:
                    # 已经是正确格式的图像URL
                    content.append({
                        "type": "image_url",
                        "image_url": image
                    })
                image_index += 1

    return {
        "role": role,
        "content": content
    }


def custom_call_tool(tool_name: str, tool_args: dict, **kwargs):
    if tool_name in TOOL_MAP:
        tool_args["params"] = tool_args
        if "python" in tool_name.lower():
            result = TOOL_MAP['PythonInterpreter'].call(tool_args)
            result_images = []
        elif tool_name == "parse_file":
            params = {"files": tool_args["files"]}
            raw_result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path="./eval_data/file_corpus"))
            result = raw_result
            result_images = []
            if not isinstance(raw_result, str):
                result = str(raw_result)
        elif tool_name == "web_image_to_image_search":
            # 图像搜索工具返回多模态结果
            tool_args["cache_id"] = kwargs["cache_id"]
            raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
            print(raw_result)
            # 处理新的返回格式（字典形式）
            if isinstance(raw_result, dict):
                result = raw_result.get("text", str(raw_result))
                result_images = raw_result.get("images", [])
            # 兼容旧的返回格式（元组形式）
            elif isinstance(raw_result, tuple) and len(raw_result) >= 2:
                result = raw_result[0]  # 文本结果
                result_images = raw_result[1] if len(raw_result) > 1 else []  # 图像结果
            else:
                result = str(raw_result)
                result_images = []
        else:
            raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
            result = raw_result
            result_images = []
        return result, result_images

    else:
        return f"Error: Tool {tool_name} not found", []



def run_multimodal_dialogue(plan: str, item: Dict) -> List[Dict]:
    """
    根据给定的plan和item执行多轮对话工具调用
    
    Args:
        plan: 指导计划
        item: 包含question和images的字典
    
    Returns:
        List[Dict]: 对话消息数组
    """
    question = item.get('question', '')
    images = item.get('images', [])
    data_id = item.get('data_id', '')
    # 根据是否有图像选择不同的prompt
    plan_prompt = f"\nHere is a guide for your reference:\n{plan}\nBegin your answer:\n"
    
    if images:
        question_1 = SYSTEM_PROMPT_IMAGE + PROMPT_IMAGE + question 
    else:
        question_1 = SYSTEM_PROMPT_TEXT + PROMPT_TEXT + question
        
    if images:
        user_message = create_multimodal_message("user", question_1 + plan_prompt, images)
    else:
        user_message = {"role": "user", "content": question_1 + plan_prompt}
    messages = [user_message]
    num_llm_calls_available = MAX_LLM_CALL_PER_RUN
    round_num = 0
    while num_llm_calls_available > 0:
        round_num += 1
        num_llm_calls_available -= 1
        try:
            content = call_vllm_server(messages)
            if content and content.strip():
                if '<tool_response>' in content:
                    pos = content.find('<tool_response>')
                    content = content[:pos]
                if '<tool_call>' in content and '</tool_call>' in content:
                    first_end = content.find('</tool_call>')
                    content = content[:first_end + len('</tool_call>')]
                messages.append({"role": "assistant", "content": content.strip()})
                tool_call = None
                if '<tool_call>' in content:
                    tool_call = content.split('<tool_call>', 1)[1]
                    if '</tool_call>' in tool_call:
                        tool_call = tool_call.split('</tool_call>', 1)[0]
                    else:
                        tool_call = tool_call.strip()
                if tool_call is not None and tool_call.strip():
                    parsed_tool_call = json5.loads(tool_call)
                    if isinstance(parsed_tool_call, dict):
                        tool_name = parsed_tool_call.get('name', '')
                        tool_args = parsed_tool_call.get('arguments', {})
                    else:
                        tool_name = ''
                        tool_args = {}
                    result, result_images = custom_call_tool(tool_name, tool_args, cache_id=data_id)
                    if result_images:
                        tool_response_message = create_multimodal_message("user", f"<tool_response>\{result}\n<tool_response>", result_images)
                        
                    else:
                        tool_response_message = {"role": "user", "content": f"<tool_response>\{result}\n<tool_response>"}
                    messages.append(tool_response_message)
                else:
                    break
        except (APIError, APIConnectionError, APITimeoutError) as e:
            print(f"Error calling model: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    useful_messages = messages[1:]
    return useful_messages



def continue_multimodal_dialogue(replan: str, item: Dict) -> List[Dict]:
    # 复制原始消息，避免修改原始数据
    messages = item.get('messages', [])
    question = item.get('question', '')
    images = item.get('images', [])
    data_id = item.get('data_id', '')
    plan = item.get('plan', '')
    
    plan_prompt = f"\nHere is a guide for your reference:\n{plan}\nBegin your answer:\n"
    if images:
        question_1 = SYSTEM_PROMPT_IMAGE + PROMPT_IMAGE + question 
    else:
        question_1 = SYSTEM_PROMPT_TEXT + PROMPT_TEXT + question
    if images:
        user_message = create_multimodal_message("user", question_1 + plan_prompt, images)
    else:
        user_message = {"role": "user", "content": question_1 + plan_prompt}
    extended_messages = [user_message, ]
    extended_messages.extend(messages)
    temp_messages = []
    # 检查最后一条消息是否为用户消息，如果不是则添加replan作为用户输入
    if extended_messages and extended_messages[-1]["role"] == "assistant":
        def extract_before_answer(text):
            answer_index = text.find("<answer>")
            if answer_index == -1:
                return text.strip()
            result = text[:answer_index].strip()
            return result
        extended_messages[-1]["content"] = extract_before_answer(extended_messages[-1]["content"])
        
    replan_message = f"\nI have a revised plan for you to follow:\n{replan}\nPlease continue with this updated guidance:\n"
    extended_messages.append({"role": "user", "content": replan_message})
    num_llm_calls_available = MAX_LLM_CALL_PER_RUN
    round_num = 0
    while num_llm_calls_available > 0:
        round_num += 1
        num_llm_calls_available -= 1
        try:
            content = call_vllm_server(extended_messages)
            if content and content.strip():
                if '<tool_response>' in content:
                    pos = content.find('<tool_response>')
                    content = content[:pos]
                if '<tool_call>' in content and '</tool_call>' in content:
                    first_end = content.find('</tool_call>')
                    content = content[:first_end + len('</tool_call>')]
                
                extended_messages.append({"role": "assistant", "content": content.strip()})
                temp_messages.append({"role": "assistant", "content": content.strip()})
                # 检查是否包含工具调用
                tool_call = None
                if '<tool_call>' in content:
                    tool_call = content.split('<tool_call>', 1)[1]
                    if '</tool_call>' in tool_call:
                        tool_call = tool_call.split('</tool_call>', 1)[0]
                    else:
                        tool_call = tool_call.strip()
                if tool_call is not None and tool_call.strip():
                    parsed_tool_call = json5.loads(tool_call)
                    if isinstance(parsed_tool_call, dict):
                        tool_name = parsed_tool_call.get('name', '')
                        tool_args = parsed_tool_call.get('arguments', {})
                    else:
                        tool_name = ''
                        tool_args = {}
                    result, result_images = custom_call_tool(tool_name, tool_args, cache_id=data_id)
                    if result_images:
                        tool_response_message = create_multimodal_message("user", f"<tool_response>\{result}\n<tool_response>", result_images)
                    else:
                        tool_response_message = {"role": "user", "content": f"<tool_response>\{result}\n<tool_response>"}
                    extended_messages.append(tool_response_message)
                    temp_messages.append(tool_response_message)
                else:
                    break
            else:
                break
                
        except (APIError, APIConnectionError, APITimeoutError) as e:
            print(f"Error calling model in continue_multimodal_dialogue: {e}")
            break
        except Exception as e:
            print(f"Unexpected error in continue_multimodal_dialogue: {e}")
            break
    useful_messages = extended_messages[1:]
    return temp_messages, useful_messages

# 使用示例
if __name__ == "__main__":
    # 示例调用
    plan = "这是一个解决问题的计划"
    item = {
        "question": "这是问题",
        "images": []  # 可以是图像列表
    }
    
    # 首先运行初始对话
    messages = run_multimodal_dialogue(plan, item)
    print("初始对话结果:")
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    
    # 根据初始结果继续对话
    replan = "这是根据初始结果制定的补充计划"
    continued_messages = continue_multimodal_dialogue(replan, messages)
    print("\n继续对话结果:")
    print(json.dumps(continued_messages, indent=2, ensure_ascii=False))

