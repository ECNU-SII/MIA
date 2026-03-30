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
# from tool_search_local import *

from tool_serper import *
# from tool_search import *
# from tool_visit import *
from tool_image import *

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.environ.get('MAX_LLM_CALL_PER_RUN', 100))
PROMPT_IMAGE = USER_PROMPT_IMAGE_SHORT
PROMPT_TEXT = USER_PROMPT_TEXT_NEW

TOOL_CLASS = [
    Search(),
    ImageSearch(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

import random
import datetime


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")

def truncate_by_whitespace_words(text: str, max_words: int = 1024) -> str:
    if not text.strip():
        return text
    words = text.split()
    if len(words) > max_words:
        truncated_words = words[:max_words]
        truncated_words.append("... (Omitted part of the results returned by the tool)")
    else:
        truncated_words = words[:max_words]
    return " ".join(truncated_words)

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
                            "url": image
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

class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):
        if llm and isinstance(llm, dict):
            self.llm_generate_cfg = llm.get("generate_cfg", {})
            self.llm_local_path = llm.get("model", "")
        else:
            self.llm_generate_cfg = {}
            self.llm_local_path = ""

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content
    
    def call_server(self, msgs, planning_port, max_tries=10):
        
        openai_api_key = "EMPTY"
        openai_api_base = f"http://127.0.0.1:{planning_port}/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )
        
        base_sleep_time = 1 
        for attempt in range(max_tries):
            try:
                print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model="qwen",
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    logprobs=True,
                    max_tokens=self.llm_generate_cfg.get('max_tokens', 10000),
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1)
                )
                content = chat_response.choices[0].message.content

                # OpenRouter provides API calling. If you want to use OpenRouter, you need to uncomment line 89 - 90.
                # reasoning_content = "<think>\n" + chat_response.choices[0].message.reasoning.strip() + "\n</think>"
                # content = reasoning_content + content                
                
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
        
        return f"vllm server error!!!"

    def count_tokens(self, messages):
        tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path) 
        
        # 处理多模态消息
        processed_messages = []
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), list):
                # 多模态消息，需要特殊处理
                text_content = ""
                for content_item in message["content"]:
                    if isinstance(content_item, dict):
                        if content_item.get("type") == "text":
                            text_content += content_item.get("text", "")
                        elif content_item.get("type") == "image_url":
                            # 对于图像，估算token数量（通常图像占用较多token）
                            text_content += " [IMAGE] "  # 占位符，每个图像大约占用100-200个token
                
                processed_message = {
                    "role": message.get("role", "user"),
                    "content": text_content
                }
                processed_messages.append(processed_message)
            else:
                processed_messages.append(message)
        
        try:
            full_prompt = tokenizer.apply_chat_template(processed_messages, tokenize=False)
            tokens = tokenizer(full_prompt, return_tensors="pt")
            token_count = len(tokens["input_ids"][0])
        except Exception as e:
            print(f"Token counting error: {e}")
            # 如果出错，使用简单的字符数估算
            total_chars = 0
            for msg in processed_messages:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        total_chars += len(content)
                    else:
                        total_chars += len(str(content))
            token_count = total_chars // 4  # 粗略估算：4个字符约等于1个token
        
        return token_count

    def _run(self, data: Union[str, Dict], model: str, **kwargs) -> Dict:
        self.model=model
        plan_url=os.environ.get('PLAN_URL')
        replan_judge_url=os.environ.get('REPLAN_JUDGE_URL')
        replan_url=os.environ.get('REPLAN_URL')
        max_reflection_times=int(os.environ.get('MAX_REFLECTION_TIMES', 1))

        try:
            if isinstance(data, dict):
                item = data.get('item', {})
                question = item.get('question', '')
                ids = item.get('ids', '')
                # 检查是否有图像输入
                images = item.get('images', [])
                planning_port = data.get('planning_port', 6001)
                answer = item.get('answer', '')
            else:
                raise ValueError("Invalid data format")
        except: 
            if isinstance(data, dict) and 'item' in data:
                ids = item.get('ids', '')
                item = data['item']
                if isinstance(item, dict) and 'messages' in item:
                    raw_msg = item['messages'][1]["content"]
                    question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 
                else:
                    question = str(item)
                images = []
                planning_port = data.get('planning_port', 6001)
                answer = item.get('answer', '') if isinstance(item, dict) else ''
            else:
                question = str(data)
                images = []
                planning_port = 6001
                answer = ''
        try:
            image_caption = item.get('image_caption',[])
            data = {
                "data_id": ids, 
                "question": question, 
                "image_caption": image_caption, 
                "mem_top_k": 2,
            }
            response = requests.post(plan_url, json=[data,])
            results = response.json()
            result = results[0]
            plan = result["plan"]
            planner_messages = result["messages"]
            pos_indices = result["pos_indices"]
            neg_indices = result["neg_indices"]
            plan_prompt = f"\nHere is a guide for your reference:\n{plan}\nBegin your answer:\n"
        except:
            plan = ""
            planner_messages = []
            pos_indices = []
            neg_indices = []
            plan_prompt = ""
        
        memory_indices = []
        memory_indices.extend(pos_indices)
        memory_indices.extend(neg_indices)
        start_time = time.time()
        
        # 根据是否有图像输入选择使用文字或图像的prompt
        if images:
            question_1 = SYSTEM_PROMPT_IMAGE + PROMPT_IMAGE + question 
        else:
            question_1 = SYSTEM_PROMPT_TEXT + PROMPT_TEXT + question
        # question = DIRECT_PROMPT + question
        self.user_prompt = question_1
        system_prompt = SYSTEM_PROMPT_IMAGE if images else SYSTEM_PROMPT_TEXT
        cur_date = today_date()
        
        # 创建多模态消息
        if images:
            user_message = create_multimodal_message("user", question_1 + plan_prompt, images)
        else:
            user_message = {"role": "user", "content": question_1 + plan_prompt}
        
        messages = [user_message]

        # messages = [{"role": "system", "content": system_prompt}, user_message]
        # messages = [user_message, {"role": "assistant", "content":"<think>\n"}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        termination = ''
        reflection_times = 0
        reflection = ''
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                result = {
                    "question": question,
                    "answer": answer,
                    "id": ids,
                    "plan": plan,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "used_memory_indices": memory_indices,
                }
                return result
            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages, planning_port)
            print(f'Round {round}: {content}')
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
                if "python" in tool_call.lower():
                    try:
                        if '<code>' in tool_call and '</code>' in tool_call:
                            code_raw = tool_call.split('<code>', 1)[1].split('</code>', 1)[0].strip()
                        else:
                            code_raw = tool_call
                        result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                        result_images = []
                    except:
                        result = "[Python Interpreter Error]: Formatting error."
                        result_images = []

                else:
                    parsed_tool_call = json5.loads(tool_call)
                    if isinstance(parsed_tool_call, dict):
                        tool_name = parsed_tool_call.get('name', '')
                        tool_args = parsed_tool_call.get('arguments', {})
                    else:
                        tool_name = ''
                        tool_args = {}
                    result, result_images = self.custom_call_tool(tool_name, tool_args, cache_id=ids)

                
                # 创建多模态工具响应
                if result_images:
                    tool_response_message = create_multimodal_message("user", f"<tool_response>\n{result}\n</tool_response>", result_images)
                else:
                    tool_response_message = {"role": "user", "content": f"<tool_response>\n{result}\n</tool_response>"}
                
                messages.append(tool_response_message)
            else:
                termination = 'www'
                trace = f"### Question: {question}\n"
                useful_messages = messages[1:]
                j = 1
                for message in useful_messages:
                    role = message["role"]
                    content = message["content"]
                    if isinstance(content, str):
                        text_content = content
                    elif isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if item["type"] == "text":
                                text = item.get("text", "").strip()
                                text_parts.append(text)
                        text_content = " ".join(text_parts)
                    else:
                        text_content = str(content)
                    if role == "assistant":
                        trace += f"### Round {j}:\n"
                        trace += f"#### Agent Reasoning and Tool Call:\n{text_content}\n"
                        j += 1
                    else:
                        text_content = truncate_by_whitespace_words(text_content)
                        trace += f"#### Tool Call Return Results:\n{text_content}\n"
                
                try:
                    if reflection_times >= max_reflection_times:
                        break
                    response = requests.post(replan_judge_url, json=[{"memory_indices": memory_indices, "workflow": trace, "past_messages": planner_messages}])
                    results = response.json()
                    result = results[0]
                    need_replan = result["need_replan"]
                    planner_messages = result["messages"]
                    print("")
                    print("")
                    print(need_replan)
                    print("")
                    print("")
                    if "yes" in need_replan and reflection_times < max_reflection_times:

                        data = {
                            "data_id": ids, 
                            "question": question, 
                            "past_messages": planner_messages,
                            "image_caption": image_caption, 
                            # "mem_top_k": 1,
                            # "pass_num": 2 + reflection_times,
                            "mem_top_k": 0,
                            "pass_num": 2
                            
                        }
                        response = requests.post(replan_url, json=[data,])
                        results = response.json()
                        result = results[0]
                        replan = result["replan"]
                        planner_messages = result["messages"]
                        pos_indices = result["pos_indices"]
                        neg_indices = result["neg_indices"]
                        # memory_indices.extend(pos_indices)
                        # memory_indices.extend(neg_indices)
                        if messages and messages[-1]["role"] == "assistant":
                            def extract_before_answer(text):
                                answer_index = text.find("<answer>")
                                if answer_index == -1:
                                    return text.strip()
                                result = text[:answer_index].strip()
                                return result
                            messages[-1]["content"] = extract_before_answer(messages[-1]["content"])
                        replan_message = f"\nI have a revised plan for you to follow:\n{replan}\nPlease continue with this updated guidance:\n"
                        messages.append({"role": "user", "content": replan_message})
                        reflection_times += 1
                    else:
                        break
                except:
                    break

            
            
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            max_tokens = 110 * 1024
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")
                
                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                content = self.call_server(messages, planning_port)
                messages.append({"role": "assistant", "content": content.strip()})
                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "id": ids,
                    "plan": plan,
                    "prediction": prediction,
                    "trace": trace,
                    "used_memory_indices": memory_indices,
                }
                return result

        final_content = messages[-1]['content'] if messages else ''
        if '<answer>' in final_content:
            prediction = final_content.split('<answer>')[1].split('</answer>')[0]
        else:
            prediction = final_content
        if not termination:
            termination = 'exceed available llm calls' if num_llm_calls_available == 0 else 'answer not found'
                    
        # del messages[0]
        # Only keep the part after "Here is the question and image:\n" for question

        # 过滤掉含有图片的消息
        filtered_messages = []
        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                # 过滤掉 content 里类型为 image_url 的 item
                new_content = [
                    item for item in msg["content"]
                    if not (isinstance(item, dict) and item.get("type") == "image_url")
                ]
                filtered_msg = msg.copy()
                filtered_msg["content"] = new_content
                filtered_messages.append(filtered_msg)
            else:
                filtered_messages.append(msg)
        result = {
            "question": question,
            "answer": answer,
            "prediction": prediction,
            "messages": filtered_messages,
            "id": ids,
            "plan": plan,
            "rounds": round,
            "trace": trace,
            "used_memory_indices": memory_indices,
        }
        return result
    

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
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

    def extract_images_from_message(self, message):
        """从消息中提取图像"""
        images = []
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            for content_item in message["content"]:
                if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                    image_url_dict = content_item.get("image_url", {})
                    if isinstance(image_url_dict, dict):
                        image_url = image_url_dict.get("url", "")
                        if image_url.startswith("data:image"):
                            # 提取base64数据
                            base64_data = image_url.split(",")[1]
                            try:
                                image_data = base64.b64decode(base64_data)
                                image = Image.open(io.BytesIO(image_data))
                                images.append(image)
                            except Exception as e:
                                print(f"Error decoding image: {e}")
        return images

    def create_multimodal_response(self, text, images=None):
        """创建多模态响应消息"""
        if images:
            return create_multimodal_message("assistant", text, images)
        else:
            return {"role": "assistant", "content": text}


