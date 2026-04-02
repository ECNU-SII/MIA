# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import io
import json
import logging
import os
import random
import re
import torch
import requests
from openai import OpenAI
from PIL import Image

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

from local_search.prompt import *



logger = logging.getLogger(__name__)

openai_api_key = "EMPTY"
openai_api_base = os.getenv("JUDGE_URL")
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model_name = "qwen"


prompt = PROMPT_TEXT_ONLY
prompt_image = PROMPT_TEXT_IMAGE



class CustomRLHFDataset(RLHFDataset):
    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        question = row_dict[self.prompt_key][0]["content"]
        plan = row_dict["plan"].replace("<image>", "")
        plan_prompt = f"\nHere is a guide for your reference:\n{plan}\nBegin your answer:\n"
        row_dict[self.prompt_key] = [
            {
                "role": "system",
                # We don't need tool description, because custom_chat_template will add it.
                "content": (
                    "You are a helpful assistant. You can call functions to assist with the user query. "
                    "Important: You must call only one function at a time. After each function call, "
                    "wait for the execution result before making the next function call if needed."
                ),
            },
            {
                "role": "user",
                "content": prompt_image + row_dict[self.prompt_key][0]["content"] + plan_prompt,
            },
        ]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                image = Image.open(io.BytesIO(row_dict_images)).convert("RGB")
                image = image.resize((448, 448))  # ✅ 限制分辨率
                images = [image]
                multi_modal_data["image"] = images
            model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")
            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)
                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        if "category" in row_dict:
            if row_dict["category"] is not None:
                row_dict["extra_info"]["category"]=row_dict["category"]
        if "level" in row_dict:
            if row_dict["level"] is not None:
                row_dict["extra_info"]["level"]=row_dict["level"]
        if "data_source" in row_dict:
            if row_dict["data_source"] is not None:
                row_dict["extra_info"]["data_source"]=row_dict["data_source"]
        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        row_dict["extra_info"]["question"] = question
        row_dict["extra_info"]["ground_truth"] = row_dict["reward_model"]["ground_truth"]
        row_dict["extra_info"]["plan"] = plan
        row_dict["extra_info"]["mem_context1"] = row_dict["memories_context"]
        row_dict["extra_info"]["mem_context2"] = row_dict["memories_context_2"]
        row_dict["extra_info"]["modality"] = row_dict["modality"]
        row_dict["extra_info"]["data_id"] = row_dict["data_id"]
        tools_kwargs = {
            "web_image_to_image_search": {
                "create_kwargs": {"ids": row_dict['data_id']},
                # "execute_kwargs": {},
                # "calc_reward_kwargs": {},
                # "release_kwargs": {},
            },
        }
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["agent_name"] = "tool_agent"
        return row_dict
    



def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Compute reward score for model solutions with robust handling of various formats.

    The score is a weighted combination of three components, ranging from 0 to 1:
    - Accuracy Score (0.7 weight): Whether the final answer is semantically correct.
    - Tool Score (0.2 weight): Whether the model made the correct decision to use (or not use) a tool.
    - Format Score (0.1 weight): Whether the output follows the expected tag format.
    """

    # Initialize tracking variables
    is_format_error = False

    # 1. Check <think> tag format
    count_think_1 = solution_str.count("<think>")
    count_think_2 = solution_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    # 2. Extract answer text with multiple fallback strategies
    answer_text = ""
    predict_no_think = (
        solution_str.split("</think>")[-1].strip() if "</think>" in solution_str else solution_str.strip()
    )

    # Check <answer> tag format
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    # Strategy 1: Try to extract from <answer> tags
    answer_match = re.findall(r"<answer>(.*?)</answer>", predict_no_think, re.DOTALL)
    if answer_match:
        answer_text = answer_match[-1].strip()
    else:
        is_format_error = True
        # Strategy 2: Fallback to content after tool responses
        tool_response_match = re.search(
            r"</tool_response>\s*assistant\s*\n(.*?)$", predict_no_think, re.DOTALL | re.MULTILINE
        )
        if tool_response_match:
            answer_text = tool_response_match.group(1).strip()
        else:
            # Strategy 3: Fallback to content after </think> tag
            if "</think>" in solution_str:
                remaining_content = predict_no_think
                remaining_content = re.sub(r"<tool_call>.*?</tool_call>", "", remaining_content, flags=re.DOTALL)
                remaining_content = re.sub(r"<tool_response>.*?</tool_response>", "", remaining_content, flags=re.DOTALL)
                remaining_content = re.sub(r"\b(user|assistant)\b", "", remaining_content)
                answer_text = remaining_content.strip()
            else:
                # Strategy 4: Use the entire solution as a last resort
                answer_text = solution_str.strip()

    # Clean up answer text
    answer_text = answer_text.strip()
    if not answer_text:
        is_format_error = True
        answer_text = solution_str.strip()

    accuracy_score = extra_info["acc_score"]
    tool_call_counts = {"search":0, "web_image_to_image_search":0, "replan":0}
    segments = re.split(r'I have a revised plan for you to follow:\n', solution_str)
    total_tool_score = 0
    num_segments = len(segments)
    for segment in segments:
        tool_call_matches = re.findall(r"<tool_call>\s*({.*?})\s*</tool_call>", segment, re.DOTALL)
        illegal_tool_used = False
        def _validate_search(arguments):
            if not isinstance(arguments, dict):
                return False
            if set(arguments.keys()) != {"query_list"}:
                return False
            query_list = arguments.get("query_list")
            if not isinstance(query_list, list) or not query_list:
                return False
            return all(isinstance(item, str) and item.strip() for item in query_list)
        def _validate_web_image(arguments):
            if not isinstance(arguments, dict):
                return False
            if set(arguments.keys()) != {"img_idx"}:
                return False
            img_idx = arguments.get("img_idx")
            return isinstance(img_idx, str) and img_idx == "0"
        allowed_tool_validators = {
            "search": _validate_search,
            "web_image_to_image_search": _validate_web_image,
        }
        has_web_image_to_image_search_usage = False
        has_search_usage = False
        for tool_payload in tool_call_matches:
            try:
                tool_info = json.loads(tool_payload)
            except json.JSONDecodeError:
                illegal_tool_used = True
                continue
            tool_name = tool_info.get("name")
            if not tool_name or tool_name not in allowed_tool_validators:
                illegal_tool_used = True
                continue
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            if tool_name == "web_image_to_image_search":
                has_web_image_to_image_search_usage = True
                if tool_call_counts[tool_name] > 1:
                    illegal_tool_used = True
                    continue
            if tool_name == "search":
                has_search_usage = True
            arguments = tool_info.get("arguments")
            validator = allowed_tool_validators[tool_name]
            if not validator(arguments):
                illegal_tool_used = True
        if illegal_tool_used:
            is_format_error = True
        if has_web_image_to_image_search_usage or has_search_usage:
            total_tool_score += 1.0
        else:
            total_tool_score += 0.0

    average_tool_score = total_tool_score / num_segments if num_segments > 0 else 0.0
    format_score = 0.0 if is_format_error else 1.0    
    
    # Log debug information for problematic cases
    if is_format_error or not answer_text:
        logger.debug(
            f"Format issue detected:\n"
            f"Solution: {solution_str[:200]}...\n"
            f"Extracted answer: '{answer_text}'\n"
            f"Format error: {is_format_error}\n"
            f"Tool usage: {average_tool_score}"
        )

    final_score = (0.7 * accuracy_score) + (0.2 * average_tool_score) + (0.1 * format_score)

    return final_score

