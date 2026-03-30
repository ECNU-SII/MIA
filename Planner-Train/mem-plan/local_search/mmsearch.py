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

class CustomRLHFDataset(RLHFDataset):
    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        question = row_dict[self.prompt_key][0]["content"]
        
        memories_context = row_dict["memories_context"]
        memories_context_2 = row_dict["memories_context_2"]
        modality = row_dict["modality"]
        if modality == "text-only":
            row_dict[self.prompt_key] = [
                {
                    "role": "system",
                    "content": SYSTEM_PLAN_PROMPT,
                },
                {
                    "role": "user",
                    "content": PLAN_PROMPT.format(memory=memories_context, question=question),
                },
            ]
        else:
            row_dict[self.prompt_key] = [
                {
                    "role": "system",
                    "content": SYSTEM_PLAN_PROMPT,
                },
                {
                    "role": "user",
                    "content": PLAN_PROMPT_IMG.format(memory=memories_context, question=question),
                },
            ]

        messages = self._build_messages(row_dict)
        
        model_inputs = {}

        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.processor(text=[raw_prompt], return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
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

        # replan_prompt = REPLAN_PROMPT.format(question=question, memory=memories_context_2)
        replan_prompt = REPLAN_PROMPT.format(question=question)
        row_dict["extra_info"]["question"] = question
        row_dict["extra_info"]["plan"] = ""
        row_dict["extra_info"]["replan"] = ""
        row_dict["extra_info"]["modality"] = row_dict["modality"]
        row_dict["extra_info"]["data_id"] = row_dict["data_id"]
        row_dict["extra_info"]["replan_prompt"] = replan_prompt
        row_dict["extra_info"]["messages"] = []
        row_dict["agent_name"] = "multi_turn_agent"
        return row_dict


def normalize_text(text):
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip())

def extract_after_think(text):
    last_think_index_close = text.rfind("</think>")
    last_think_index_open = text.rfind("<think>")
    last_think_index = max(last_think_index_open, last_think_index_close)
    if last_think_index == -1:
        return text.replace("<|im_end|>", "").strip()
    start_index = last_think_index + len("</think>")
    result = text[start_index:]
    result = result.replace("<|im_end|>", "").strip()
    return result

def extract_answer(solution_str: str) -> tuple[str, bool]:
    predict_no_think = extract_after_think(solution_str)
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_no_think, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        tool_response_match = re.search(
            r"<tool_call>\s*assistant\s*\n(.*?)$", predict_no_think, re.DOTALL | re.MULTILINE
        )
        if tool_response_match:
            answer_text = tool_response_match.group(1).strip()
        else:
            if "</think>" in solution_str:
                remaining_content = predict_no_think
                remaining_content = re.sub(r"<tool_call>.*?<tool_call>", "", remaining_content, flags=re.DOTALL)
                remaining_content = re.sub(r"\b(?:user|assistant)\b", "", remaining_content, flags=re.IGNORECASE)
                answer_text = remaining_content.strip()
            else:
                answer_text = solution_str.strip()
    answer_text = answer_text.replace("<|im_end|>", "").strip()
    if not answer_text:
        answer_text = solution_str.strip()
    answer_text = normalize_text(answer_text)
    return answer_text



def judge_answer(question_text, ans, ground_truth) -> bool:
    if not ans.strip():
        return False
    user_prompt = JUDGE_PROMPT.format(
        question=question_text,
        correct_answer=ground_truth,
        response=ans
    )
    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": user_prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            temperature=0.0,
        )
        response = chat_response.choices[0].message.content.strip()
        response = response.replace("\\n", "\n").replace("\\r", "\r")
        cleaned = extract_after_think(response)
        cleaned = cleaned.strip()
        if cleaned == "A":
            judgement = True
        elif cleaned == "B":
            judgement = False
        elif cleaned == "C":
            judgement = False
        else:
            judgement = False
        return judgement
    except Exception as e:
        print("Judger Connect Error")
        pt = ans.lower().strip()
        gt = ground_truth.lower().strip()
        if gt in pt:
            return True
        else:
            return False


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    format_score = 0.0
    decision_score = 0.0
    messages = extra_info.get("messages", [])
    question_text = extra_info.get("question", "")
    if len(messages) not in (3, 6):
        return 0.0
    answer1 = extract_answer(messages[1])
    answer2 = extract_answer(messages[5]) if len(messages) == 6 else ""
    # 判断是否需要 replan
    do_replan = extract_after_think(messages[2]).lower()
    if do_replan not in ["yes", "no"]:
        format_score = 0.0
    else:
        format_score = 1.0
    # 新增格式检查：messages[0], messages[2], messages[4] 中 <think> 和 </think> 数量必须相等
    indices_to_check = [0, 2]
    if len(messages) == 6:
        indices_to_check.append(4)
    for idx in indices_to_check:
        msg = messages[idx]
        open_count = msg.count("<think>")
        close_count = msg.count("</think>")
        if open_count != close_count:
            format_score = 0.0
            break  # 一旦发现不匹配，直接判定格式错误
    # 归一化标准答案
    ground_truth = normalize_text(ground_truth)
    # 判断答案正确性
    is_answer1_correct = judge_answer(question_text, answer1, ground_truth)
    if len(messages) == 6:
        is_answer2_correct = judge_answer(question_text, answer2, ground_truth) 
    else:
        is_answer2_correct = is_answer1_correct
        
    if is_answer1_correct and "no" in do_replan:
        decision_score = 1.0
    elif (not is_answer1_correct) and "yes" in do_replan:
        decision_score = 1.0
    else:
        decision_score = 0.0
    # 计算准确率得分
    accuracy_score1 = 1.0 if is_answer1_correct else 0.0
    accuracy_score2 = 1.0 if is_answer2_correct else 0.0

    # 最终得分
    final_score = 0.7 * accuracy_score2 + 0.2 * accuracy_score1 + 0.05 * format_score + 0.05 * decision_score

    return final_score