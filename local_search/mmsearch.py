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
import torch
import requests
from openai import OpenAI
from PIL import Image
import asyncio
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
import time
from local_search.prompt import *


logger = logging.getLogger(__name__)

openai_api_key = "EMPTY"
openai_api_base = os.getenv("JUDGE_URL")
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model_name = "qwen"

memory_bank_save_url = os.getenv("MEMORY_BANK_SAVE_URL")


class CustomRLHFDataset(RLHFDataset):
    @classmethod
    def from_config_only(cls, tokenizer, processor, data_config):
        """
        Build a helper instance **without loading any parquet** (no ``dataframe``).
        Use :meth:`sample_from_row_dict` to tokenize client-provided rows for streaming TTRL.
        """
        from omegaconf import DictConfig, OmegaConf

        cfg = data_config if isinstance(data_config, DictConfig) else OmegaConf.create(data_config)
        self = cls.__new__(cls)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = cfg
        self.prompt_key = cfg.get("prompt_key", "prompt")
        self.image_key = cfg.get("image_key", "images")
        self.video_key = cfg.get("video_key", "videos")
        self.max_prompt_length = cfg.get("max_prompt_length", 1024)
        self.return_raw_chat = cfg.get("return_raw_chat", False)
        self.return_full_prompt = cfg.get("return_full_prompt", False)
        self.truncation = cfg.get("truncation", "error")
        self.apply_chat_template_kwargs = cfg.get("apply_chat_template_kwargs", {})
        self.dataframe = None  # not used
        return self

    def sample_from_row_dict(self, row_dict: dict) -> dict:
        """Same preprocessing as :meth:`__getitem__` but for one plain row dict (e.g. from HTTP JSON)."""
        import copy

        row_dict = copy.deepcopy(row_dict)
        return self._fill_sample_from_row(row_dict)

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        return self._fill_sample_from_row(row_dict)

    def _fill_sample_from_row(self, row_dict: dict) -> dict:
        question = row_dict[self.prompt_key][0]["content"]
        modality = row_dict["modality"]

        if "image_caption" in row_dict.keys():
            if row_dict["image_caption"]:
                image_caption = row_dict["image_caption"]
                if isinstance(image_caption, list):
                    image_caption = image_caption[0] if image_caption else ""
                else:
                    image_caption = image_caption if image_caption is not None else ""
            else:
                image_caption = ""
        else:
            image_caption = ""
        plan_prompt_format = PLAN_PROMPT
        if modality == "text-only":
            row_dict[self.prompt_key] = [
                {
                    "role": "system",
                    "content": SYSTEM_PLAN_PROMPT,
                },
            ]
            plan_prompt_format = PLAN_PROMPT
        else:
            row_dict[self.prompt_key] = [
                {
                    "role": "system",
                    "content": SYSTEM_PLAN_PROMPT,
                },
            ]
            plan_prompt_format = PLAN_PROMPT_IMG
        messages = self._build_messages(row_dict)
        
        model_inputs = {}

        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            model_inputs = self.processor(text=[raw_prompt], return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
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
        replan_prompt = REPLAN_PROMPT.format(question=question)
        row_dict["extra_info"]["slow_plan"] = ""
        row_dict["extra_info"]["slow_replan"] = ""
        row_dict["extra_info"]["question"] = question
        row_dict["extra_info"]["image_caption"] = image_caption
        row_dict["extra_info"]["plan_prompt_format"] = plan_prompt_format
        row_dict["extra_info"]["modality"] = row_dict["modality"]
        row_dict["extra_info"]["data_id"] = row_dict["data_id"]
        row_dict["extra_info"]["replan_prompt"] = replan_prompt
        row_dict["extra_info"]["messages"] = []
        row_dict["agent_name"] = "multi_turn_agent"
        return row_dict


def extract_after_think(text: str) -> str:
    last_think_index_close = text.rfind("</think>")
    last_think_index_open = text.rfind("<think>")
    last_think_index = max(last_think_index_open, last_think_index_close)
    if last_think_index == -1:
        return text.replace("<|im_end|>", "").strip()
    start_index = last_think_index + len("</think>")
    result = text[start_index:]
    result = result.replace("<|im_end|>", "").strip()
    return result


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    gt_reward_url: str | None = None,
) -> float:
    """
    GT 判分在独立服务 ``GT_REWARD_URL``（默认 ``http://127.0.0.1:6000``）按 ``data_id`` 查表；
    ``ground_truth`` 参数保留以兼容 reward manager，流式场景下应为空。

    memory bank 侧车仍在本进程触发。
    """
    extra_info = extra_info or {}
    _ = ground_truth
    base = (gt_reward_url or os.getenv("GT_REWARD_URL", "http://127.0.0.1:6000")).rstrip("/")
    payload = {
        "data_source": data_source,
        "solution_str": solution_str,
        "extra_info": extra_info,
        "data_id": extra_info.get("data_id"),
        "question": extra_info.get("question", ""),
        "image_caption": extra_info.get("image_caption", ""),
    }
    try:
        body = json.dumps(payload, default=str)
        r = requests.post(
            f"{base}/compute_gt_reward",
            data=body,
            headers={"Content-Type": "application/json"},
            timeout=600,
        )
        r.raise_for_status()
        out = r.json()
    except Exception as e:
        logger.error("GT reward HTTP failed: %s", e)
        return 0.0

    final_score = float(out.get("score", 0.0))
    judgement = out.get("judgement", "incorrect")
    messages = extra_info.get("messages", [])
    if len(messages) not in (3, 6):
        return final_score

    data_id = extra_info.get("data_id", "")
    plan = extract_after_think(messages[0]).strip()
    slow_plan = extra_info.get("slow_plan", "").strip()
    question = extra_info.get("question", "")
    image_caption = extra_info.get("image_caption", "").strip()
    used_memory_indices = extra_info.get("used_memory_indices", [])
    temp_messages = extra_info.get("temp_messages", [])
    data = {
        "data_id": data_id,
        "slow_plan": slow_plan,
        "plan": plan,
        "question": question,
        "image_caption": image_caption,
        "used_memory_indices": used_memory_indices,
        "temp_messages": temp_messages,
        "judgement": judgement,
    }
    get_user_response_from_url(memory_bank_save_url, data)
    return final_score







def get_user_response_from_url(url, data, max_retries=3, base_delay=1.0):
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                json=data,
                timeout=600,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
            
        except (requests.exceptions.Timeout, 
                requests.exceptions.RequestException,
                ValueError) as e:
            last_exception = e
            attempt_num = attempt + 1
            if isinstance(e, requests.exceptions.Timeout):
                err_type = "TIMEOUT"
                msg = f"Attempt {attempt_num}/{max_retries} failed: Timeout to {url}"
            elif isinstance(e, ValueError):
                err_type = "JSON_PARSE"
                msg = f"Attempt {attempt_num}/{max_retries} failed: Invalid JSON from {url} - {str(e)}"
            else:
                err_type = "NETWORK"
                msg = f"Attempt {attempt_num}/{max_retries} failed: {type(e).__name__} - {str(e)}"
            
            logger.warning(f"[{err_type}] {msg}")
            
            # 决定是否重试
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # 指数退避: 1s → 2s → 4s
                logger.info(f"Retrying in {delay:.1f}s... (attempt {attempt_num + 1}/{max_retries})")
                time.sleep(delay)
            else:
                # 最后一次失败，记录最终错误
                logger.error(
                    f"All {max_retries} attempts failed for {url}. "
                    f"Last error [{err_type}]: {str(e)}"
                )
    return "Default user response due to persistent network error"