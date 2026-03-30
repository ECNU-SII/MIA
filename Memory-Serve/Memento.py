import os
import json
import argparse
import re
from flask import Flask, request, jsonify
from openai import OpenAI, AzureOpenAI
from transformers import AutoTokenizer
import dotenv
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import torch.nn.functional as F
import time
from typing import List, Dict, Any, Tuple, Optional
from memory_functions import get_memory_tool_schemas
from tqdm import tqdm
import http.client
import json



dotenv.load_dotenv()
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_NAME = "qwen"
SERVER_URL = None
SAVE_PATH = None



SYSTEM_PROMPT = """
You are a planning assistant in a three-step loop:  

1. **Plan**: Given a goal and background info, output a clear action plan.  
2. **Evaluate**: Given an execution trace, decide if replanning is needed.  
3. **Replan (if needed)**: With new reference memories, provide a revised plan targeting unmet goals.  

Keep responses concise and action-focused.
"""

planning_system_prompt = """
You are the META-PLANNER in a hierarchical AI system. A user will ask a high-level question. **First**: break the problem into a *minimal sequence*\n

The agent has access to the following tools:
- `search`: perform text-based web queries to retrieve external information.

### [Relevant Memories] 
Each memory indicates whether it was **correct** or **incorrect**, along with its plan:
{memory}

Output should be **clear and concise**, and present your response as a step-by-step plan.

**[Question]** (Global Objective): 
{question}
"""

planning_system_prompt_img = """
You are the META-PLANNER in a hierarchical AI system. A user will ask a high-level question. **First**: break the problem into a *minimal sequence*\n

The agent has access to the following tools:
- `web_image_to_image_search`: find visually similar images online (Can only be used once).
- `search`: perform text-based web queries to retrieve external information.

### [Relevant Memories] 
Each memory indicates whether it was **correct** or **incorrect**, along with its plan:
{memory}

Output should be **clear and concise**, and present your response as a step-by-step plan.

**[Question]** (Global Objective): 
{question}
"""

judge_prompt = """
{trace}

Based on the sufficiency of the reasons and evidence, assess whether the agent needs to replan. Only in the following cases:
- Tool support is missing or unclear,
- Reasoning has gaps or assumptions,
- Answer does not fully address the question,
- Any doubt exists on correctness or clarity.

Use `replan` over accepting borderline solutions.

The internal thinking block (<think>..</think>) should be extremely concise.

Respond with exactly one word: "yes" to trigger replanning, "no" to accept the current solution.
"""

replanning_system_prompt = """
### Reflection and Replanning

In the last conversation, you indicated that reflect and replan needed to be performed. Now you need to perform them.

**The original question:** {question}

**The only tool you can recommend is `search`.**

Your task is to:
- Analyze the memories and reference their useful strategies.
- Analyze the current workflow so far to understand what has been attempted and why it failed.
- Recommend a **clear, generalizable work plan or action strategy** that builds on existing work, avoids past mistakes, and addresses the remaining challenges.

### Critical Requirements:
- **Leverage all completed steps from the current workflow**. Do not repeat searches, queries, or reasoning already performed.
- **Identify the exact failure point** and propose the next logical step toward the goal.

### Output should:
1. Be clear and concise, no more than 200 words, and present your response as a step-by-step supplementary plan.
2. You must include **reflection** on the previous failure in the thinking process (<think>...</think>), but **do not mention them in the final output**.
3. Don't give the answer directly, but provide **a supplementary plan**.
4. Prohibit generating content unrelated to the plan.
5. Your response must not contain any factual information.

Your output should only contain the guideline, with no additional explanations.
"""

CATEGORIES = ["location", "human", "time", "career", "process", "definition", "event", "function", "property", "others"]
MODALITIES = ["text-only", "text-image"]
MEMORY_URL = os.getenv("MEMORY_URL")
api_key = "EMPTY"
memory_client = OpenAI(base_url=MEMORY_URL, api_key=api_key)
PLAN_URL = os.getenv("PLAN_URL")
plan_client = OpenAI(base_url=PLAN_URL, api_key=api_key)

def filter_result(result: str) -> str:
    marker = "</think>"
    if marker in result:
        return result.split(marker, 1)[1].strip()
    else:
        return result.strip()



def llm_classify_question(question: str) -> str:
    return "others"
    


class MemoryBucket:
    def __init__(self, device: str):
        self.device = device
        self.question_embeddings: Optional[torch.Tensor] = None   # (N, D)
        self.caption_embeddings: Optional[torch.Tensor] = None    # (N, D)
        self.memory_data: List[Dict[str, Any]] = []               # no data_id

    def add_entry(self, q_vec: torch.Tensor, ic_vec: torch.Tensor, entry: Dict[str, Any]):
        q_vec = q_vec.unsqueeze(0)
        ic_vec = ic_vec.unsqueeze(0)
        if self.question_embeddings is None:
            self.question_embeddings = q_vec
            self.caption_embeddings = ic_vec
        else:
            self.question_embeddings = torch.cat([self.question_embeddings, q_vec], dim=0)
            self.caption_embeddings = torch.cat([self.caption_embeddings, ic_vec], dim=0)
        if 'usage_count' not in entry.keys():
            entry['usage_count'] = 1
        if 'success_count' not in entry.keys():
            entry['success_count'] = 1
        if 'win_rate' not in entry.keys():
            entry['win_rate'] = 1.0
        self.memory_data.append(entry)

    def update_entry(self, idx: int, q_vec: torch.Tensor, ic_vec: torch.Tensor, entry: Dict[str, Any]):
        self.question_embeddings[idx] = q_vec
        self.caption_embeddings[idx] = ic_vec
        entry['usage_count'] = 1
        entry['success_count'] = 1
        entry['win_rate'] = 1.0
        self.memory_data[idx] = entry

    def update_memory_stats(self, indices: List[int], success: bool):
        for idx in indices:
            if 0 <= idx < len(self.memory_data):
                self.memory_data[idx]['usage_count'] += 1
                if success:
                    self.memory_data[idx]['success_count'] += 1
                if self.memory_data[idx]['usage_count'] > 0:
                    win_rate = self.memory_data[idx]['success_count'] / self.memory_data[idx]['usage_count']
                    self.memory_data[idx]['win_rate'] = round(win_rate, 4)
    
    def get_size(self) -> int:
        return len(self.memory_data)

    def clear(self):
        self.question_embeddings = None
        self.caption_embeddings = None
        self.memory_data.clear()


class MemoryProcessor:
    def __init__(self):
        """Initialize the OpenAI client based on model configuration."""
        self.model_name = MODEL_NAME
        bert_path = "/your_path/bert/sup-simcse-bert-base-uncased"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bert_model_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model_encoder = AutoModel.from_pretrained(bert_path).to(self.device)
        self.bert_model_encoder.eval()
        self.yunwu_key = os.getenv("YUNWU_KEY")
        self.client = plan_client
        self.start_client = memory_client
        self.question_weight = 1.0
        self.caption_weight = 0.0
        self.cosine_weight = 1.0
        self.win_rate_weight = 0.0
        self.memory_store: Dict[str, Dict[str, MemoryBucket]] = {
            modality: {cat: MemoryBucket(self.device) for cat in CATEGORIES}
            for modality in MODALITIES
        }
        if not "qwen" in self.model_name:
            self.conn = http.client.HTTPSConnection("yunwu.ai", timeout=300)

    def _determine_modality(self, image_caption: str) -> str:
        return "text-image" if image_caption.strip() else "text-only"

    def _classify_question_type(self, question: str) -> str:
        return llm_classify_question(question)

    def direct_store_memory(self, data_id, question: str, image_caption: str, trace: str, judgement: str, plan: str, used_memory_indices):
        modality = self._determine_modality(image_caption)
        category = self._classify_question_type(question)
        bucket = self.memory_store[modality][category]
        q_vec = self._encode_text(question)
        ic_vec = self._encode_text(image_caption) if image_caption.strip() else torch.zeros_like(q_vec)
        workflow_summary = trace
        new_entry = {
            "data_id": data_id,
            "question": question,
            "image_caption": image_caption,
            "workflow_summary": workflow_summary,
            "plan": plan,
            "judgement": judgement
        }
        if bucket.get_size() > 0:
            q_query_vec = q_vec.unsqueeze(0)
            with torch.no_grad():
                sim_q = F.cosine_similarity(q_query_vec, bucket.question_embeddings)  # (N,)
                if image_caption.strip():
                    ic_query_vec = ic_vec.unsqueeze(0)
                    sim_ic = F.cosine_similarity(ic_query_vec, bucket.caption_embeddings)
                    total_sim = self.question_weight * sim_q + self.caption_weight * sim_ic
                else:
                    total_sim = sim_q
            max_sim, max_idx = torch.max(total_sim, dim=0)
            if max_sim.item() >= 0.9999:
                existing_entry = bucket.memory_data[max_idx.item()]
                if existing_entry["judgement"] == "incorrect":
                    bucket.update_entry(max_idx.item(), q_vec, ic_vec, new_entry)
                    logger.info(f"Updated incorrect memory in {modality}/{category}")
                else:
                    old_len = self._count_words(existing_entry["workflow_summary"])
                    new_len = self._count_words(workflow_summary)
                    if new_len < old_len:
                        bucket.update_entry(max_idx.item(), q_vec, ic_vec, new_entry)
                        logger.info(f"Replaced correct memory with shorter version in {modality}/{category}")
                    else:
                        logger.info(f"Kept existing correct memory (shorter or equal)")
                return
        bucket.add_entry(q_vec, ic_vec, new_entry)
        logger.info(f"Added new memory to {modality}/{category}. Total: {bucket.get_size()}")
        print(used_memory_indices)
        if used_memory_indices:
            success = (judgement == "correct")
            bucket.update_memory_stats(used_memory_indices, success)

        
        
    def _encode_text(self, text: str) -> torch.Tensor:
        text = text.strip()
        if not text:
            return torch.zeros(self.bert_model_encoder.config.hidden_size, device=self.device)
        inputs = self.bert_model_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model_encoder(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding

    def _count_words(self, text: str) -> int:
        return len(text.split())
    
    def retrieve_balanced_memories(
        self,
        query_question: str,
        query_image_caption: str,
        pos_top_k: int = 3,
        neg_top_k: int = 3,
        pos_pass_num: int = 0,
        neg_pass_num: int = 0
    ) -> Tuple[List[Tuple[int, Dict]], List[Tuple[int, Dict]]]:
        modality = self._determine_modality(query_image_caption)
        category = self._classify_question_type(query_question)
        bucket = self.memory_store[modality][category]
        if bucket.get_size() == 0:
            return [], []
        q_query_vec = self._encode_text(query_question).unsqueeze(0)
        ic_query_vec = self._encode_text(query_image_caption).unsqueeze(0) if query_image_caption.strip() else None
        with torch.no_grad():
            sim_q = F.cosine_similarity(q_query_vec, bucket.question_embeddings)
            if query_image_caption.strip():
                sim_ic = F.cosine_similarity(ic_query_vec, bucket.caption_embeddings)
                total_sim = self.question_weight * sim_q + self.caption_weight * sim_ic
            else:
                total_sim = sim_q
                
        win_rates = torch.tensor([entry.get('win_rate', 0.0) for entry in bucket.memory_data], dtype=torch.float32)
        total_sim_normalized = (total_sim - total_sim.min()) / (total_sim.max() - total_sim.min() + 1e-8)
        total_sim_normalized = total_sim_normalized.cpu()
        win_rates_normalized = win_rates
        combined_scores = self.cosine_weight * total_sim_normalized + self.win_rate_weight * win_rates_normalized
        candidates = [(combined_scores[i].item(), i, entry) for i, entry in enumerate(bucket.memory_data)]
        correct_candidates = [c for c in candidates if c[2]["judgement"] == "correct"]
        incorrect_candidates = [c for c in candidates if c[2]["judgement"] == "incorrect"]
        correct_candidates.sort(key=lambda x: x[0], reverse=True)
        incorrect_candidates.sort(key=lambda x: x[0], reverse=True)
        start_pos_idx = pos_pass_num
        end_pos_idx = start_pos_idx + pos_top_k
        correct_selected = [(c[1], c[2]) for c in correct_candidates[start_pos_idx:end_pos_idx]]
        start_neg_idx = neg_pass_num
        end_neg_idx = start_neg_idx + neg_top_k
        incorrect_selected = [(c[1], c[2]) for c in incorrect_candidates[start_neg_idx:end_neg_idx]]
        return correct_selected, incorrect_selected
        
    def get_memories_context(
        self, 
        question: str, 
        image_caption: str, 
        pos_top_k: int = 3, 
        neg_top_k: int = 3, 
        pos_pass_num: int = 0, 
        neg_pass_num: int = 0
    ) -> Tuple[str, List[int], List[int]]:  # 返回上下文、正面记忆索引列表、负面记忆索引列表
        try:
            pos_memories_with_idx, neg_memories_with_idx = self.retrieve_balanced_memories(
                query_question=question,
                query_image_caption=image_caption,
                pos_top_k=pos_top_k,
                neg_top_k=neg_top_k,
                pos_pass_num=pos_pass_num,
                neg_pass_num=neg_pass_num
            )
            pos_indices = [idx for idx, data in pos_memories_with_idx]
            neg_indices = [idx for idx, data in neg_memories_with_idx]
            pos_memories = [data for idx, data in pos_memories_with_idx]
            neg_memories = [data for idx, data in neg_memories_with_idx]
        except:
            pos_memories, neg_memories = [], []
            pos_indices, neg_indices = [], []
        
        memories_context = ""
        if pos_memories or neg_memories:
            memories_context += "\n### Retrieved Relevant Memories:\n"
            if pos_memories:
                memories_context += f"\n#### Positive Examples (Successful Strategies - {len(pos_memories)}):\n"
                for i, entry in enumerate(pos_memories, 1):
                    memories_context += (
                        f"\n--- Example {i} (Text-only modality) ---\n"
                        f"- Judgement: {entry['judgement']}\n"
                        f"- Question: {entry['question']}\n"
                        f"- Plan: {entry.get('plan', 'N/A')}\n"
                    )
            if neg_memories:
                memories_context += f"\n#### Negative Examples (Failure Lessons - {len(neg_memories)}):\n"
                for i, entry in enumerate(neg_memories, 1):
                    memories_context += (
                        f"\n--- Example {i} (Text-only modality) ---\n"
                        f"- Judgement: {entry['judgement']}\n"
                        f"- Question: {entry['question']}\n"
                        f"- Plan: {entry.get('plan', 'N/A')}\n"
                    )
        else:
            memories_context = "\n### Retrieved Relevant Memories:\nNone found."
        return memories_context, pos_indices, neg_indices
    
    
    def build_plan(self, question: str, image_caption: str, pos_top_k: int = 2, neg_top_k: int = 2, pos_pass_num: int = 0, neg_pass_num: int = 0):
        memories_context, pos_indices, neg_indices = self.get_memories_context(
            question=question,
            image_caption=image_caption,
            pos_top_k=pos_top_k,
            neg_top_k=neg_top_k,
            pos_pass_num=pos_pass_num,
            neg_pass_num=neg_pass_num
        )
        print("plan", len(pos_indices), len(neg_indices))
        if image_caption:
            user_prompt = planning_system_prompt_img.format(
                question=question,
                memory=memories_context
            )
        else:
            user_prompt = planning_system_prompt.format(
                question=question,
                memory=memories_context
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        if len(pos_indices) < pos_top_k or len(neg_indices) < neg_top_k:
            response = self.start_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=4096,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=4096,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        output = response.choices[0].message.content
        messages.append({"role": "assistant", "content": output})
        return output, messages, pos_indices, neg_indices

    def if_replan(self, prompt: str, past_messages):
        user_prompt = judge_prompt.format(
            trace=prompt,
        )
        messages = past_messages
        messages.append({"role": "user", "content": user_prompt})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        output = response.choices[0].message.content
        messages.append({"role": "assistant", "content": output})
        return output, messages

    def build_replan(self, question: str, past_messages, image_caption: str, pos_top_k: int = 2, neg_top_k: int = 2, pos_pass_num: int = 0, neg_pass_num: int = 0):
        memories_context, pos_indices, neg_indices = self.get_memories_context(
            question=question,
            image_caption=image_caption,
            pos_top_k=pos_top_k,
            neg_top_k=neg_top_k,
            pos_pass_num=pos_pass_num,
            neg_pass_num=neg_pass_num
        )
        print("replan", len(pos_indices), len(neg_indices))
        user_prompt = replanning_system_prompt.format(
            question=question,
            # memory=memories_context,
        )
        messages = past_messages
        messages.append({"role": "user", "content": user_prompt})
        if len(pos_indices) < pos_top_k or len(neg_indices) < neg_top_k:
            response = self.start_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=4096,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=4096,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        output = response.choices[0].message.content
        messages.append({"role": "assistant", "content": output})
        return output, messages, pos_indices, neg_indices

    def save_memory(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        serializable = {}
        for modality in MODALITIES:
            serializable[modality] = {}
            for cat in CATEGORIES:
                bucket = self.memory_store[modality][cat]
                serializable[modality][cat] = bucket.memory_data
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=4, ensure_ascii=False)

    def load_memory_from_file(self, file_path: str):
        if not os.path.exists(file_path):
            logger.warning(f"Memory file not found: {file_path}")
            return
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for modality in MODALITIES:
            for cat in CATEGORIES:
                self.memory_store[modality][cat].clear()
        total_loaded = 0
        for modality in MODALITIES:
            if modality not in data:
                continue
            for cat in CATEGORIES:
                if cat not in data[modality]:
                    continue
                entries = data[modality][cat]
                bucket = self.memory_store[modality][cat]
                for entry in tqdm(entries, desc=f"Loading {modality}/{cat}"):
                    try:
                        q = str(entry.get("question", ""))
                        ic = str(entry.get("image_caption", ""))
                        q_vec = self._encode_text(q)
                        ic_vec = self._encode_text(ic) if ic.strip() else torch.zeros_like(q_vec)
                        bucket.add_entry(q_vec, ic_vec, entry)
                        total_loaded += 1
                    except Exception as e:
                        logger.error(f"Failed to load entry in {modality}/{cat}: {e}")
        logger.info(f"Loaded {total_loaded} memories from {file_path}")


        
@app.route('/memory', methods=['POST'])
def memory():
    data = request.get_json()
    results = []
    for item in data:
        pos_top_k = int(item.get('mem_top_k', 3))
        neg_top_k = int(item.get('mem_top_k', 3))
        pos_pass_num = int(item.get('pass_num', 0))
        neg_pass_num = int(item.get('pass_num', 0))
        data_id = item.get('data_id', "").strip()
        question = item.get("question", "").strip()
        image_captions = item.get("image_caption", [])
        if isinstance(image_captions, list):
            if len(image_captions) > 0:
                image_caption = image_captions[0].strip()
            else:
                image_caption = ""
        else:
            image_caption = image_captions
            
        memories_context, pos_indices, neg_indices = processor.get_memories_context(
            question, image_caption, 
            pos_top_k=2, 
            neg_top_k=2, 
            pos_pass_num=0, 
            neg_pass_num=0
        )
        memories_context_2, pos_indices, neg_indices = processor.get_memories_context(
            question, image_caption, 
            pos_top_k=1, 
            neg_top_k=1, 
            pos_pass_num=2, 
            neg_pass_num=2
        )
        context, pos_indices, neg_indices = processor.get_memories_context(
            question, image_caption, 
            pos_top_k=pos_top_k, 
            neg_top_k=neg_top_k, 
            pos_pass_num=pos_pass_num, 
            neg_pass_num=neg_pass_num
        )
        result = {
            'context': context.strip(),
            'pos_indices': pos_indices,
            'neg_indices': neg_indices
        }
        results.append(result)
    return jsonify(results)


@app.route('/plan', methods=['POST'])
def plan():
    data = request.get_json()
    results = []
    for item in data:
        pos_top_k = int(item.get('mem_top_k', 2))
        neg_top_k = int(item.get('mem_top_k', 2))
        pos_pass_num = int(item.get('pass_num', 0))
        neg_pass_num = int(item.get('pass_num', 0))
        data_id = item.get('data_id', "").strip()
        question = item.get("question", "").strip()
        image_captions = item.get("image_caption", [])
        if isinstance(image_captions, list):
            if len(image_captions) > 0:
                image_caption = image_captions[0].strip()
            else:
                image_caption = ""
        else:
            image_caption = image_captions
        result, messages, pos_indices, neg_indices = processor.build_plan(
            question, image_caption,
            pos_top_k=pos_top_k,
            neg_top_k=neg_top_k,
            pos_pass_num=pos_pass_num,
            neg_pass_num=neg_pass_num
        )
        result = filter_result(result)
        result_with_indices = {
            'plan': result,
            'messages': messages,
            'pos_indices': pos_indices,
            'neg_indices': neg_indices
        }
        results.append(result_with_indices)
    return jsonify(results)

@app.route('/judge_replan', methods=['POST'])
def judge_replan():
    data = request.get_json()
    results = []
    for item in data:
        workflow = item.get("workflow", "")
        past_messages = item.get("past_messages", [])
        result, messages = processor.if_replan(workflow, past_messages)
        result = filter_result(result)
        result_with_indices = {
            'need_replan': result,
            'messages': messages
        }
        results.append(result_with_indices)
    return jsonify(results)

@app.route('/replan', methods=['POST'])
def replan():
    data = request.get_json()
    results = []
    for item in data:
        pos_top_k = int(item.get('mem_top_k', 1))
        neg_top_k = int(item.get('mem_top_k', 1))
        pos_pass_num = int(item.get('pass_num', 0))
        neg_pass_num = int(item.get('pass_num', 0))
        data_id = item.get('data_id', "")
        question = item.get("question", "").strip()
        past_messages = item.get("past_messages", [])
        image_captions = item.get("image_caption", [])
        if isinstance(image_captions, list):
            if len(image_captions) > 0:
                image_caption = image_captions[0].strip()
            else:
                image_caption = ""
        else:
            image_caption = image_captions
        result, messages, pos_indices, neg_indices = processor.build_replan(
            question, past_messages, image_caption, 
            pos_top_k=pos_top_k,
            neg_top_k=neg_top_k,
            pos_pass_num=pos_pass_num,
            neg_pass_num=neg_pass_num,
        )
        result = filter_result(result)
        result_with_indices = {
            'replan': result,
            'messages': messages,
            'pos_indices': pos_indices,
            'neg_indices': neg_indices
        }
        results.append(result_with_indices)
    return jsonify(results)



@app.route('/save_memory', methods=['POST'])
def save_memory():
    save_path = request.json.get('save_path')
    processor.save_memory(save_path)
    return jsonify({"status": "success"})


@app.route('/load_memory', methods=['POST'])
def load_memory():
    load_path = request.json.get('load_path')
    processor.load_memory_from_file(load_path)
    return jsonify({"status": "success"})
    
@app.route('/batch_memory_save', methods=['POST'])
def batch_memory_save():
    data = request.get_json()
    for item in data:
        data_id = item.get("data_id", "")
        plan = item.get("plan", "")
        question = item.get("question", "").strip()
        image_caption = item.get("image_caption", "").strip()
        used_memory_indices = item.get("used_memory_indices", [])
        trace = f"### Question: {question}\n"
        messages = item["messages"]
        j = 1
        for message in messages:
            if message["role"] == "assistant":
                trace += f"### Round {j}:\n"
                trace += f"#### Agent Reasoning and Tool Call:\n{message['content']}\n"
                j += 1
            else:
                trace += f"#### Tool Call Return Results:\n{message['content']}\n"
        judgement = "correct" if item["judgement"] == "correct" else "incorrect"
        data_batch_memory_manager = {
            "data_id": data_id,
            "question": question,
            "image_caption": image_caption,
            "trace": trace,
            "judgement": judgement,
            "plan": plan,
            "used_memory_indices": used_memory_indices,
        }
        processor.direct_store_memory(**data_batch_memory_manager)
    return jsonify({
        "status": "success"
    })

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    for modality in MODALITIES:
        for cat in CATEGORIES:
            processor.memory_store[modality][cat].clear()
    logger.info("All in-memory memories have been cleared.")
    return jsonify({"status": "success", "message": "Memory cleared successfully."})



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Memory-powered Question Answering Server')
    parser.add_argument('--port',
                      type=int,
                      default=5000,
                      help='Port to run the server on (default: 5000)')
    parser.add_argument('--host',
                      default='0.0.0.0',
                      help='Host to bind the server to (default: 0.0.0.0)')
    parser.add_argument('--model_name',
                      help='Model name to use for API calls',
                      default=None)

    args = parser.parse_args()
    if args.model_name:
        MODEL_NAME = args.model_name
        
    
    # Initialize the processor with the server URL
    processor = MemoryProcessor()
    
    app.run(host=args.host, port=args.port)
    



