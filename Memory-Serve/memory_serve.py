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


classify_question_prompt = """
You are a text classifier. Given an input question, your job is to determine exactly one category that best describes what the question is asking about. 
Allowed Categories (choose one and only one): Location, Human, Time, Career, Process, Definition, Event, Function, Property.
Instructions: 
1. Read the question.
2. Decide which single category from the list above best matches the intent of the question.
3. Output only the category name, with no explanation, no additional text, and no formatting.

Examples:
Input: "Danny Jones and Aleksi Sihvonen are both what?" Output: "Human"
Input: "Abraham Lokin was born on an archipelago between the Norwegian Sea and North Atlantic, that is about 200 miles north-northwest of where?" Output: "Location"
Input: "According to Google Finance, when was the first year the Apple stock went above $50 (without adjusting for stock split)?" Output: "Time"

Now classify the following question:
{question}
"""

get_trace_prompt = """
You will be given a transcript of a multi-round agent interaction (including reasoning traces, tool calls, and tool results). Your goal is to abstractly summarize the sequence of actions the model performed across rounds.

Your Output Requirements:  
Summarize what the model was doing in each major step, focusing on **abstract action patterns** while **including the key query content** (e.g., search keywords or tool parameters) that drove each step. Format each step as:  
**action purpose** (input → output), where *input* includes representative queries or hypotheses (e.g., “search('Palace of Lost City location')”), and *output* describes the resulting insight or state change.

Output in numbered steps (1., 2., 3., …). Do not copy verbatim; provide high-level abstractions that preserve intent and critical query details.  

Example Input: 
### Question: Where does the town belong to? 
### Round 1: <think>... use web_image_to_image_search ...</think> ...
### Round 2: <think>... appears to be Palace of the Lost City in Sun City... use search tool...</think> ...
### Round 3: <think>... use search tool with query "Sun City South Africa Palace of Lost City"</think> ...

Example Output: 
1. Use visual search to generate candidate locations (image → possible locations: e.g., “Palace of the Lost City”).
2. Use text search to narrow hypothesis (possible locations → likely country: South Africa).
3. Use text search to verify with specific query (“Sun City South Africa Palace of Lost City”) → confirmed factual answer.

Output only the numbered step summary—no explanations, headings, or extra text. Keep it clear, concise, and ≤300 words.

Now summarize the following multi-round interaction:
{trace}
"""

SYSTEM_PROMPT = """
You are a planning assistant in a three-step loop:  

1. **Plan**: Given a goal and background info, output a clear action plan.  
2. **Evaluate**: Given an execution trace, decide if replanning is needed.  
3. **Replan (if needed)**: With new reference memories, provide a revised plan targeting unmet goals.  

Keep responses concise and action-focused.
"""

planning_system_prompt = """
You are a memory-based planning assistant assisting an agent by providing strategic guidance. 

The agent has access to the following tools:
- `search`: perform text-based web queries to retrieve external information.

### [Relevant Memories] 
Each memory indicates whether it was **correct** or **incorrect**, along with its workflow:
{memory}

### Your Task:
- Analyze the **Question** and compare it with both **correct strategies** and **incorrect patterns** in the memories.
- If similar successful approaches exist, adapt their core idea to guide the next steps.
- If the context resembles past failures, highlight what to avoid and suggest corrective actions.
- Recommend **one clear, generalizable work plan or action strategy** that directly addresses the current situation and guides agents step-by-step on what to do.

### Output should:
1. Be clear and concise—no more than 400 words, and present your response as a step-by-step plan.
2. Each step in the plan must be atomic and actionable, specifying a single operation such as invoking a tool (e.g., `search` with precise query intent), performing logical inference, executing a calculation, cross-verifying facts, or synthesizing prior observations.
3. You must include relevant memories in the thinking process (<think>...</think>), but do not mention them in the final output.
4. Don't try to give the answer directly, but give a plan.
5. Prohibit the generation of content unrelated to the plan.
6. Your response must not contain any factual information.

Your output should only contain the guideline, with no additional explanations.

**[Question]** (Global Objective): 
{question}
"""

planning_system_prompt_img = """
You are a memory-based planning assistant assisting an agent by providing strategic guidance. 

The agent has access to the following tools:
- `web_image_to_image_search`: find visually similar images online (Can only be used once).
- `search`: perform text-based web queries to retrieve external information.

### [Relevant Memories] 
Each memory indicates whether it was **correct** or **incorrect**, along with its workflow:
{memory}

### Your Task:
- Analyze the **Question** and compare it with both **correct strategies** and **incorrect patterns** in the memories.
- If similar successful approaches exist, adapt their core idea to guide the next steps.
- If the context resembles past failures, highlight what to avoid and suggest corrective actions.
- Recommend **one clear, generalizable work plan or action strategy** that directly addresses the current situation and guides agents step-by-step on what to do.

### Output should:
1. Be clear and concise—no more than 300 words, and present your response as a step-by-step plan.
2. Each step in the plan must be atomic and actionable, specifying a single operation such as invoking a tool (e.g., `search` with precise query intent), performing logical inference, executing a calculation, cross-verifying facts, or synthesizing prior observations.
3. You must include relevant memories in the thinking process (<think>...</think>), but do not mention them in the final output.
4. Don't try to give the answer directly, but give a plan.
5. Prohibit the generation of content unrelated to the plan.
6. Your response must not contain any factual information.

Your output should only contain the guideline, with no additional explanations.

**IMPORTANT: The `web_image_to_image_search` tool can only be called once.** Otherwise, the agent will be severely penalized.

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


best_plan_selection_prompt = """
You are evaluating multiple plans for a given question to select the best one. Consider both success probability and conciseness.

### [Relevant Memories] 
Each memory indicates whether it was **correct** or **incorrect**, along with its workflow:
{memory}

### Question:
{question}

### Available Plans:
{plans}

### Evaluation Criteria:
1. Success probability: How likely the plan is to lead to a correct answer based on:
   - Alignment with proven successful strategies from memories
   - Avoidance of known failure patterns from memories
   - Logical coherence and completeness
2. Conciseness: How efficiently the plan achieves the goal without unnecessary steps

### Your Task:
Analyze each plan against the criteria above, considering the relevant memories. Select the plan that best balances high success probability with conciseness.

Output only the number of the selected plan (e.g., "1", "2", "3", or "4").
"""


CATEGORIES = ["location", "human", "time", "career", "process", "definition", "event", "function", "property", "others"]
MODALITIES = ["text-only", "text-image"]
MEMORY_URL = os.getenv("MEMORY_URL")
api_key = "EMPTY"
memory_client = OpenAI(base_url=MEMORY_URL, api_key=api_key)
PLAN_URL = os.getenv("PLAN_URL")
plan_client = OpenAI(base_url=PLAN_URL, api_key=api_key)


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


def filter_result(result: str) -> str:
    marker = "</think>"
    if marker in result:
        return result.split(marker, 1)[1].strip()
    else:
        return result.strip()



def llm_classify_question(question: str) -> str:
    
    # return "others"
    
    classify = {
        "type": "json_schema",
        "json_schema": {
            "name": "extracted_answer",
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "enum": ["location", "human", "time", "career", "process", "definition", "event", "function", "property"]},
                },
                "required": ["answer"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    prompt = classify_question_prompt.format(question=question)
    messages = [
        {"role": "user", "content": prompt}
    ]
    try:
        response_obj = memory_client.chat.completions.create(
            model="qwen",
            temperature=0,
            messages=messages,
            max_tokens=1024,
            timeout=100.0,
            response_format=classify,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response_obj.choices[0].message.content.strip()
        try:
            raw_content = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError(f"无法从 LLM 输出中提取 JSON: {repr(content)}")
            json_str = match.group(0)
            try:
                raw_content = json.loads(json_str)
            except json.JSONDecodeError:
                fixed_json = re.sub(r"'(\w+)'\s*:", r'"\1":', json_str)
                fixed_json = re.sub(r":\s*'([^']*)'", r': "\1"', fixed_json) 
                raw_content = json.loads(fixed_json)
        answer = raw_content["answer"].strip().lower()
        if answer not in CATEGORIES:
            answer = "others"
        return answer
    except Exception as e:
        print(f"[WARN] LLM 分类失败，使用默认类别 'others'。错误: {e}")
        if 'content' in locals():
            print(f"  原始 LLM 输出: {repr(content)}")
        return "others"

def llm_get_trace(trace: str) -> str:
    
    # return trace
    
    prompt = get_trace_prompt.format(trace=trace)
    messages = [
        {"role": "user", "content": prompt}
    ]
    response_obj = memory_client.chat.completions.create(
        model="qwen",
        temperature=0,
        messages=messages,
        max_tokens=4096,
        timeout=100.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = response_obj.choices[0].message.content.strip()
    return content


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
        # entry['usage_count'] = 0
        # entry['success_count'] = 0
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
        
        self.question_weight = 0.8
        self.caption_weight = 0.2
        
        # self.cosine_weight = 1.0
        # self.win_rate_weight = 0.0
        self.cosine_weight = 0.7
        self.win_rate_weight = 0.3
        
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
        workflow_summary = llm_get_trace(trace)
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
        pos_top_k: int = 2,
        neg_top_k: int = 2,
        pos_pass_num: int = 0,
        neg_pass_num: int = 0
    ) -> Tuple[List[Tuple[int, Dict]], List[Tuple[int, Dict]]]:  # 返回索引和数据的元组列表
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
        correct_selected = [(c[1], c[2]) for c in correct_candidates[start_pos_idx:end_pos_idx]]  # 返回索引和数据
        start_neg_idx = neg_pass_num
        end_neg_idx = start_neg_idx + neg_top_k
        incorrect_selected = [(c[1], c[2]) for c in incorrect_candidates[start_neg_idx:end_neg_idx]]  # 返回索引和数据
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
                    if entry['image_caption']:
                        memories_context += (
                            f"\n--- Example {i} (Image-Text modality) ---\n"
                            f"- Judgement: {entry['judgement']}\n"
                            f"- Question: {entry['question']}\n"
                            f"- Image Caption: {entry['image_caption']}\n"
                            f"- Workflow Summary: {entry.get('workflow_summary', 'N/A')}\n"
                        )
                    else:
                        memories_context += (
                            f"\n--- Example {i} (Text-only modality) ---\n"
                            f"- Judgement: {entry['judgement']}\n"
                            f"- Question: {entry['question']}\n"
                            f"- Workflow Summary: {entry.get('workflow_summary', 'N/A')}\n"
                        )
            if neg_memories:
                memories_context += f"\n#### Negative Examples (Failure Lessons - {len(neg_memories)}):\n"
                for i, entry in enumerate(neg_memories, 1):
                    if entry['image_caption']:
                        memories_context += (
                            f"\n--- Example {i} (Image-Text modality) ---\n"
                            f"- Judgement: {entry['judgement']}\n"
                            f"- Question: {entry['question']}\n"
                            f"- Image Caption: {entry['image_caption']}\n"
                            f"- Workflow Summary: {entry.get('workflow_summary', 'N/A')}\n"
                        )
                    else:
                        memories_context += (
                            f"\n--- Example {i} (Text-only modality) ---\n"
                            f"- Judgement: {entry['judgement']}\n"
                            f"- Question: {entry['question']}\n"
                            f"- Workflow Summary: {entry.get('workflow_summary', 'N/A')}\n"
                        )
        else:
            memories_context = "\n### Retrieved Relevant Memories:\nNone found."
        return memories_context, pos_indices, neg_indices
    
    
    def build_plan(self, question: str, image_caption: str, pos_top_k: int = 2, neg_top_k: int = 2, pos_pass_num: int = 0, neg_pass_num: int = 0, temperature=0):
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
                temperature=temperature,
                max_tokens=4096,
                # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
                # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        output = response.choices[0].message.content
        messages.append({"role": "assistant", "content": output})
        return output, messages, pos_indices, neg_indices

    def build_best_plan(self, question: str, image_caption: str, pos_top_k: int = 2, neg_top_k: int = 2, pos_pass_num: int = 0, neg_pass_num: int = 0):
        memories_context, pos_indices, neg_indices = self.get_memories_context(
            question=question,
            image_caption=image_caption,
            pos_top_k=pos_top_k,
            neg_top_k=neg_top_k,
            pos_pass_num=pos_pass_num,
            neg_pass_num=neg_pass_num
        )
        plans = []
        messages_list = []
        all_pos_indices = []
        all_neg_indices = []
        choices = {
            "type": "json_schema",
            "json_schema": {
                "name": "extracted_answer",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "enum": ["1", "2", "3", "4"]},
                    },
                    "required": ["answer"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        for i in range(4):
            try:
                plan, messages, pos_indices, neg_indices = self.build_plan(
                    question, image_caption,
                    pos_top_k=pos_top_k,
                    neg_top_k=neg_top_k,
                    pos_pass_num=pos_pass_num,
                    neg_pass_num=neg_pass_num,
                    temperature=1.0
                )
                plans.append(filter_result(plan))
                messages_list.append(messages)
                all_pos_indices.append(pos_indices)
                all_neg_indices.append(neg_indices)
            except:
                plans.append("")
                messages_list.append([])
                all_pos_indices.append(pos_indices)
                all_neg_indices.append(neg_indices)
        
        # Select the best plan using memory_client
        plan_prompt = f"Plan 1: {plans[0]}\nPlan 2: {plans[1]}\nPlan 3: {plans[2]}\nPlan 4: {plans[3]}"
        
        selection_prompt = best_plan_selection_prompt.format(
            memory=memories_context,
            question=question,
            plans=plan_prompt,
        )
        selection_messages = [
            {"role": "user", "content": selection_prompt}
        ]
        response = memory_client.chat.completions.create(
            model="qwen",
            temperature=0,
            messages=selection_messages,
            max_tokens=1024,
            timeout=100.0,
            response_format=choices,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        selection_result = response.choices[0].message.content.strip()
        
        try:
            raw_content = json.loads(selection_result)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", selection_result, re.DOTALL)
            if not match:
                raise ValueError(f"无法从 LLM 输出中提取 JSON: {repr(selection_result)}")
            json_str = match.group(0)
            try:
                raw_content = json.loads(json_str)
            except json.JSONDecodeError:
                fixed_json = re.sub(r"'(\w+)'\s*:", r'"\1":', json_str)
                fixed_json = re.sub(r":\s*'([^']*)'", r': "\1"', fixed_json) 
                raw_content = json.loads(fixed_json)
        plan_number = raw_content["answer"].strip()
        print(plan_number)
        print(plans)
        try:
            plan_number = int(plan_number)
            if plan_number < 1 or plan_number > 4:
                plan_number = 1  # Default to first plan if invalid
        except:
            plan_number = 1
        selected_plan = plans[plan_number - 1]
        selected_messages = messages_list[plan_number - 1]
        selected_pos_indices = all_pos_indices[plan_number - 1]
        selected_neg_indices = all_neg_indices[plan_number - 1]
        return selected_plan, selected_messages, selected_pos_indices, selected_neg_indices, plan_number, plans


    def if_replan(self, memory_indices, prompt: str, past_messages):
        
        user_prompt = judge_prompt.format(
            trace=prompt,
        )
        messages = past_messages
        messages.append({"role": "user", "content": user_prompt})
        
        print(len(memory_indices))
        
        if len(memory_indices) < 4:
            response = self.start_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=4096,
                # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=4096,
                # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        output = response.choices[0].message.content
        messages.append({"role": "assistant", "content": output})
        return output, messages

    def build_replan(self, question: str, past_messages, image_caption: str, pos_top_k: int = 2, neg_top_k: int = 2, pos_pass_num: int = 0, neg_pass_num: int = 0):
        memories_context, pos_indices, neg_indices = self.get_memories_context(
            question=question,
            image_caption=image_caption,
            # pos_top_k=pos_top_k,
            # neg_top_k=neg_top_k,
            # pos_pass_num=pos_pass_num,
            # neg_pass_num=neg_pass_num
            pos_top_k=pos_top_k + pos_pass_num,
            neg_top_k=neg_top_k + neg_pass_num,
            pos_pass_num=0,
            neg_pass_num=0
        )
        print("replan", len(pos_indices), len(neg_indices))
        user_prompt = replanning_system_prompt.format(
            question=question,
            # memory=memories_context,
        )
        messages = past_messages
        messages.append({"role": "user", "content": user_prompt})
        if len(pos_indices) < pos_pass_num or len(neg_indices) < neg_pass_num:
            response = self.start_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=4096,
                # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=4096,
                # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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
            
        # memories_context, pos_indices, neg_indices = processor.get_memories_context(
        #     question, image_caption, 
        #     pos_top_k=2, 
        #     neg_top_k=2, 
        #     pos_pass_num=0, 
        #     neg_pass_num=0
        # )
        # memories_context_2, pos_indices, neg_indices = processor.get_memories_context(
        #     question, image_caption, 
        #     pos_top_k=1, 
        #     neg_top_k=1, 
        #     pos_pass_num=2, 
        #     neg_pass_num=2
        # )
        # return jsonify({"memories_context": memories_context, "memories_context_2": memories_context_2})
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


@app.route('/best_plan', methods=['POST'])
def best_plan():
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
        result, messages, pos_indices, neg_indices, selected_plan_number, all_plans = processor.build_best_plan(
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
            'neg_indices': neg_indices,
            'selected_plan_number': selected_plan_number,
            'all_plans': all_plans
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
        memory_indices = item.get("memory_indices", [])
        result, messages = processor.if_replan(memory_indices, workflow, past_messages)
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
            # 'pos_indices': pos_indices,
            # 'neg_indices': neg_indices
            'pos_indices': [],
            'neg_indices': []
        }
        results.append(result_with_indices)
    return jsonify(results)


@app.route('/replan_train', methods=['POST'])
def replan_train():
    item = request.get_json()
    question = item.get("question", "")
    plan = item.get("plan", "")
    trace = item.get("trace", "")
    mem_context1 = item.get("mem_context1", "")
    mem_context2 = item.get("mem_context2", "")
    user_context1 = planning_system_prompt_img.format(question=question, memory=mem_context1)
    user_context2 = judge_prompt.format(trace=trace)
    user_context3 = replanning_system_prompt.format(question=question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_context1},
        {"role": "assitant", "content": plan},
        {"role": "user", "content": user_context2},
        {"role": "assitant", "content": "<think>The answer is incorrect, supplementary planning is needed.</think>\nyes"},
        {"role": "user", "content": user_context3},
    ]
    response = plan_client.chat.completions.create(
        model="qwen",
        messages=messages,
        temperature=0,
        max_tokens=4096,
    )
    output2 = response.choices[0].message.content
    replan_content = filter_result(output2)
    print(replan_content)
    return jsonify({"replan": True, "content": replan_content})



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
                trace += f"#### Tool Call Return Results:\n{truncate_by_whitespace_words(message['content'])}\n"
        judgement = "correct" if item["judgement"] == "correct" else "incorrect"
        data_batch_memory_manager = {
            "data_id": data_id,
            "question": question,
            "image_caption": image_caption,
            "trace": trace,
            "plan": plan,
            "judgement": judgement,
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

@app.route('/hallo')
def hallo():
    print("hallo")
    return jsonify({"sussflu": "hallo"}), 200

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
    
    processor = MemoryProcessor()
    app.run(host=args.host, port=args.port)
    



