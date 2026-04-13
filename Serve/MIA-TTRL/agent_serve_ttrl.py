from flask import Flask, request, jsonify
import pandas as pd
from call_agent import run_multimodal_dialogue, continue_multimodal_dialogue
import os
import json
import argparse
import re
from openai import OpenAI, AzureOpenAI
from transformers import AutoTokenizer
import dotenv
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import torch.nn.functional as F
import time
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import http.client
import json
import random
import base64
from PIL import Image
from io import BytesIO

def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

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


best_plan_selection_prompt = """
You are an expert agent evaluator. You will be given a question, a set of relevant historical memories, and a list of candidate plans generated for the current question.

Your goal is to select the **single plan most likely to succeed**.

### Analysis of Memories:
The provided memories contain **paired examples** for similar questions: a "Correct Plan" (which succeeded) and an "Incorrect Plan" (which failed). 
**Your first mental step is to identify the critical difference between them.** 

### Criteria for Plan Selection:
1. **Success Probability Only**: Choose the plan that maximizes the chance of getting the correct answer.
2. **Pattern Matching**:
   - **Positive Alignment**: Does the candidate plan incorporate the specific successful strategies seen in the "Correct Plans"?
   - **Negative Avoidance**: Does the candidate plan strictly avoid the pitfalls seen in the "Incorrect Plans"?

### Relevant Memories:
{memory}

### Question:
{question}

### Candidate Plans:
{plans}

Based on the contrastive analysis of the memories, select the plan that best mimics the successful patterns and avoids the failure patterns.

Output only the number of the selected plan (e.g., "1", "2", "3", or "4").
"""


CATEGORIES = ["location", "human", "time", "career", "process", "definition", "event", "function", "property", "others"]
MODALITIES = ["text-only", "text-image"]
MEMORY_URL = os.getenv("MEMORY_URL")
api_key = "EMPTY"
memory_client = OpenAI(base_url=MEMORY_URL, api_key=api_key)

def truncate_by_whitespace_words(text: str, max_words: int = 512) -> str:
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
    return "others"


def llm_get_trace(trace: str) -> str:
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
        self.question_weight = 0.8
        self.caption_weight = 0.2
        self.cosine_weight = 0.7
        self.win_rate_weight = 0.3
        self.memory_store: Dict[str, Dict[str, MemoryBucket]] = {
            modality: {cat: MemoryBucket(self.device) for cat in CATEGORIES}
            for modality in MODALITIES
        }


    def _determine_modality(self, image_caption: str) -> str:
        return "text-image" if image_caption.strip() else "text-only"

    def _classify_question_type(self, question: str) -> str:
        return llm_classify_question(question)

    def direct_store_memory(self, data_id, question, image_caption, trace, messages, judgement, plan, slow_plan, used_memory_indices):
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
            "slow_plan": slow_plan,
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
                if existing_entry["win_rate"] < 0.7:
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
    ):
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
        pos_top_k: int = 2, 
        neg_top_k: int = 2, 
        pos_pass_num: int = 0, 
        neg_pass_num: int = 0
    ):
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


    def direct_store_plan_memory(self, data_id, question, image_caption, correct_plan, incorrect_plan):
        modality = self._determine_modality(image_caption)
        category = self._classify_question_type(question)
        bucket = self.memory_store[modality][category]
        q_vec = self._encode_text(question)
        ic_vec = self._encode_text(image_caption) if image_caption.strip() else torch.zeros_like(q_vec)
        new_entry = {
            "data_id": data_id,
            "question": question,
            "image_caption": image_caption,
            "correct_plan": correct_plan,
            "incorrect_plan": incorrect_plan
        }
        bucket.add_entry(q_vec, ic_vec, new_entry)

    def retrieve_balanced_plan_memories(
        self,
        query_question: str,
        query_image_caption: str,
        top_k: int = 2,
    ):
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
        total_sim_normalized = (total_sim - total_sim.min()) / (total_sim.max() - total_sim.min() + 1e-8)
        total_sim_normalized = total_sim_normalized.cpu()
        combined_scores = total_sim_normalized
        candidates = [(combined_scores[i].item(), i, entry) for i, entry in enumerate(bucket.memory_data)]
        selected = [(c[1], c[2]) for c in candidates[:top_k]]
        return selected
        
    def get_memories_plan_context(
            self, 
            question: str, 
            image_caption: str, 
            top_k: int = 2, 
    ):
        try:
            memories_with_idx = self.retrieve_balanced_plan_memories(
                query_question=question,
                query_image_caption=image_caption,
                top_k=top_k,
            )
            indices = [idx for idx, data in memories_with_idx]
            memories = [data for idx, data in memories_with_idx]
        except:
            indices = []
            memories = []
            
        memories_context = ""
        if memories:
            memories_context += "\n### Retrieved Relevant Memories:\n"
            for i, entry in enumerate(memories, 1):
                if entry['image_caption']:
                    memories_context += (
                        f"\n--- Example {i} (Image-Text modality) ---\n"
                        f"- Question: {entry['question']}\n"
                        f"- Correct Plan: {entry.get('correct_plan', 'N/A')}\n"
                        f"- Incorrect Plan: {entry.get('incorrect_plan', 'N/A')}\n"
                    )
                else:
                    memories_context += (
                        f"\n--- Example {i} (Text-only modality) ---\n"
                        f"- Question: {entry['question']}\n"
                        f"- Correct Plan: {entry.get('correct_plan', 'N/A')}\n"
                        f"- Incorrect Plan: {entry.get('incorrect_plan', 'N/A')}\n"
                    )
        else:
            memories_context = "\n### Retrieved Relevant Memories:\nNone found."
        return memories_context
    

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




def truncate_by_whitespace_words(text: str, max_words: int = 512) -> str:
    if not text.strip():
        return text
    words = text.strip().split()
    if len(words) > max_words:
        truncated_words = words[:max_words]
        truncated_words.append("... (Omitted part of the results returned by the tool)")
    else:
        truncated_words = words[:max_words]
    return " ".join(truncated_words)

REFLECTION_JUDGE_PROMPT = """
{trace}

Based on the sufficiency of the reasons and evidence, assess whether the agent needs to replan. Only in the following cases:
- Tool support is missing or unclear,
- Reasoning has gaps or assumptions,
- Answer doesn’t fully address the question,
- Any doubt exists on correctness or clarity.

Use `replan` over accepting borderline solutions. The internal thinking block (<think>..</think>) should be extremely concise.

Respond with exactly one word: "yes" to trigger replanning, "no" to accept the current solution.
"""

memory_bank = {}

DATA_ID_TO_IMAGES = {}

def load_parquet_images(parquet_path: str):
    global DATA_ID_TO_IMAGES
    global memory_bank
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    for _, row in df.iterrows():
        data_id = str(row['data_id'])
        try:
            image_data = [row['images'][0], ]
        except:
            image_data = []
        DATA_ID_TO_IMAGES[data_id] = image_data
        memory_bank[data_id] = []
    print(f"Loaded {len(DATA_ID_TO_IMAGES)} entries from {parquet_path}")
    



@app.route('/plan', methods=['POST'])
def plan_route():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    data_id = data.get("data_id")
    question = data.get("question")
    plan = data.get("plan", "")
    if not data_id:
        return jsonify({"error": "Missing 'data_id'"}), 400
    images = DATA_ID_TO_IMAGES.get(data_id, [])
    item = {
        "data_id": data_id,
        "question": question,
        "images": images
    }
    useful_messages = run_multimodal_dialogue(plan, item)
    trace = f"### Question: {question}\n"
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
                    text_parts.append(item.get("text", ""))
            text_content = "\n".join(text_parts)
        else:
            text_content = str(content)
        if role == "assistant":
            trace += f"### Round {j}:\n"
            trace += f"#### Agent Reasoning and Tool Call:\n{text_content}\n"
            j += 1
        else:
            text_content = truncate_by_whitespace_words(text_content)
            if "I have a revised plan for you to follow:" in text_content:
                trace += f"#### Revise Plan:\n{text_content}\n"
            else:
                trace += f"#### Tool Call Return Results:\n{text_content}\n"
    prompt = REFLECTION_JUDGE_PROMPT.format(trace = trace)
    return jsonify({
        "prompt": prompt,
        "trace": trace,
        "history_messages": useful_messages
    })




@app.route('/replan', methods=['POST'])
def replan_route():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    data_id = data.get("data_id")
    replan = data.get("replan", "")
    question = data.get("question")
    messages_history = data.get("messages", [])
    original_plan = data.get("plan", "")
    if not data_id:
        return jsonify({"error": "Missing 'data_id'"}), 400
    images = DATA_ID_TO_IMAGES.get(data_id, [])
    item = {
        "data_id": data_id,
        "question": question,
        "images": images,
        "plan": original_plan,
        "messages": messages_history
    }
    temp_messages, extended_messages = continue_multimodal_dialogue(replan, item)
    trace = f"### Question: {question}\n"
    j = 1
    for message in temp_messages:
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
    prompt = REFLECTION_JUDGE_PROMPT.format(trace = trace)
    
    return jsonify({
        "prompt": prompt,
        "trace": trace,
        "history_messages": extended_messages
    })
    

@app.route('/memory', methods=['POST'])
def memory_route():
    data = request.get_json()
    data_id = data.get("data_id")
    question = data.get("question")
    image_caption = data.get("image_caption", "")
    pos_top_k = 2
    neg_top_k = 2
    pos_pass_num = 0
    neg_pass_num = 0
    context, pos_indices, neg_indices = processor.get_memories_context(
        question, image_caption, 
        pos_top_k=pos_top_k, 
        neg_top_k=neg_top_k, 
        pos_pass_num=pos_pass_num, 
        neg_pass_num=neg_pass_num
    )
    indices = list(set(pos_indices + neg_indices))
    result = {
        'prompt': context.strip(),
        'indices': indices
    }
    return jsonify(result)
    
    
@app.route('/save_memory', methods=['POST'])
def save_memory():
    save_prefix = os.environ.get('TTRL_SAVE', '.').rstrip(os.sep)
    save_path1 = os.path.join(save_prefix, 'workflow_memory.jsonl')
    processor.save_memory(save_path1)
    save_path2 = os.path.join(save_prefix, 'plan_memory.jsonl')
    plan_processor.save_memory(save_path2)
    return jsonify({"status": "success"})
    
    
@app.route('/memory_bank_save', methods=['POST'])
def memory_bank_save():
    data = request.get_json()
    data_id = data.get("data_id", "")
    slow_plan = data.get("slow_plan", "").strip()
    plan = data.get("plan", "").strip()
    question = data.get("question", "").strip()
    image_caption = data.get("image_caption", "").strip()
    
    used_memory_indices = data.get("used_memory_indices", [])
    temp_messages = data.get("temp_messages", [])
    save_messages = []
    trace = f"### Question: {question}\n"
    j = 1
    for message in temp_messages:
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
            save_messages.append({"role": "assistant", "content": text_content})
            trace += f"### Round {j}:\n"
            trace += f"#### Agent Reasoning and Tool Call:\n{text_content}\n"
            j += 1
        else:
            save_messages.append({"role": "user", "content": text_content})
            text_content = truncate_by_whitespace_words(text_content)
            trace += f"#### Tool Call Return Results:\n{text_content}\n"
    judgement = "correct" if data["judgement"] == "correct" else "incorrect"
    data_batch_memory_manager = {
        "data_id": data_id,
        "plan": plan,
        "slow_plan": slow_plan,
        "question": question,
        "image_caption": image_caption,
        "trace": trace,
        "messages": save_messages,
        "judgement": judgement,
        "used_memory_indices": used_memory_indices,
    }
    memory_bank[data_id].append(data_batch_memory_manager)
    return jsonify({
        "status": "success",
    })



@app.route('/consolidate_memories', methods=['POST'])
def consolidate_memories():
    saved_pos_count = 0
    saved_neg_count = 0
    data_ids_to_process = [k for k, v in memory_bank.items() if v]
    print(f"consolidate {len(data_ids_to_process)} memories")
    for data_id in data_ids_to_process:
        entries = memory_bank[data_id]
        if not entries:
            continue
        correct_entries = [e for e in entries if e.get('judgement') == 'correct']
        incorrect_entries = [e for e in entries if e.get('judgement') == 'incorrect']
        if correct_entries:
            best_positive = min(correct_entries, key=lambda x: len(x.get('messages', [])))
            saved_pos_count += 1
            if len(correct_entries) <= len(incorrect_entries):
                best_positive["used_memory_indices"] = []
            processor.direct_store_memory(**best_positive)
        if incorrect_entries:
            random_choice = random.randrange(0, len(incorrect_entries))
            random_negative = incorrect_entries[random_choice]
            saved_neg_count += 1
            if len(correct_entries) > len(incorrect_entries):
                random_negative["used_memory_indices"] = []
            processor.direct_store_memory(**random_negative)
            
        if correct_entries and incorrect_entries:
            plan_item = {
                "data_id": data_id,
                "question": best_positive["question"],
                "image_caption": best_positive["image_caption"],
                "correct_plan": best_positive["plan"],
                "incorrect_plan": random_negative["plan"]
            }
            plan_processor.direct_store_plan_memory(**plan_item)
        memory_bank[data_id] = []
            
    
    return jsonify({
        "status": "success",
        "saved_positives": saved_pos_count,
        "saved_negatives": saved_neg_count,
    })

def calcuate_shortest_longest(valid_plans):
    item_pattern = re.compile(r'^\s*\d+\.\s')
    def get_plan_metrics(plan_str):
        item_count = sum(1 for line in plan_str.splitlines() 
                        if item_pattern.match(line))
        word_count = len(plan_str.split())
        return (item_count, word_count)
    min_idx = max_idx = 0
    min_key = max_key = get_plan_metrics(valid_plans[0])
    for idx in range(1, len(valid_plans)):
        curr_key = get_plan_metrics(valid_plans[idx])
        if curr_key < min_key:
            min_key, min_idx = curr_key, idx
        if curr_key > max_key:
            max_key, max_idx = curr_key, idx
    return min_idx, max_idx

def find_plan_index(valid_plans, slow_plan):
    if not slow_plan:
        return -1
    target = slow_plan.strip()
    for idx, plan in enumerate(valid_plans):
        if plan.strip() == target:
            return idx
    return -1
    
    
    
    
@app.route('/batch_evaluate', methods=['POST'])
def batch_evaluate():
    save_prefix = os.environ.get('TTRL_SAVE', '.').rstrip(os.sep)
    save_file1 = os.path.join(save_prefix, 'evaluated-random.jsonl')
    save_file2 = os.path.join(save_prefix, 'evaluated-choice.jsonl')
    save_file3 = os.path.join(save_prefix, 'evaluated-short.jsonl')
    save_file4 = os.path.join(save_prefix, 'evaluated-long.jsonl')
    save_file5 = os.path.join(save_prefix, 'evaluated-slow.jsonl')
    save_file6 = os.path.join(save_prefix, 'evaluated-first.jsonl')
    save_file7 = os.path.join(save_prefix, 'evaluated-choice-few-reflect.jsonl')
    os.makedirs(os.path.dirname(save_file1), exist_ok=True)
    os.makedirs(os.path.dirname(save_file2), exist_ok=True)
    os.makedirs(os.path.dirname(save_file3), exist_ok=True)
    os.makedirs(os.path.dirname(save_file4), exist_ok=True)
    os.makedirs(os.path.dirname(save_file5), exist_ok=True)
    os.makedirs(os.path.dirname(save_file6), exist_ok=True)
    os.makedirs(os.path.dirname(save_file7), exist_ok=True)
    processed_count = 0
    data_ids_to_process = [k for k, v in memory_bank.items() if v]

    with open(save_file1, 'a', encoding='utf-8') as f1, open(save_file2, 'a', encoding='utf-8') as f2, \
        open(save_file3, 'a', encoding='utf-8') as f3, open(save_file4, 'a', encoding='utf-8') as f4, \
            open(save_file5, 'a', encoding='utf-8') as f5, open(save_file6, 'a', encoding='utf-8') as f6, \
                open(save_file7, 'a', encoding='utf-8') as f7:
        for data_id in tqdm(data_ids_to_process, desc="Evaluating Plans"):
            entries = memory_bank[data_id]
            if not entries:
                continue
            # 1. 准备基础信息
            # 假设同一 data_id 下 question 和 image_caption 是一致的，取第一个非空的
            first_entry = entries[0]
            question = first_entry.get("question", "")
            image_caption = first_entry.get("image_caption", "")
            slow_plan = first_entry.get("slow_plan", "").strip()
            # 2. 聚合 Plans
            no_reflect_plans_text = ""
            reflect_plans_text = ""
            plans_text = ""
            valid_plans = []     # 存储 plan 文本内容
            valid_entries = []   # 存储对应的原始 entry 对象，以便后续保存
            reflect_valid_entries = []
            no_reflect_valid_entries = []
            for idx, entry in enumerate(entries):
                plan_content = entry.get("plan", "").strip()
                if plan_content:
                    valid_plans.append(plan_content)
                    valid_entries.append(entry)
                    plans_text += f"Plan {len(valid_plans)}:\n{plan_content}\n\n"
                    if "I have a revised plan for you to follow:" in entry["trace"]:
                        reflect_valid_entries.append(entry)
                        reflect_plans_text += f"Plan {len(reflect_valid_entries)}:\n{plan_content}\n\n"
                    else:
                        no_reflect_valid_entries.append(entry)
                        no_reflect_plans_text += f"Plan {len(no_reflect_valid_entries)}:\n{plan_content}\n\n"
                    
            if not valid_plans:
                continue
            else:
                min_idx, max_idx = calcuate_shortest_longest(valid_plans)
                slow_idx = find_plan_index(valid_plans, slow_plan)
            # 如果只有一个计划，直接选它，跳过 LLM 评估以节省成本
            if len(valid_plans) == 1:
                best_entry = valid_entries[0]
                f1.write(json.dumps(best_entry, ensure_ascii=False) + '\n')
                f2.write(json.dumps(best_entry, ensure_ascii=False) + '\n')
                f3.write(json.dumps(best_entry, ensure_ascii=False) + '\n')
                f4.write(json.dumps(best_entry, ensure_ascii=False) + '\n')
                f5.write(json.dumps(best_entry, ensure_ascii=False) + '\n')
                f6.write(json.dumps(best_entry, ensure_ascii=False) + '\n')
                f7.write(json.dumps(best_entry, ensure_ascii=False) + '\n')
                processed_count += 1
                continue
            # 3. 准备 Memory 上下文 (使用 plan_processor)
            memory_context = plan_processor.get_memories_plan_context(question, image_caption)
            # 4. 构建 Prompt
            prompt = best_plan_selection_prompt.format(
                question=question,
                memory=memory_context,
                plans=plans_text
            )
            # 5. 动态构建 JSON Schema
            # 根据 valid_plans 的长度生成 enum，例如 ["1", "2", "3"]
            plan_indices = [str(i) for i in range(1, len(valid_plans) + 1)]
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "select_best_plan",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string", 
                                "enum": plan_indices,
                                "description": "The number of the selected best plan (e.g., '1')"
                            },
                        },
                        "required": ["answer"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
            # 6. 调用 LLM
            try:
                messages = [{"role": "user", "content": prompt}]
                response = memory_client.chat.completions.create(
                    model="qwen", # 确保模型名称与全局变量或参数一致
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1024,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    response_format=response_format
                )
                content = response.choices[0].message.content
                # print(messages, "\n", content)
                try:
                    result_json = json.loads(content)
                    selected_index_str = result_json.get("answer")
                    selected_index = int(selected_index_str) - 1
                except:
                    selected_index = 0
                if 0 <= selected_index < len(valid_entries):
                    best_entry = valid_entries[selected_index]
                    random_choice = random.randrange(0, len(plan_indices))
                    f1.write(json.dumps(valid_entries[random_choice], ensure_ascii=False) + '\n')
                    f2.write(json.dumps(best_entry, ensure_ascii=False) + '\n')
                    f3.write(json.dumps(valid_entries[min_idx], ensure_ascii=False) + '\n')
                    f4.write(json.dumps(valid_entries[max_idx], ensure_ascii=False) + '\n')
                    f5.write(json.dumps(valid_entries[slow_idx], ensure_ascii=False) + '\n')
                    f6.write(json.dumps(valid_entries[0], ensure_ascii=False) + '\n')
                    processed_count += 1
                    no_reflect_best_entry = best_entry
                else:
                    entry = valid_entries[0]
                    f1.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    f2.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    f3.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    f4.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    f5.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    f6.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    processed_count += 1
                    no_reflect_best_entry = entry
                    logger.error(f"LLM returned out-of-bound index {selected_index} for data_id {data_id}")
            except Exception as e:
                entry = valid_entries[0]
                f1.write(json.dumps(entry, ensure_ascii=False) + '\n')
                f2.write(json.dumps(entry, ensure_ascii=False) + '\n')
                f3.write(json.dumps(entry, ensure_ascii=False) + '\n')
                f4.write(json.dumps(entry, ensure_ascii=False) + '\n')
                f5.write(json.dumps(entry, ensure_ascii=False) + '\n')
                f6.write(json.dumps(entry, ensure_ascii=False) + '\n')
                processed_count += 1
                no_reflect_best_entry = entry
                logger.error(f"Error processing data_id {data_id}: {e}")
    
            if len(no_reflect_valid_entries) > 0:
                no_reflect_prompt = best_plan_selection_prompt.format(
                    question=question,
                    memory=memory_context,
                    plans=no_reflect_plans_text
                )
                no_reflect_plan_indices = [str(i) for i in range(1, len(no_reflect_valid_entries) + 1)]
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "select_best_plan",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string", 
                                    "enum": no_reflect_plan_indices,
                                    "description": "The number of the selected best plan (e.g., '1')"
                                },
                            },
                            "required": ["answer"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
                try:
                    no_reflect_messages = [{"role": "user", "content": no_reflect_prompt}]
                    no_reflect_response = memory_client.chat.completions.create(
                        model="qwen", # 确保模型名称与全局变量或参数一致
                        messages=no_reflect_messages,
                        temperature=0.0,
                        max_tokens=1024,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                        response_format=response_format
                    )
                    no_reflect_content = no_reflect_response.choices[0].message.content
                    try:
                        result_json = json.loads(no_reflect_content)
                        selected_index_str = result_json.get("answer")
                        selected_index = int(selected_index_str) - 1
                    except:
                        selected_index = 0
                    if 0 <= selected_index < len(no_reflect_valid_entries):
                        no_reflect_best_entry = no_reflect_valid_entries[selected_index]
                        random_choice = random.randrange(0, len(plan_indices))
                        f7.write(json.dumps(no_reflect_best_entry, ensure_ascii=False) + '\n')
                        processed_count += 1
                    else:
                        entry = no_reflect_valid_entries[0]
                        f7.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        processed_count += 1  
                except Exception as e:
                    entry = no_reflect_valid_entries[0]
                    f7.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
            else:
                f7.write(json.dumps(no_reflect_best_entry, ensure_ascii=False) + '\n')


    return jsonify({
        "status": "success",
        "processed_count": processed_count,
        "message": f"Evaluated and saved {processed_count} best plans."
    })


if __name__ == '__main__':
    parquet_path = os.getenv("PARQUET_PATH")

    load_parquet_images(parquet_path)
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
    plan_processor = MemoryProcessor()
    app.run(host=args.host, port=args.port)
