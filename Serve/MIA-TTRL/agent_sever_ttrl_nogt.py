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
from judge import judge_answer



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

Output only the numbered step summary—no explanations, headings, or extra text. Keep it clear, concise.

Now summarize the following multi-round interaction:
{trace}
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

    def direct_store_memory(self, data_id, question, image_caption, trace, plan, used_memory_indices, judgement_nogt, judgement_data, **kwargs):
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
            "judgement_nogt": judgement_nogt,
            "judgement_data": judgement_data
        }
        # if bucket.get_size() > 0:
        #     q_query_vec = q_vec.unsqueeze(0)
        #     with torch.no_grad():
        #         sim_q = F.cosine_similarity(q_query_vec, bucket.question_embeddings)  # (N,)
        #         if image_caption.strip():
        #             ic_query_vec = ic_vec.unsqueeze(0)
        #             sim_ic = F.cosine_similarity(ic_query_vec, bucket.caption_embeddings)
        #             total_sim = self.question_weight * sim_q + self.caption_weight * sim_ic
        #         else:
        #             total_sim = sim_q
        #     max_sim, max_idx = torch.max(total_sim, dim=0)
        #     if max_sim.item() >= 0.9999:
        #         existing_entry = bucket.memory_data[max_idx.item()]
        #         if existing_entry["win_rate"] < 0.7:
        #             if existing_entry["judgement_nogt"] == "incorrect":
        #                 bucket.update_entry(max_idx.item(), q_vec, ic_vec, new_entry)
        #                 logger.info(f"Updated incorrect memory in {modality}/{category}")
        #             else:
        #                 old_len = self._count_words(existing_entry["workflow_summary"])
        #                 new_len = self._count_words(workflow_summary)
        #                 if new_len < old_len:
        #                     bucket.update_entry(max_idx.item(), q_vec, ic_vec, new_entry)
        #                     logger.info(f"Replaced correct memory with shorter version in {modality}/{category}")
        #                 else:
        #                     logger.info(f"Kept existing correct memory (shorter or equal)")
        #             return
        bucket.add_entry(q_vec, ic_vec, new_entry)
        logger.info(f"Added new memory to {modality}/{category}. Total: {bucket.get_size()}")
        print(used_memory_indices)
        if used_memory_indices:
            success = (judgement_nogt == "correct")
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
        correct_candidates = [c for c in candidates if c[2]["judgement_nogt"] == "correct"]
        incorrect_candidates = [c for c in candidates if c[2]["judgement_nogt"] == "incorrect"]
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
                            f"- Question: {entry['question']}\n"
                            # f"- Image Caption: {entry['image_caption']}\n"
                            f"- Workflow Summary: {entry.get('workflow_summary', 'N/A')}\n"
                            f"- No Ground-Truth Judgement: {entry['judgement_nogt']}\n"
                            f"- Reasoning & Logic Feedback: {entry['judgement_data']['evaluator_1_feedback']}\n"
                            f"- Information Sourcing Feedback: {entry['judgement_data']['evaluator_2_feedback']}\n"
                            f"- Result Validity Feedback: {entry['judgement_data']['evaluator_3_feedback']}\n"
                        )
                    else:
                        memories_context += (
                            f"\n--- Example {i} (Text-only modality) ---\n"
                            f"- Question: {entry['question']}\n"
                            f"- Workflow Summary: {entry.get('workflow_summary', 'N/A')}\n"
                            f"- No Ground-Truth Judgement: {entry['judgement_nogt']}\n"
                            f"- Reasoning & Logic Feedback: {entry['judgement_data']['evaluator_1_feedback']}\n"
                            f"- Information Sourcing Feedback: {entry['judgement_data']['evaluator_2_feedback']}\n"
                            f"- Result Validity Feedback: {entry['judgement_data']['evaluator_3_feedback']}\n"
                        )
            if neg_memories:
                memories_context += f"\n#### Negative Examples (Failure Lessons - {len(neg_memories)}):\n"
                for i, entry in enumerate(neg_memories, 1):
                    if entry['image_caption']:
                        memories_context += (
                            f"\n--- Example {i} (Image-Text modality) ---\n"
                            f"- Question: {entry['question']}\n"
                            # f"- Image Caption: {entry['image_caption']}\n"
                            f"- Workflow Summary: {entry.get('workflow_summary', 'N/A')}\n"
                            f"- No Ground-Truth Judgement: {entry['judgement_nogt']}\n"
                            f"- Reasoning & Logic Feedback: {entry['judgement_data']['evaluator_1_feedback']}\n"
                            f"- Information Sourcing Feedback: {entry['judgement_data']['evaluator_2_feedback']}\n"
                            f"- Result Validity Feedback: {entry['judgement_data']['evaluator_3_feedback']}\n"
                        )
                    else:
                        memories_context += (
                            f"\n--- Example {i} (Text-only modality) ---\n"
                            f"- Question: {entry['question']}\n"
                            f"- Workflow Summary: {entry.get('workflow_summary', 'N/A')}\n"
                            f"- No Ground-Truth Judgement: {entry['judgement_nogt']}\n"
                            f"- Reasoning & Logic Feedback: {entry['judgement_data']['evaluator_1_feedback']}\n"
                            f"- Information Sourcing Feedback: {entry['judgement_data']['evaluator_2_feedback']}\n"
                            f"- Result Validity Feedback: {entry['judgement_data']['evaluator_3_feedback']}\n"
                        )
        else:
            memories_context = "\n### Retrieved Relevant Memories:\nNone found."
        return memories_context, pos_indices, neg_indices


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
    return jsonify({"status": "success"})
    
    
@app.route('/memory_bank_save', methods=['POST'])
def memory_bank_save():
    data = request.get_json()
    data_id = data.get("data_id", "")
    plan = data.get("plan", "").strip()
    question = data.get("question", "").strip()
    image_caption = data.get("image_caption", "").strip()
    ground_truth = str(data.get("ground_truth", "")).strip()
    used_memory_indices = data.get("used_memory_indices", [])
    temp_messages = data.get("temp_messages", [])
    judgement_data = data["judgement_data"]
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
    judgement = data["judgement_nogt"]
    model_answer = data.get("model_answer", "").strip()
    data_batch_memory_manager = {
        "data_id": data_id,
        "plan": plan,
        "question": question,
        "image_caption": image_caption,
        "trace": trace,
        "messages": save_messages,
        "model_answer": model_answer,
        "ground_truth": ground_truth, 
        "judgement_nogt": judgement,
        "judgement_data": judgement_data,
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
        correct_entries = [e for e in entries if e.get('judgement_nogt') == 'correct']
        incorrect_entries = [e for e in entries if e.get('judgement_nogt') == 'incorrect']
        # if correct_entries:
        #     best_positive = min(correct_entries, key=lambda x: len(x.get('messages', [])))
        #     saved_pos_count += 1
        #     processor.direct_store_memory(**best_positive)
        # else:
        #     random_choice = random.randrange(0, len(incorrect_entries))
        #     random_negative = incorrect_entries[random_choice]
        #     saved_neg_count += 1
        #     processor.direct_store_memory(**random_negative)
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
        memory_bank[data_id] = []
    return jsonify({
        "status": "success",
        "saved_positives": saved_pos_count,
        "saved_negatives": saved_neg_count,
    })


@app.route('/batch_evaluate', methods=['POST'])
def batch_evaluate():
    save_prefix = os.environ.get('TTRL_SAVE', '.').rstrip(os.sep)
    save_file1 = os.path.join(save_prefix, 'evaluated.jsonl')
    os.makedirs(os.path.dirname(save_file1), exist_ok=True)
    processed_count = 0
    data_ids_to_process = [k for k, v in memory_bank.items() if v]
    with open(save_file1, 'a', encoding='utf-8') as f1:
        for data_id in tqdm(data_ids_to_process, desc="Evaluating Plans"):
            entries = memory_bank[data_id]
            if not entries:
                continue
            correct_entries = [e for e in entries if e.get("judgement_nogt") == "correct"]
            chosen_entry = random.choice(correct_entries) if correct_entries else random.choice(entries)
            question = chosen_entry.get("question", "")
            model_answer = chosen_entry.get("model_answer", "")
            correct_answer = chosen_entry.get("ground_truth", "")
            judgement = judge_answer(question=question, model_answer=model_answer, correct_answer=correct_answer) 
            chosen_entry.update({"judgement": judgement})
            f1.write(json.dumps(chosen_entry, ensure_ascii=False) + '\n')
            processed_count += 1
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
    app.run(host=args.host, port=args.port)
