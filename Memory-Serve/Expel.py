import os
import json
import argparse
import re
from flask import Flask, request, jsonify
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import dotenv
from typing import List, Dict, Any, Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

app = Flask(__name__)

MODEL_NAME = "qwen"  # 默认模型
MEMORY_URL = os.getenv("MEMORY_URL")
PLAN_URL = os.getenv("PLAN_URL")
# 假设这两个 Client 是一样的，或者根据实际情况配置
api_key = "EMPTY"
plan_client = OpenAI(base_url=PLAN_URL, api_key=api_key)
memory_client = OpenAI(base_url=MEMORY_URL, api_key=api_key)

# --- Prompts ---


RULE_OPT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rule_optimization",
        "schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string", 
                    "enum": ["AGREE", "EDIT", "ADD"],
                    "description": "The operation to perform on the rules."
                },
                "rule_number": {
                    "type": "integer", 
                    "description": "The 1-based index of the existing rule to AGREE or EDIT. Use 0 for ADD."
                },
                "rule_content": {
                    "type": "string", 
                    "description": "The content of the new rule (for ADD) or modified rule (for EDIT). Use empty string for AGREE."
                }
            },
            "required": ["operation", "rule_number", "rule_content"],
            "additionalProperties": False
        },
        "strict": True
    }
}
        

     
EXPEL_PROMPT = """
Here are the two previous trials to compare and critique:

SUCCESSFUL TRIAL:
{success_history}

FAILED TRIAL:
{fail_history}

Here are the EXISTING RULES (retrieved from similar past questions):
{existing_rules}

By examining and contrasting the successful trial against the failed trial and existing rules, please generate **ONE** optimal rule for the current question.

You must choose exactly **ONE** operation:
1. **AGREE**: If an existing rule is perfect for this case. (Provide the existing rule number).
2. **EDIT**: If an existing rule needs modification to be better. (Provide the rule number and the NEW content).
3. **ADD**: If no existing rule applies and a new one is needed. (Provide the NEW content).

Your goal is to ensure the agent avoids the failure in the future.
"""

judge_prompt = """You are a strict evaluator for an autonomous agent.

You will be given:

1. The original question the agent is trying to solve.
2. The agent’s historical trajectory, consisting of previous reasoning steps and intermediate results.
3. The agent’s current intermediate result.

Your task is to determine whether the current intermediate result is logically correct and consistent with:

- the original question, and
- the agent’s historical trajectory.

Evaluation rules:

- Focus on factual correctness, logical validity, and consistency.
- Do NOT assume missing steps are correct.
- If the intermediate result contains any logical error, contradiction, unjustified assumption, or incorrect conclusion, it should be judged as incorrect.

Output rules (STRICT):

- Respond with ONLY one word: "Yes" or "No".
- Do NOT provide explanations, reasoning, or any additional text.
- Do NOT include punctuation, markdown, or whitespace.

Question: {question}

Historical Trajectory: {traj}

Current Intermediate Result: {result}

Your response is:
"""

reflex_rule_prompt = """
You are an expert AI Rule Optimizer. Your task is to refine an existing behavioral rule based on a failed execution trace.

### Context
The agent attempted to answer the following question but failed, despite having a guiding rule.

**Question:** 
{question}

**Current Rule (which failed to prevent the error):** 
{rule}

**Failed Trajectory (Execution Trace):**
{traj}

### Analysis Instructions
1. **Diagnose the Failure:** Look closely at the "Failed Trajectory". Identify the exact step where the agent went wrong (e.g., ineffective search queries, hallucinating information, misinterpreting tool outputs, or logical errors).
2. **Critique the Current Rule:** Determine why the "Current Rule" was insufficient. Was it too vague? Did it focus on the wrong aspect? Was it ignored because it wasn't specific enough?
3. **Refine:** Rewrite the rule to specifically prevent this type of failure in the future.

### Output Requirements
- Only include the **Revised Rule** must be concise, actionable.

**Revised Rule:**
"""


CATEGORIES = ["others"]
MODALITIES = ["text-only", "text-image"]

def filter_result(result: str) -> str:
    marker = "</think>"
    if marker in result:
        return result.split(marker, 1)[1].strip()
    return result.strip()

class MemoryBucket:
    def __init__(self, device: str):
        self.device = device
        self.question_embeddings: Optional[torch.Tensor] = None   # (N, D)
        self.caption_embeddings: Optional[torch.Tensor] = None    # (N, D)
        self.memory_data: List[Dict[str, Any]] = []               # List of entries

    def add_entry(self, q_vec: torch.Tensor, ic_vec: torch.Tensor, entry: Dict[str, Any]):
        q_vec = q_vec.unsqueeze(0)
        ic_vec = ic_vec.unsqueeze(0)
        if self.question_embeddings is None:
            self.question_embeddings = q_vec
            self.caption_embeddings = ic_vec
        else:
            self.question_embeddings = torch.cat([self.question_embeddings, q_vec], dim=0)
            self.caption_embeddings = torch.cat([self.caption_embeddings, ic_vec], dim=0)
        
        # 初始化统计数据
        entry.setdefault('usage_count', 1)
        entry.setdefault('success_count', 1 if entry.get('judgement') == 'correct' else 0)
        entry.setdefault('win_rate', 1.0 if entry.get('judgement') == 'correct' else 0.0)
        entry.setdefault('rules', '') # 确保有 rules 字段
        
        self.memory_data.append(entry)

    def update_entry(self, idx: int, q_vec: torch.Tensor, ic_vec: torch.Tensor, entry: Dict[str, Any]):
        # 更新 Embedding (通常不需要变，除非问题文本变了)
        self.question_embeddings[idx] = q_vec
        self.caption_embeddings[idx] = ic_vec
        
        # 保留旧的统计数据
        old_entry = self.memory_data[idx]
        entry['usage_count'] = old_entry.get('usage_count', 1) + 1
        is_success = 1 if entry.get('judgement') == 'correct' else 0
        entry['success_count'] = old_entry.get('success_count', 0) + is_success
        entry['win_rate'] = entry['success_count'] / entry['usage_count']
        
        # 如果新 entry 没有 rules，保留旧的 rules
        if 'rule' not in entry or not entry['rules']:
            entry['rule'] = old_entry.get('rule', '')

        self.memory_data[idx] = entry

    def get_size(self) -> int:
        return len(self.memory_data)

    def clear(self):
        self.question_embeddings = None
        self.caption_embeddings = None
        self.memory_data.clear()

class MemoryProcessor:
    def __init__(self):
        self.model_name = MODEL_NAME
        # 请根据实际路径修改 BERT 路径
        bert_path = "/your_path/bert/sup-simcse-bert-base-uncased"
        # 如果本地没有，可以尝试用 huggingface 的 ID: "sentence-transformers/all-mpnet-base-v2" 或其他
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.bert_model_tokenizer = AutoTokenizer.from_pretrained(bert_path)
            self.bert_model_encoder = AutoModel.from_pretrained(bert_path).to(self.device)
        except:
            logger.warning("Local BERT path not found, using default 'bert-base-uncased' for demo.")
            self.bert_model_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model_encoder = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
            
        self.bert_model_encoder.eval()
        
        self.client = plan_client
        
        self.question_weight = 1.0
        self.caption_weight = 0.0 # 简化处理，可根据需要调整
        
        self.memory_store: Dict[str, Dict[str, MemoryBucket]] = {
            modality: {cat: MemoryBucket(self.device) for cat in CATEGORIES}
            for modality in MODALITIES
        }

    def _determine_modality(self, image_caption: str) -> str:
        return "text-image" if image_caption.strip() else "text-only"

    def _classify_question_type(self, question: str) -> str:
        return "others"

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
        

    def extract_and_update_rules(self, 
                                 question: str, 
                                 image_caption: str, 
                                 current_trace: str, 
                                 current_judgement: str,
                                 bucket: MemoryBucket) -> List[str]:
        if bucket.get_size() == 0:
            existing_rules = []
        else:
            q_vec = self._encode_text(question).unsqueeze(0)
            ic_vec = self._encode_text(image_caption).unsqueeze(0) if image_caption.strip() else None
            
            with torch.no_grad():
                sim_q = F.cosine_similarity(q_vec, bucket.question_embeddings)
                if image_caption.strip():
                    sim_ic = F.cosine_similarity(ic_vec, bucket.caption_embeddings)
                    total_sim = self.question_weight * sim_q + self.caption_weight * sim_ic
                else:
                    total_sim = sim_q

            # 1. 获取 Existing Rules (Top-3 最相似问题的规则集合)
            # 我们取最相似的那个问题的规则作为基准，或者合并 Top-3。
            # 为了符合 ExpeL 的 "Edit Existing Rules" 逻辑，最好取最相似的一个非空规则集。
            top_k_indices = torch.topk(total_sim, k=min(3, bucket.get_size())).indices.tolist()
            existing_rules = []
            for idx in top_k_indices:
                rule = bucket.memory_data[idx].get('rule', '')
                existing_rules.append(rule)

        
        
        if current_judgement == "correct":
            success_history = f"Question: {question}\nTrace: {current_trace}"
            fail_history = ""
        else:
            success_history = ""
            fail_history = f"Question: {question}\nTrace: {current_trace}"

        # 格式化 Existing Rules 给 Prompt
        existing_rules_str = ""
        if existing_rules:
            for i, r in enumerate(existing_rules, 1):
                existing_rules_str += f"{i}: {r}\n"
        else:
            existing_rules_str = "None"

        # 3. 调用 LLM
        prompt = EXPEL_PROMPT.format(
            success_history=success_history,
            fail_history=fail_history,
            existing_rules=existing_rules_str
        )
        
        # try:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=8192,
            response_format=RULE_OPT_SCHEMA,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content.strip()
        result_json = json.loads(content)
        print(result_json)
        operation = result_json.get("operation", "ADD")
        rule_num = result_json.get("rule_number", 0)
        rule_num = int(rule_num) - 1
        new_content = result_json.get("rule_content", "").strip()
        if operation == "AGREE":
            return ""
        elif operation == "EDIT":
            if new_content:
                bucket.memory_data[rule_num]["rule"] = new_content
            return ""
        elif operation == "ADD":
            return new_content
        
        # except Exception as e:
        #     logger.error(f"Error during ExpeL rule extraction: {e}")
        #     return ""

    def direct_store_memory(self, data_id, question: str, image_caption: str, trace: str, judgement: str, rule: str, used_memory_indices):
        modality = self._determine_modality(image_caption)
        category = self._classify_question_type(question)
        bucket = self.memory_store[modality][category]
        new_rule = self.extract_and_update_rules(question, image_caption, trace, judgement, bucket)
        if new_rule:
            q_vec = self._encode_text(question)
            ic_vec = self._encode_text(image_caption) if image_caption.strip() else torch.zeros_like(q_vec)
            workflow_summary = trace
            new_entry = {
                "data_id": data_id,
                "question": question,
                "image_caption": image_caption,
                "workflow_summary": workflow_summary,
                "judgement": judgement,
                "rule": new_rule
            }
            if bucket.get_size() > 0:
                q_query_vec = q_vec.unsqueeze(0)
                with torch.no_grad():
                    sim_q = F.cosine_similarity(q_query_vec, bucket.question_embeddings)
                    if image_caption.strip():
                        ic_query_vec = ic_vec.unsqueeze(0)
                        sim_ic = F.cosine_similarity(ic_query_vec, bucket.caption_embeddings)
                        total_sim = self.question_weight * sim_q + self.caption_weight * sim_ic
                    else:
                        total_sim = sim_q
                max_sim, max_idx = torch.max(total_sim, dim=0)
            bucket.add_entry(q_vec, ic_vec, new_entry)
            logger.info(f"Added new memory to {modality}/{category}. Total: {bucket.get_size()}")
            if used_memory_indices:
                success = (judgement == "correct")
                bucket.update_memory_stats(used_memory_indices, success)


    def retrieve_balanced_memories(
        self,
        query_question: str,
        query_image_caption: str,
        top_k: int = 3,
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
        candidates.sort(key=lambda x: x[0], reverse=True)
        start_pos_idx = 0
        end_pos_idx = top_k
        selected = [(c[1], c[2]) for c in candidates[start_pos_idx:end_pos_idx]]  # 返回索引和数据
        return selected

    def get_memories_context(
        self, 
        question: str, 
        image_caption: str, 
        top_k=1,
        traj=""
    ) -> Tuple[str, List[int], List[int]]:  # 返回上下文、正面记忆索引列表、负面记忆索引列表
        try:
            memories_with_idx = self.retrieve_balanced_memories(
                query_question=question,
                query_image_caption=image_caption,
                top_k=1
            )
            indices = [idx for idx, data in memories_with_idx]
            memories = [data for idx, data in memories_with_idx]
        except:
            memories = []
            indices = []
        
        memories_context = ""
        if memories:
            memories_context += memories[0].get('rule', 'N/A')
            indices = [indices[0], ]
        else:
            memories_context = ""
            indices = []
        if traj:
            messages = [{"role": "user", "content": reflex_rule_prompt.format(question=question,traj=traj,rule=memories_context)}]
            response_obj = memory_client.chat.completions.create(
                model="qwen",
                temperature=0,
                messages=messages,
                max_tokens=1024,
                timeout=100.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            memories_context = response_obj.choices[0].message.content.strip()
        return memories_context, indices
    
    
    def build_plan(self, question: str, image_caption: str, top_k: int = 1, traj = ""):
        memories_context, indices = self.get_memories_context(
            question=question,
            image_caption=image_caption,
            top_k=1,
            traj=traj
        )
        if traj:
            print("replan", memories_context)
        else:
            print("plan", memories_context)
        return memories_context, [], indices

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







@app.route('/plan', methods=['POST'])
def plan():
    data = request.get_json()
    results = []
    for item in data:
        # 简化参数获取
        question = item.get("question", "").strip()
        image_captions = item.get("image_caption", [])
        image_caption = image_captions[0].strip() if isinstance(image_captions, list) and image_captions else (image_captions if isinstance(image_captions, str) else "")
        result, messages, indices = processor.build_plan(
            question, image_caption,
            top_k=1
        )
        result = filter_result(result)
        results.append({
            'plan': result,
            'messages': [],
            'pos_indices': [],
            'neg_indices': []
        })
    return jsonify(results)

@app.route('/judge_replan', methods=['POST'])
def judge_replan():
    data = request.get_json()
    results = []
    for item in data:
        question = item.get("question", "")
        traj = item.get("traj", "")
        result = item.get("result", "")
        messages = [{"role": "user", "content": judge_prompt.format(question=question,traj=traj,result=result)}]
        response_obj = memory_client.chat.completions.create(
            model="qwen",
            temperature=0,
            messages=messages,
            max_tokens=1024,
            timeout=100.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response_obj.choices[0].message.content.strip().lower()
        
        print("judge", content)
        
        if "yes" in content:
            results.append(True)
        else:
            results.append(False)
    return jsonify(results)

@app.route('/replan', methods=['POST'])
def replan():
    data = request.get_json()
    results = []
    for item in data:
        # 简化参数获取
        question = item.get("question", "").strip()
        image_captions = item.get("image_caption", [])
        traj = item.get("traj", "")
        image_caption = image_captions[0].strip() if isinstance(image_captions, list) and image_captions else (image_captions if isinstance(image_captions, str) else "")
        result, messages, indices = processor.build_plan(
            question, image_caption,
            1, traj
        )
        result = filter_result(result)
        results.append({
            'plan': result,
            'messages': [],
            'pos_indices': [],
            'neg_indices': []
        })
    return jsonify(results)

@app.route('/batch_memory_save', methods=['POST'])
def batch_memory_save():
    data = request.get_json()
    for item in data:
        data_id = item.get("data_id", "")
        rule = item.get("rule", "")
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
            "rule": rule,
            "used_memory_indices": used_memory_indices,
        }
        processor.direct_store_memory(**data_batch_memory_manager)
    return jsonify({
        "status": "success"
    })


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
        
    processor = MemoryProcessor()
    app.run(host=args.host, port=args.port)