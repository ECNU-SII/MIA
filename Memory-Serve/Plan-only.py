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
import requests


dotenv.load_dotenv()
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_NAME = "qwen"
SERVER_URL = None
SAVE_PATH = None

memory_manager_system_prompt = """
### Memory Management System

As a memory management system, your primary task is to extract high-value, generalizable insights from the reasoning trajectories of a multimodal agent. Your goal is to provide practical guidance for similar future tasks by identifying successful strategies or learning from mistakes.

#### Key Responsibilities:
- For **correct reasoning**, extract and store reusable experiences.
- For **incorrect reasoning**, identify the root cause and suggest ways to avoid repeating the same mistakes.

Since you are a **unimodal model**, you do not have direct access to image content. Therefore, all visual information must be inferred solely based on textual descriptions.

### Input Structure:

You will receive the following inputs:
1. **[judgement]**: {judgement} 
   - Indicates whether the interaction was correct or incorrect.
   - If "correct", focus on extracting successful strategies.
   - If "incorrect", focus on lessons learned from failures.
   
2. **[question]**: {question}
   - The original question asked by the user.
   
3. **[image_caption]**: {image_caption}
   - A text description of the input image.
   
4. **[trajectory]**: {trace}
   - The complete action trajectory of the multimodal agent, including internal reasoning steps, tool invocations, intermediate inferences, and final answer.

#### Action Definitions in Trajectory:
- `<think> ... </think>`: Contains the agent's thinking content.
- `<tool_call> ... </tool_response>`: Represents tool calls made by the agent.
  - **web_image_to_image_search**: Searches the web using an input image.
  - **search**: Searches the web for text content.
- `<answer> ... </answer>`: Final answer provided by the agent.

{similar_memory_prompt}

### Memory Management Tools:

You can manage your memory through the following tools:
- **memory_insert**: Add a new memory entry that carries the experience.
- **memory_update**: Merge the most similar memory entry with new insights to get an updated memory entry.

**Note:** You must call the memory manage tools once and only once.
"""


planning_system_prompt = """
You are a senior expert assisting an agent by providing strategic guidance. 

The agent has access to the following tools:
- `search`: perform text-based web queries to retrieve external information.

### [Relevant Memories] 
({memory_len} retrieved experiences) Each memory indicates whether it was **correct** or **incorrect**, along with its workflow:
{memory}

### Your Task:
- Analyze the **Question** and compare it with both **correct strategies** and **incorrect patterns** in the memories.
- If similar successful approaches exist, adapt their core idea to guide the next steps.
- If the context resembles past failures, highlight what to avoid and suggest corrective actions.
- Recommend **one clear, generalizable work plan or action strategy** that directly addresses the current situation and guides the agent on what to do next.

### Output should:
1. Be clear and concise—no more than 200 words.
2. You must include relevant memories in the thinking process (<think>...</think>), but do not mention them in the final output.
3. Don't try to give the answer directly, but give a plan.
4. Prohibit the generation of content unrelated to the plan.
5. Your response must not contain any factual information.

Your output should only contain the guideline, with no additional explanations.

**[Question]** (Global Objective): 
{question}
"""

planning_system_prompt_img = """
You are a senior expert assisting an agent by providing strategic guidance. 

The agent has access to the following tools:
- `search`: perform text-based web queries to retrieve external information.
- `web_image_to_image_search`: find visually similar images online (Can only be used once).

### [Relevant Memories] 
({memory_len} retrieved experiences) Each memory indicates whether it was **correct** or **incorrect**, along with its workflow:
{memory}

### Your Task:
- Analyze the **Question** and compare it with both **correct strategies** and **incorrect patterns** in the memories.
- If similar successful approaches exist, adapt their core idea to guide the next steps.
- If the context resembles past failures, highlight what to avoid and suggest corrective actions.
- Recommend **one clear, generalizable work plan or action strategy** that directly addresses the current situation and guides the agent on what to do next.

### Output should:
1. Be clear and concise—no more than 200 words.
2. You must include relevant memories in the thinking process (<think>...</think>), but do not mention them in the final output.
3. Don't try to give the answer directly, but give a plan.
4. Prohibit the generation of content unrelated to the plan.
5. Your response must not contain any factual information.

Your output should only contain the guideline, with no additional explanations.

**IMPORTANT: The `web_image_to_image_search` tool can only be called once.** Otherwise, the agent will be severely penalized.

**[Question]** (Global Objective): 
{question}
"""


guide_system_prompt = """
You are a senior expert assisting an agent by providing strategic guidance based on the provided context summary. The agent has given you a detailed context of their current situation, including any context or previous tool usage. Your role is to analyze this information and provide a clear, actionable plan for the next steps.

The agent has access to the following tools:
- `search`: perform text-based web queries to retrieve external information.
- `memory`: consult expert memory for strategic advice (the current channel).

### [Relevant Memories] 
({memory_len} retrieved experiences) Each memory indicates whether it was **correct** or **incorrect**, along with its workflow:
{memory}

### Your Task:
- Analyze the **Context** and compare it with both **correct strategies** and **incorrect patterns** in the memories.
- Analyze **Workflow** to understand current task progress and avoid creating repetitive or redundant plans.
- If similar successful approaches exist, adapt their core idea to guide the next steps.
- If the context resembles past failures, highlight what to avoid and suggest corrective actions.
- Recommend **one clear, generalizable work plan or action strategy** that directly addresses the current situation and guides the agent on what to do next.

### Output should:
1. Be clear and concise—no more than 200 words.
2. You must include relevant memories in the thinking process (<think>...</think>), but do not mention them in the final output.
3. Don't try to give the answer directly, but give a plan.
4. Prohibit the generation of content unrelated to the plan.
5. Your response must not contain any factual information.

Your output should only contain the guideline, with no additional explanations.

**[Question]** (Global Objective): 
{question}

**[Workflow]** (Tool Usage Result): 
{workflow}

**[Context]** (Current Task Status): 
{context}
"""

guide_system_prompt_img = """
You are a senior expert assisting an agent by providing strategic guidance based on the provided context summary. The agent has given you a detailed context of their current situation, including any context or previous tool usage. Your role is to analyze this information and provide a clear, actionable plan for the next steps.

The agent has access to the following tools:
- `search`: perform text-based web queries to retrieve external information.
- `web_image_to_image_search`: find visually similar images online (Can only be used once).
- `memory`: consult expert memory for strategic advice (the current channel).

### [Relevant Memories] 
({memory_len} retrieved experiences) Each memory indicates whether it was **correct** or **incorrect**, along with its workflow:
{memory}

### Your Task:
- Analyze the **Context** and compare it with both **correct strategies** and **incorrect patterns** in the memories.
- Analyze **Workflow** to understand current task progress and avoid creating repetitive or redundant plans.
- If similar successful approaches exist, adapt their core idea to guide the next steps.
- If the context resembles past failures, highlight what to avoid and suggest corrective actions.
- Recommend **one clear, generalizable work plan or action strategy** that directly addresses the current situation and guides the agent on what to do next.

### Output should:
1. Be clear and concise—no more than 200 words.
2. You must include relevant memories in the thinking process (<think>...</think>), but do not mention them in the final output.
3. Don't try to give the answer directly, but give a plan.
4. Prohibit the generation of content unrelated to the plan.
5. Your response must not contain any factual information.

Your output should only contain the guideline, with no additional explanations.

**IMPORTANT: The `web_image_to_image_search` tool can only be called once.** Otherwise, the agent will be severely penalized.

**[Question]** (Global Objective): 
{question}

**[Workflow]** (Tool Usage Result): 
{workflow}

**[Context]** (Current Task Status): 
{context}
"""



def filter_result(result: str) -> str:
    marker = "</think>"
    if marker in result:
        return result.split(marker, 1)[1].strip()
    else:
        return result.strip()

class MemoryProcessor:
    def __init__(self, server_url=None):
        """Initialize the OpenAI client based on model configuration."""
        self.model_name = MODEL_NAME
        bert_path = "/your_path/bert/sup-simcse-bert-base-uncased"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bert_model_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model_encoder = AutoModel.from_pretrained(bert_path).to(self.device)
        self.bert_model_encoder.eval()
        if server_url:
            base_url = server_url
        else:
            base_url = os.getenv("QWEN_URL")
            
        self.yunwu_key = os.getenv("YUNWU_KEY")
        
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY"
        )
        
        
        self.question_weight = 0.5
        self.caption_weight = 0.5
        self.question_embeddings: Optional[torch.Tensor] = None   # (N, D)
        self.caption_embeddings: Optional[torch.Tensor] = None    # (N, D)
        self.memory_data: List[Dict[str, Any]] = []
        self.data_ids: List[str] = []
        
        

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
        if text:  # only normalize non-zero vectors
            embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding
    
    def add_memory(self, data_id: str, memory_entry: Dict[str, Any]):
        if not data_id:
            data_id = ""
        q_vec = self._encode_text(memory_entry['question'])          # (D,)
        ic_vec = self._encode_text(memory_entry['image_caption'])    # (D,)
        self.data_ids.append(data_id)
        self.memory_data.append(memory_entry)
        q_vec = q_vec.unsqueeze(0)   # (1, D)
        ic_vec = ic_vec.unsqueeze(0) # (1, D)
        if self.question_embeddings is None:
            self.question_embeddings = q_vec
            self.caption_embeddings = ic_vec
        else:
            self.question_embeddings = torch.cat([self.question_embeddings, q_vec], dim=0)
            self.caption_embeddings = torch.cat([self.caption_embeddings, ic_vec], dim=0)
        logger.info(f"Added memory. Total: {len(self.data_ids)}")

    def update_memory(self, data_id: str, new_memory_entry: Dict[str, Any]):
        try:
            idx = self.data_ids.index(data_id)
        except ValueError:
            logger.error(f"data_id '{data_id}' not found in memory.")
            self.add_memory(data_id, new_memory_entry)
            return
        q_vec = self._encode_text(new_memory_entry['question'])
        ic_vec = self._encode_text(new_memory_entry['image_caption'])
        self.memory_data[idx] = new_memory_entry
        self.data_ids[idx] = data_id  # redundant but safe
        self.question_embeddings[idx] = q_vec
        self.caption_embeddings[idx] = ic_vec
        logger.info(f"Updated memory for data_id='{data_id}'.")
    
    def retrieve_balanced_memories(
        self,
        query_question: str,
        query_image_caption: str,
        exclude_data_id: Optional[str] = None,
        pos_top_k: int = 3,
        neg_top_k: int = 3
    ) -> List[Dict[str, Any]]:
        if len(self.data_ids) == 0:
            return [], []
        q_query_vec = self._encode_text(query_question).unsqueeze(0)          # (1, D)
        ic_query_vec = self._encode_text(query_image_caption).unsqueeze(0)    # (1, D)
        with torch.no_grad():
            sim_q = F.cosine_similarity(q_query_vec, self.question_embeddings)      # (N,)
            if query_image_caption:
                sim_ic = F.cosine_similarity(ic_query_vec, self.caption_embeddings)     # (N,)
                total_sim = self.question_weight * sim_q + self.caption_weight * sim_ic  # (N,)
            else:
                total_sim = sim_q
        total_sim = total_sim.cpu()
        # Build list of (sim, idx, entry) for filtering
        candidates = []
        for idx, (data_id, entry) in enumerate(zip(self.data_ids, self.memory_data)):
            if exclude_data_id and data_id and data_id == exclude_data_id:
                continue
            candidates.append((total_sim[idx].item(), idx, entry))
        # Separate by judgement
        correct_candidates = [c for c in candidates if c[2].get("judgement") == "correct"]
        incorrect_candidates = [c for c in candidates if c[2].get("judgement") == "incorrect"]
        # Sort by similarity (descending)
        correct_candidates.sort(key=lambda x: x[0], reverse=True)
        incorrect_candidates.sort(key=lambda x: x[0], reverse=True)
        # Take top-k
        correct_selected = [self.memory_data[c[1]] for c in correct_candidates[:pos_top_k]]
        incorrect_selected = [self.memory_data[c[1]] for c in incorrect_candidates[:neg_top_k]]
        return correct_selected, incorrect_selected

    def find_nearest_memory(self, query_question: str, query_image_caption: str, top_k: int = 1) -> List[Tuple[str, float]]:
        if len(self.data_ids) == 0:
            return []
        q_query_vec = self._encode_text(query_question).unsqueeze(0)          # (1, D)
        ic_query_vec = self._encode_text(query_image_caption).unsqueeze(0)    # (1, D)
        with torch.no_grad():
            sim_q = F.cosine_similarity(q_query_vec, self.question_embeddings)      # (N,)
            if query_image_caption:
                sim_ic = F.cosine_similarity(ic_query_vec, self.caption_embeddings)     # (N,)
                total_sim = self.question_weight * sim_q + self.caption_weight * sim_ic  # (N,)
            else:
                total_sim = sim_q
        total_sim = total_sim.cpu()
        top_k = min(top_k, total_sim.size(0))
        top_indices = torch.topk(total_sim, top_k).indices.tolist()
        scores = total_sim[top_indices].tolist()
        return [(self.data_ids[i], score) for i, score in zip(top_indices, scores)]

    def extract_and_store_memory(self, data_id: str, question: str, image_caption: str, trace: str, judgement: str):
        """Main function: call LLM to extract memory and store it."""
        similar = self.find_nearest_memory(
            query_question=question,
            query_image_caption=image_caption,
            top_k=1
        )
        similar_memory_prompt = ""
        if similar:

            mid, score = similar[0]
            idx = self.data_ids.index(mid)
            entry = self.memory_data[idx]
            if not entry['image_caption'].strip():
                similar_display_caption = "[No image provided]"
            else:
                similar_display_caption = entry['image_caption']
            similar_memory_prompt = (
                f"\n### Most Similar Existing Memory (similarity={score:.2f}):\n"
                f"- Tag: {entry['tag']}\n"
                f"- Judgement: {entry['judgement']}\n"
                f"- Question: {entry['question']}\n"
                f"- Image Caption: {similar_display_caption}\n"
                f"- Workflow Summary: {entry['workflow_summary']}\n"
            )
            update_idx = self.data_ids[idx]
            
        if not image_caption.strip():
            display_caption = "[No image provided]"
        else:
            display_caption = image_caption
                
        user_prompt = memory_manager_system_prompt.format(judgement=judgement, question=question, image_caption=display_caption, trace=trace, similar_memory_prompt=similar_memory_prompt)
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        tools = get_memory_tool_schemas()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                temperature=0.7,
                max_tokens=8192,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            return {"status": "error", "message": str(e)}
        msg = response.choices[0].message
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
        else:
            content = msg.content or ""
            match = re.search(r'<tool_call>\s*({.*})\s*</tool_call>', content, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(1))
                    func_name = payload.get("name")
                    func_args = payload.get("arguments", {})
                    tool_call = {
                        "function": {"name": func_name, "arguments": json.dumps(func_args)}
                    }
                except Exception as e:
                    logging.error(f"Failed to parse tool call from content: {e}")
                    return {"status": "error", "message": "Invalid tool call format in content"}
            else:
                return {"status": "no_tool_call", "message": "No tool call found in content or tool_calls."}
        new_entry = {"data_id": data_id, "question": question, "image_caption": image_caption, "judgement": judgement}
        new_entry["tag"] = func_args.get("tag", "")
        new_entry["workflow_summary"] = func_args.get("workflow_summary", "")
        new_entry["experience_summary"] = func_args.get("experience_summary", "")
        new_entry["memory_description"] = func_args.get("memory_description", "")
        
        if func_name == "memory_insert":
            self.add_memory(data_id, new_entry)
            return {"status": "inserted", "id": data_id}
        elif func_name == "memory_update":
            self.update_memory(update_idx, new_entry)
            return {"status": "updated", "id": update_idx}
        else:
            return {"status": "invalid_tool_or_no_similar", "tool": func_name}


    def direct_store_memory(self, data_id: str, question: str, image_caption: str, trace: str, judgement: str):
        new_entry = {"data_id": data_id, "question": question, "image_caption": image_caption, "workflow_summary": trace, "judgement": judgement}
        self.add_memory(data_id, new_entry)
    

    def get_memories_context(self, data_id, question: str, image_caption: str) -> str:
        """Construct the planning prompt (without calling LLM)."""
        try:
            pos_memories, neg_memories = self.retrieve_balanced_memories(
                query_question=question,
                query_image_caption=image_caption,
                exclude_data_id=data_id,
                pos_top_k=5,
                neg_top_k=3
            )
        except:
            pos_memories, neg_memories = [], []

        memories_context = ""
        if pos_memories or neg_memories:
            memories_context += "\n### Retrieved Relevant Memories:\n"
            if pos_memories:
                memories_context += f"\n#### Positive Examples (Successful Strategies - {len(pos_memories)}):\n"
                for i, entry in enumerate(pos_memories, 1):
                    if not entry['image_caption'].strip():
                        similar_display_caption = "[No image provided]"
                    else:
                        similar_display_caption = entry['image_caption']
                    memories_context += (
                        f"\n--- Example {i} ---\n"
                        f"- Judgement: {entry['judgement']}\n"
                        f"- Question: {entry['question']}\n"
                        f"- Image Caption: {similar_display_caption}\n"
                        f"- Workflow Summary: {entry.get('workflow_summary', 'N/A')}\n"
                    )
            if neg_memories:
                memories_context += f"\n#### Negative Examples (Failure Lessons - {len(neg_memories)}):\n"
                for i, entry in enumerate(neg_memories, 1):
                    if not entry['image_caption'].strip():
                        similar_display_caption = "[No image provided]"
                    else:
                        similar_display_caption = entry['image_caption']
                    memories_context += (
                        f"\n--- Example {i} ---\n"
                        f"- Judgement: {entry['judgement']}\n"
                        f"- Question: {entry['question']}\n"
                        f"- Image Caption: {similar_display_caption}\n"
                        f"- Workflow Summary: {entry.get('workflow_summary', 'N/A')}\n"
                    )
        else:
            memories_context = "\n### Retrieved Relevant Memories:\nNone found."
        
        return memories_context
    
    
    def build_plan(self, data_id, question: str, image_caption: str) -> str:
        pos_memories, neg_memories = [], [] 
        memories_context = "\n### Retrieved Relevant Memories:\nNone found."
        memory_len = len(pos_memories) + len(neg_memories)
        
        print("plan", len(pos_memories), len(neg_memories))
        
        if image_caption:
            user_prompt = planning_system_prompt_img.format(
                memory_len=memory_len,
                question=question,
                memory=memories_context
            )
        else:
            user_prompt = planning_system_prompt.format(
                memory_len=memory_len,
                question=question,
                memory=memories_context
            )
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        if "qwen" in self.model_name:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=8192,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            output = response.choices[0].message.content
            return output
        else:
            payload = json.dumps({
                "model": self.model_name,
                "max_tokens": 10000,
                "messages": messages,
                "temperature": 1,
            })
            headers = {
                'Accept': 'application/json',
                'Authorization': f"Bearer {self.yunwu_key}",
                'Content-Type': 'application/json'
            }
            conn = http.client.HTTPSConnection("yunwu.ai", timeout=300)
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            data = res.read()
            data = json.loads(data.decode('utf-8'))
            output = data["choices"][0]["message"]["content"]
            conn.close()
            return output


    def build_guide(self, data_id: str, question: str, image_caption: str, context: str, workflow: str) -> str:
        pos_memories, neg_memories = [], []
        memories_context = "\n### Retrieved Relevant Memories:\nNone found."
        memory_len = len(pos_memories) + len(neg_memories)
        print(len(pos_memories), len(neg_memories))
        if image_caption:
            user_prompt = guide_system_prompt_img.format(
                question=question,
                memory_len=memory_len,
                memory=memories_context,
                context=context,
                workflow=workflow,
            )
        else:
            user_prompt = guide_system_prompt.format(
                question=question,
                memory_len=memory_len,
                memory=memories_context,
                context=context,
                workflow=workflow,
            )
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        if "qwen" in self.model_name:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=8192,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            output = response.choices[0].message.content
            return output
        else:
            payload = json.dumps({
                "model": self.model_name,
                "max_tokens": 10000,
                "messages": messages,
                "temperature": 1,
            })
            headers = {
                'Accept': 'application/json',
                'Authorization': f"Bearer {self.yunwu_key}",
                'Content-Type': 'application/json'
            }
            conn = http.client.HTTPSConnection("yunwu.ai", timeout=300)
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            data = res.read()
            data = json.loads(data.decode('utf-8'))
            output = data["choices"][0]["message"]["content"]
            conn.close()
            return output



    def save_memory(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(processor.memory_data, f, indent=4)


    def load_memory_from_file(self, file_path: str):
        if not os.path.exists(file_path):
            logger.warning(f"Memory file not found: {file_path}. Skipping load.")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_memories = json.load(f)

        if not isinstance(raw_memories, list):
            raise ValueError("Memory file must contain a JSON list.")

        logger.info(f"Loading {len(raw_memories)} memory entries from {file_path}...")
        self.data_ids.clear()
        self.memory_data.clear()
        self.question_embeddings = None
        self.caption_embeddings = None
        for entry in tqdm(raw_memories, desc="Encoding memories"):
            data_id = entry.get("data_id")
            if not data_id:
                logger.warning("Skipping memory entry without 'data_id'")
                continue
            caption = entry.get("image_caption", "")
            if isinstance(caption, list):
                caption = " ".join(str(c) for c in caption)
            elif not isinstance(caption, str):
                caption = str(caption)

            question = str(entry.get("question", ""))

            try:
                q_vec = self._encode_text(question).unsqueeze(0)      # (1, D)
                ic_vec = self._encode_text(caption).unsqueeze(0)     # (1, D)
            except Exception as e:
                logger.error(f"Failed to encode memory {data_id}: {e}")
                continue
            self.data_ids.append(data_id)
            self.memory_data.append(entry)

            if self.question_embeddings is None:
                self.question_embeddings = q_vec
                self.caption_embeddings = ic_vec
            else:
                self.question_embeddings = torch.cat([self.question_embeddings, q_vec], dim=0)
                self.caption_embeddings = torch.cat([self.caption_embeddings, ic_vec], dim=0)

        logger.info(f"Successfully loaded and encoded {len(self.data_ids)} memories.")

@app.route('/batch_memory_manager', methods=['POST'])
def batch_memory_manager():
    try:
        data = request.get_json()
        for entry in data:
            data_id = entry.get('data_id', "")
            question = entry.get('question', "")
            image_caption = entry.get('image_caption', "")
            messages = entry["messages"]
            trace = ""
            for message in messages:
                if message["role"] == "assistant":
                    trace += f"### Assistant Reasoning:\n\n{message['content']}\n\n"
                else:
                    trace += f"### Tool Call Return Results:\n\n{message['content']}\n\n"
            judgement = "correct" if entry["judgement"] == "correct" else "incorrect"
            processor.extract_and_store_memory(data_id, question, image_caption, trace, judgement)
        return jsonify({
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/memory', methods=['POST'])
def memory():
    data = request.get_json()
    results = []
    for item in data:
        data_id = item.get('data_id', "").strip()
        question = item.get("question", "").strip()
        image_captions = item.get("image_caption", [])
        if isinstance(image_captions, list):
            if len(image_captions)>0:
                image_caption = image_captions[0].strip()
            else:
                image_caption = ""
        else:
            image_caption = image_captions
        result = processor.get_memories_context(data_id, question, image_caption)
        results.append(result.strip())
    return jsonify(results)


@app.route('/plan', methods=['POST'])
def plan():
    data = request.get_json()
    results = []
    for item in data:
        data_id = item.get('data_id', "").strip()
        question = item.get("question", "").strip()
        image_captions = item.get("image_caption", [])
        if isinstance(image_captions, list):
            if len(image_captions)>0:
                image_caption = image_captions[0].strip()
            else:
                image_caption = ""
        else:
            image_caption = image_captions
        result = processor.build_plan(data_id, question, image_caption)
        result = filter_result(result)
        results.append(result)
    return jsonify(results)


@app.route('/guide', methods=['POST'])
def guide():
    data = request.get_json()
    results = []
    item = data[0]
    data_id = item.get('data_id', "")
    question = item.get("question", "").strip()
    image_captions = item.get("image_caption", [])
    if isinstance(image_captions, list):
        if len(image_captions)>0:
            image_caption = image_captions[0].strip()
        else:
            image_caption = ""
    else:
        image_caption = image_captions
    context = item.get("context", "").strip()
    workflow = item.get("workflow", "").strip()
    if context and workflow:
        print("yes")
    result = processor.build_guide(data_id, question, image_caption, context, workflow)
    result = filter_result(result)
    results = [result,]
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
    try:
        data = request.get_json()
        for item in data:
            data_id = item.get('data_id', "")
            question = item.get("question", "").strip()
            image_caption = item.get("image_caption", "").strip()
            trace = ""
            messages = item["messages"]
            j = 1
            for message in messages:
                if message["role"] == "assistant":
                    trace += f"## Round {j}:\n\n"
                    trace += f"### Agent Reasoning and Tool Call:\n{message['content']}\n"
                    if """<tool_call>\n{\"name\": \"web_image_to_image_search\"""" in message['content']:
                        trace += f"### The \"web_image_to_image_search\"(image search) Tool Call Return Results.\n"
                    if """<tool_call>\n{\"name\": \"search\"""" in message['content']:
                        trace += f"### The \"search\"(text search) Tool Call Return Results.\n"
                    if """<tool_call>\n{\"name\": \"memory\"""" in message['content']:
                        trace += f"### The \"memory\"(expert guidance on experiential memory) Tool Call Return Results.\n"
                    j += 1
            judgement = "correct" if item["judgement"] == "correct" else "incorrect"
            data_batch_memory_manager = [
                {
                    "data_id": data_id,
                    "question": question,
                    "image_caption": image_caption,
                    "trace": trace,
                    "judgement": judgement
                },
            ]
            processor.direct_store_memory(**data_batch_memory_manager[0])
            print(len(processor.memory_data), len(processor.data_ids), processor.question_embeddings.shape, processor.caption_embeddings.shape)
        return jsonify({
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    processor.data_ids.clear()
    processor.memory_data.clear()
    processor.question_embeddings = None
    processor.caption_embeddings = None
    logger.info("All in-memory memories have been cleared.")
    return jsonify({"status": "success", "message": "Memory cleared successfully."})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Memory-powered Question Answering Server')
    parser.add_argument('--server_url',
                      help='Server URL for the model API (for Qwen models)',
                      default=None)
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
        
    processor = MemoryProcessor(server_url=args.server_url)
    app.run(host=args.host, port=args.port)
    



