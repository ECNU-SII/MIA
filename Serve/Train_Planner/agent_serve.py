from flask import Flask, request, jsonify
import pandas as pd
import os
from call_agent import run_multimodal_dialogue, continue_multimodal_dialogue
from typing import List, Dict, Any, Tuple, Optional

app = Flask(__name__)


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



DATA_ID_TO_IMAGES = {}

def load_parquet_images(parquet_path: str):
    global DATA_ID_TO_IMAGES
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    for _, row in df.iterrows():
        data_id = str(row['data_id'])
        image_data = row['images']
        if image_data is None:
            image_data = []
        elif not isinstance(image_data, list):
            image_data = [image_data]
        DATA_ID_TO_IMAGES[data_id] = image_data
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
    


if __name__ == '__main__':
    parquet_path1 = '/your_path/datasets/fvqa_matpo_train_final.parquet'
    parquet_path2 = '/your_path/datasets/fvqa_test_final.parquet'
    load_parquet_images(parquet_path1)
    load_parquet_images(parquet_path2)
    app.run(host='0.0.0.0', port=5000)
