"""
GT 奖励核心逻辑（原 ``local_search/mmsearch.compute_score`` 中与标准答案相关的部分）。

供 ``gt_reward_server.py`` 与 ``mmsearch.compute_score``（HTTP 客户端侧）共用。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from openai import OpenAI

from local_search.prompt import JUDGE_PROMPT

logger = logging.getLogger(__name__)

openai_api_key = "EMPTY"
openai_api_base = os.getenv("JUDGE_URL")
_judge_client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
model_name = "qwen"


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


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


def extract_answer(solution_str: str) -> str:
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


def judge_answer2(question_text: str, ans: str, ground_truth: str) -> bool:
    if not ans.strip():
        return False
    user_prompt = JUDGE_PROMPT.format(
        question=question_text,
        correct_answer=ground_truth,
        response=ans,
    )
    try:
        chat_response = _judge_client.chat.completions.create(
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
    except Exception:
        print("Judger Connect Error")
        pt = ans.lower().strip()
        gt = str(ground_truth).lower().strip()
        if gt in pt:
            return True
        return False


def normalize_ground_truth(gt: Any) -> str:
    if gt is None:
        return ""
    if isinstance(gt, list):
        if not gt:
            return ""
        return str(gt[0])
    return str(gt)


def compute_gt_reward_core(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> dict[str, Any]:
    """
    与原先 ``mmsearch.compute_score`` 中 GT 相关部分一致（不含 memory_bank HTTP）。

    Returns:
        dict 含 ``score`` (float)、``judgement`` (\"correct\"|\"incorrect\")、若干调试字段。
    """
    extra_info = extra_info or {}
    format_score = 0.0
    decision_score = 0.0
    messages = extra_info.get("messages", [])
    question_text = extra_info.get("question", "")
    if len(messages) not in (3, 6):
        return {
            "score": 0.0,
            "judgement": "incorrect",
            "reason": "bad_message_len",
            "format_score": 0.0,
            "decision_score": 0.0,
        }

    answer1 = extract_answer(messages[1])
    answer2 = extract_answer(messages[5]) if len(messages) == 6 else ""
    do_replan = extract_after_think(messages[2]).lower()
    if do_replan not in ["yes", "no"]:
        format_score = 0.0
    else:
        format_score = 1.0

    indices_to_check = [0, 2]
    if len(messages) == 6:
        indices_to_check.append(4)
    for idx in indices_to_check:
        msg = messages[idx]
        open_count = msg.count("<think>")
        close_count = msg.count("</think>")
        if open_count != close_count:
            format_score = 0.0
            break

    gt = normalize_ground_truth(ground_truth)
    is_answer1_correct = judge_answer2(question_text, answer1, gt)
    if len(messages) == 6:
        is_answer2_correct = judge_answer2(question_text, answer2, gt)
    else:
        is_answer2_correct = is_answer1_correct

    if is_answer1_correct and "no" in do_replan:
        decision_score = 1.0
    elif (not is_answer1_correct) and "yes" in do_replan:
        decision_score = 1.0
    else:
        decision_score = 0.0

    accuracy_score1 = 1.0 if is_answer1_correct else 0.0
    accuracy_score2 = 1.0 if is_answer2_correct else 0.0
    final_score = 0.7 * accuracy_score2 + 0.2 * accuracy_score1 + 0.05 * format_score + 0.05 * decision_score
    judgement = "correct" if is_answer2_correct else "incorrect"

    return {
        "score": float(final_score),
        "judgement": judgement,
        "format_score": format_score,
        "decision_score": decision_score,
        "is_answer1_correct": is_answer1_correct,
        "is_answer2_correct": is_answer2_correct,
        "data_source": data_source,
    }
