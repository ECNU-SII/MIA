import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import threading
from datetime import datetime
from react_agent import MultiTurnReactAgent
import time
import math
import base64
from PIL import Image
import io
from typing import Dict, List, Union, Any
from judge import judge_answer


def validate_multimodal_data(item: Dict[str, Any]) -> bool:
    """验证多模态数据格式是否正确"""
    if not isinstance(item, dict):
        return False
    
    # 检查必要字段
    if "question" not in item and "messages" not in item:
        return False
    
    # 检查图像数据格式
    if "images" in item:
        images = item["images"]
        if not isinstance(images, list):
            return False
        
        for img in images:
            if not isinstance(img, str):
                return False
            # 验证base64格式
            try:
                if img.startswith("data:image"):
                    base64_data = img.split(",")[1]
                    base64.b64decode(base64_data)
                else:
                    base64.b64decode(img)
            except:
                return False
    
    return True

def process_multimodal_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """处理多模态数据项，确保格式正确"""
    processed_item = item.copy()
    
    # 确保有question字段
    if "question" not in processed_item:
        if "messages" in processed_item:
            try:
                user_msg = processed_item["messages"][1]["content"]
                question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
                processed_item["question"] = question
            except:
                processed_item["question"] = "No question found"
        else:
            processed_item["question"] = "No question found"
    
    # 确保images字段存在且格式正确
    if "images" not in processed_item:
        processed_item["images"] = []
    
    # 验证图像数据
    valid_images = []
    for img in processed_item["images"]:
        try:
            if isinstance(img, str):
                if img.startswith("data:image"):
                    base64_data = img.split(",")[1]
                    base64.b64decode(base64_data)
                    valid_images.append(img)
                else:
                    # 假设是纯base64数据
                    base64.b64decode(img)
                    valid_images.append(f"data:image;base64,{img}")
        except:
            print(f"Warning: Invalid image data in item, skipping: {processed_item.get('question', 'Unknown')}")
    
    processed_item["images"] = valid_images
    return processed_item

def extract_question_from_item(item: Dict[str, Any]) -> str:
    """从数据项中提取问题文本"""
    if "question" in item:
        return item["question"].strip()
    elif "messages" in item:
        try:
            user_msg = item["messages"][1]["content"]
            return user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
        except:
            return str(item)
    else:
        return str(item)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--dataset", type=str, default="gaia")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--presence_penalty", type=float, default=1.1)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=10000)
    parser.add_argument("--roll_out_count", type=int, default=1)
    parser.add_argument("--main_ports", type=str, default="8000", help="Comma-separated list of main model ports")
    args = parser.parse_args()

    model = args.model
    output_base = args.output
    roll_out_count = args.roll_out_count

    model_name = os.path.basename(model.rstrip('/'))

    dataset_dir = output_base
    
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"Model name: {model_name}")
    print(f"Data set path: {args.dataset}")
    print(f"Output directory: {dataset_dir}")
    print(f"Number of rollouts: {roll_out_count}")


    data_filepath = f"{args.dataset}"
    try:
        if data_filepath.endswith(".json"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
            if items and not isinstance(items[0], dict):
                raise ValueError("Input JSON list items must be objects.")
        elif data_filepath.endswith(".jsonl"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")
        
        # 处理多模态数据
        print("Processing multimodal data...")
        processed_items = []
        multimodal_count = 0
        invalid_count = 0
        
        for i, item in enumerate(items):
            try:
                # 验证多模态数据格式
                if validate_multimodal_data(item):
                    processed_item = process_multimodal_item(item)
                    processed_items.append(processed_item)
                    
                    # 统计多模态数据
                    if processed_item.get("images"):
                        multimodal_count += 1
                else:
                    print(f"Warning: Invalid multimodal data format at index {i}, skipping")
                    invalid_count += 1
            except Exception as e:
                print(f"Warning: Error processing item at index {i}: {e}")
                invalid_count += 1
        
        items = processed_items
        print(f"Data processing complete:")
        print(f"  - Total items: {len(items)}")
        print(f"  - Multimodal items (with images): {multimodal_count}")
        print(f"  - Invalid items skipped: {invalid_count}")
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_filepath}")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing input file {data_filepath}: {e}")
        exit(1)

    # Apply data splitting
    total_items = len(items)

    output_files = {i: os.path.join(dataset_dir, f"iter{i}.jsonl") for i in range(1, roll_out_count + 1)}

    processed_queries_per_rollout = {}

    for rollout_idx in range(1, roll_out_count + 1):
        output_file = output_files[rollout_idx]
        processed_queries = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "question" in data and "error" not in data:
                                processed_queries.add(data["question"].strip())
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid line in output file: {line.strip()}")
            except FileNotFoundError:
                pass
        processed_queries_per_rollout[rollout_idx] = processed_queries

    tasks_to_run_all = []
    per_rollout_task_counts = {i: 0 for i in range(1, roll_out_count + 1)}
    # Parse main_ports from argument
    planning_ports = [int(port.strip()) for port in args.main_ports.split(',') if port.strip()]
    # Round-robin state
    planning_rr_idx = 0
    summary_rr_idx = 0
    # Sticky assignment per question
    question_to_ports = {}
    for rollout_idx in range(1, roll_out_count + 1):
        processed_queries = processed_queries_per_rollout[rollout_idx]
        for item in items:
            question = extract_question_from_item(item)
            if not question:
                print(f"Warning: Skipping item with empty question: {item}")
                continue

            if question not in processed_queries:
                # Ensure sticky and balanced port assignment per unique question
                if question not in question_to_ports:
                    planning_port = planning_ports[planning_rr_idx % len(planning_ports)]
                    question_to_ports[question] = planning_port
                    planning_rr_idx += 1
                planning_port = question_to_ports[question]
                
                # 创建任务，包含多模态信息
                task = {
                    "item": item.copy(),
                    "rollout_idx": rollout_idx,
                    "planning_port": planning_port,
                }
                
                # 添加多模态元数据
                if item.get("images"):
                    task["multimodal"] = True
                    task["image_count"] = len(item["images"])
                else:
                    task["multimodal"] = False
                    task["image_count"] = 0
                if "image_caption" in item.keys():
                    task["image_caption"] = item["image_caption"]
                tasks_to_run_all.append(task)
                per_rollout_task_counts[rollout_idx] += 1

    # 统计多模态任务
    multimodal_tasks = sum(1 for task in tasks_to_run_all if task.get("multimodal", False))
    text_only_tasks = len(tasks_to_run_all) - multimodal_tasks
    
    print(f"Total questions in current split: {len(items)}")
    print(f"Task breakdown:")
    print(f"  - Text-only tasks: {text_only_tasks}")
    print(f"  - Multimodal tasks: {multimodal_tasks}")
    print(f"  - Total tasks to run: {len(tasks_to_run_all)}")
    
    for rollout_idx in range(1, roll_out_count + 1):
        print(f"Rollout {rollout_idx}: already successfully processed: {len(processed_queries_per_rollout[rollout_idx])}, to run: {per_rollout_task_counts[rollout_idx]}")

    if not tasks_to_run_all:
        print("All rollouts have been completed and no execution is required.")
    else:
        llm_cfg = {
            'model': model,
            'generate_cfg': {
                'max_input_tokens': 320000,
                'max_retries': 10,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'presence_penalty': args.presence_penalty
            },
            'model_type': 'qwen_dashscope'
        }

        test_agent = MultiTurnReactAgent(
            llm=llm_cfg,
            function_list=["search"]
        )

        write_locks = {i: threading.Lock() for i in range(1, roll_out_count + 1)}

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    test_agent._run,
                    task,
                    model
                ): task for task in tasks_to_run_all
            }

            for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run_all), desc="Processing All Rollouts"):
                task_info = future_to_task[future]
                rollout_idx = task_info["rollout_idx"]
                output_file = output_files[rollout_idx]
                try:
                    result = future.result()
                    
                    # 添加多模态元数据到结果中
                    if isinstance(result, dict):
                        result["multimodal"] = task_info.get("multimodal", False)
                        result["image_count"] = task_info.get("image_count", 0)
                        result["rollout_idx"] = rollout_idx
                        result["rollout_id"] = rollout_idx
                    
                    judgement = judge_answer(result["question"], result["prediction"], result["answer"])
                    
                    if "image_caption" in item.keys():
                        result["image_caption"] = task_info["image_caption"]
                    result["judgement"] = judgement
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                except concurrent.futures.TimeoutError:
                    question = extract_question_from_item(task_info["item"])
                    is_multimodal = task_info.get("multimodal", False)
                    image_count = task_info.get("image_count", 0)
                    
                    print(f'Timeout (>1800s): "{question}" (Rollout {rollout_idx}) [Multimodal: {is_multimodal}, Images: {image_count}]')
                    future.cancel()
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "error": "Timeout (>1800s)",
                        "messages": [],
                        "prediction": "[Failed]",
                        "multimodal": is_multimodal,
                        "image_count": image_count
                    }
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                except Exception as exc:
                    question = extract_question_from_item(task_info["item"])
                    is_multimodal = task_info.get("multimodal", False)
                    image_count = task_info.get("image_count", 0)
                    
                    print(f'Task for question "{question}" (Rollout {rollout_idx}) [Multimodal: {is_multimodal}, Images: {image_count}] generated an exception: {exc}')
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "error": f"Future resolution failed: {exc}",
                        "messages": [],
                        "prediction": "[Failed]",
                        "multimodal": is_multimodal,
                        "image_count": image_count
                    }
                    print("===============================")
                    print(error_result)
                    print("===============================")
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

        print("\nAll tasks completed!")

    print(f"\nAll {roll_out_count} rollouts completed!")