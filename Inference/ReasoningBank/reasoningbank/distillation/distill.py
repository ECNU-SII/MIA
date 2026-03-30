"""Distillation functions for processing agent trajectories."""

from typing import List, Dict, Any
import json

# A placeholder for a generic LLM interface.
# In a real implementation, this would be a more specific type,
# for example, a LangChain BaseLanguageModel.
LLM = Any


def judge_trajectory(trajectory: str, query: str, llm) -> bool:
    """
    Judges whether a trajectory was successful or not using an LLM.

    This function sends a prompt to a language model to evaluate if the given
    trajectory successfully addresses the query. It is a simple binary
    classification (Success/Failure).

    Args:
        trajectory (str): The sequence of actions and observations from the agent.
        query (str): The initial task or question for the agent.
        llm (LLM): The language model to use for the judgment.

    Returns:
        bool: True if the trajectory is judged as successful, False otherwise.
    """
    prompt = f"""
    Given the following query and trajectory, determine if the trajectory successfully addresses the query.
    Respond with "Success" or "Failure".

    Query: {query}

    Trajectory:
    {trajectory}
    """
    
    
    # response = llm.invoke(prompt)
    
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    response_obj = llm.chat.completions.create(
        model="qwen",
        temperature=0,
        messages=messages,
        max_tokens=8192,
        timeout=100.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    response = response_obj.choices[0].message.content.strip()
    return "success" in response.lower()


# def distill_trajectory(
#     trajectory: str, query: str, llm, is_success: bool
# ) -> List[Dict]:
#     """
#     Distills a raw trajectory into a list of structured memory items.

#     Based on whether the trajectory was successful, this function prompts the
#     language model to extract key reasoning steps, strategies, or lessons learned.
#     The output is expected to be a JSON string representing a list of memory items.

#     Args:
#         trajectory (str): The agent's trajectory.
#         query (str): The initial query.
#         llm (LLM): The language model for distillation.
#         is_success (bool): Whether the trajectory was successful.

#     Returns:
#         List[Dict]: A list of distilled memory items, each with a title,
#                     description, and content. Returns an empty list if
#                     the LLM response cannot be parsed.
#     """
#     if is_success:
#         prompt = f"""
#         The following trajectory was successful in addressing the query.
#         Distill the key reasoning steps and strategies into a few memory items.
#         Each memory item should have a title, a short description, and content.
#         Format the output as a JSON string representing a list of dictionaries. For example:
#         [
#             {{
#                 "title": "Example Title",
#                 "description": "A short description.",
#                 "content": "The detailed reasoning steps."
#             }}
#         ]

#         Query: {query}

#         Trajectory:
#         {trajectory}
#         """
#     else:
#         prompt = f"""
#         The following trajectory failed to address the query.
#         Analyze the failure and distill the lessons learned into a few memory items.
#         Each memory item should have a title, a short description, and content describing the pitfall and how to avoid it.
#         Format the output as a JSON string representing a list of dictionaries. For example:
#         [
#             {{
#                 "title": "Example Pitfall",
#                 "description": "A short description of the error.",
#                 "content": "A detailed explanation of the mistake and how to avoid it in the future."
#             }}
#         ]

#         Query: {query}

#         Trajectory:
#         {trajectory}
#         """

#     response = llm.invoke(prompt)

#     # Use json.loads for safe parsing of the LLM's JSON output.
#     try:
#         distilled_memories = json.loads(response)
#         if isinstance(distilled_memories, list):
#             return distilled_memories
#     except json.JSONDecodeError:
#         # If parsing fails, return an empty list.
#         return []

#     return []



import json
from typing import List, Dict

def distill_trajectory(
    trajectory: str,
    query: str,
    llm,
    is_success: bool
) -> List[Dict]:
    # Define the JSON schema for a single memory item
    # Define the top-level schema: a list of memory items
    if is_success:
        memory_item_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string", "description": "A short description."},
                "content": {"type": "string", "description": "The detailed reasoning steps."}
            },
            "required": ["title", "description", "content"],
            "additionalProperties": False
        }
        
        instruction = (
            "The following trajectory was successful in addressing the query. "
            "Distill the key reasoning steps and strategies into a few memory items."
        )
    else:
        memory_item_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string", "description": "A short description of the error."},
                "content": {"type": "string", "description": "A detailed explanation of the mistake and how to avoid it in the future."}
            },
            "required": ["title", "description", "content"],
            "additionalProperties": False
        }
        
        instruction = (
            "The following trajectory failed to address the query. "
            "Analyze the failure and distill lessons learned into memory items that describe pitfalls and how to avoid them."
        )

    
    response_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "memories",
            "schema": memory_item_schema,
            "strict": True
        }
    }
    prompt = f"""{instruction}

Query: {query}

Trajectory:
{trajectory}
"""

    try:
        completion = llm.chat.completions.create(
            model="qwen",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8192,
            response_format=response_schema,
            timeout=120.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        parsed = json.loads(completion.choices[0].message.content.strip())
        return parsed

    except Exception as e:
        # In production, you might log this error
        print(f"Error during trajectory distillation: {e}")
        return []