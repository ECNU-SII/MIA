# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import copy
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
import requests
import re
from openai import OpenAI

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

openai_api_key = "EMPTY"
openai_api_base = os.getenv("JUDGE_URL")
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model_name = "qwen"
def compute_acc_score(solution_str: str, ground_truth: str, question="") -> float:
    # Initialize tracking variables
    answer_text = ""
    predict_no_think = (
        solution_str.split("</think>")[-1].strip() if "</think>" in solution_str else solution_str.strip()
    )
    # Strategy 1: Try to extract from <answer> tags
    answer_match = re.findall(r"<answer>(.*?)</answer>", predict_no_think, re.DOTALL)
    if answer_match:
        answer_text = answer_match[-1].strip()
    else:
        # Strategy 2: Fallback to content after tool responses
        tool_response_match = re.search(
            r"</tool_response>\s*assistant\s*\n(.*?)$", predict_no_think, re.DOTALL | re.MULTILINE
        )
        if tool_response_match:
            answer_text = tool_response_match.group(1).strip()
        else:
            # Strategy 3: Fallback to content after </think> tag
            if "</think>" in solution_str:
                remaining_content = predict_no_think
                remaining_content = re.sub(r"<tool_call>.*?</tool_call>", "", remaining_content, flags=re.DOTALL)
                remaining_content = re.sub(r"<tool_response>.*?</tool_response>", "", remaining_content, flags=re.DOTALL)
                remaining_content = re.sub(r"\b(user|assistant)\b", "", remaining_content)
                answer_text = remaining_content.strip()
            else:
                # Strategy 4: Use the entire solution as a last resort
                answer_text = solution_str.strip()

    # Clean up answer text
    answer_text = answer_text.strip()
    if not answer_text:
        answer_text = solution_str.strip()

    def normalize_text(text):
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text.strip())

    normalized_answer = normalize_text(answer_text)
    normalized_gt = normalize_text(ground_truth)
    question_text = question

    if not client or not model_name:
        logger.warning("Reward function client not initialized or model name not found.")
        return 0.0

    judge_prompt = """
    Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
    First, I will give examples of each grade, and then you will grade a new example.

    The following are examples of CORRECT predicted answers.
    ‘‘‘
    Question: What are the names of Barack Obama’s children?
    Gold target: Malia Obama and Sasha Obama
    Predicted answer 1: sasha and malia obama
    Predicted answer 2: most people would say Malia and Sasha, but I’m not sure and would have to double check
    Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
    ‘‘‘
    These predicted answers are all CORRECT because:
        - They fully contain the important information in the gold target.
        - They do not contain any information that contradicts the gold target.
        - Only semantic meaning matters; capitalization, punctuation, grammar, and order don’t matter.
        - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.

    The following are examples of INCORRECT predicted answers.
    ‘‘‘
    Question: What are the names of Barack Obama’s children?
    Gold target: Malia and Sasha
    Predicted answer 1: Malia.
    Predicted answer 2: Malia, Sasha, and Susan.
    Predicted answer 3: Barack Obama does not have any children.
    Predicted answer 4: I think it’s either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
    Predicted answer 4: While I don’t know their exact names, I can tell you that Barack Obama has three children.
    Predicted answer 5: It’s possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
    Predicted answer 6: It may be the case that Obama’s child is named James. However, it’s recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
    ‘‘‘
    These predicted answers are all INCORRECT because:
        - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i’m not sure, i think") are also considered incorrect.

    The following are examples of NOT_ATTEMPTED predicted answers.
    ‘‘‘
    Question: What are the names of Barack Obama’s children?
    Gold target: Malia and Sasha
    Predicted answer 1: I don’t know.
    Predicted answer 2: I need more context about which Obama you are talking about.
    Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
    Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I’m not sure about the other one.
    ‘‘‘
    These predicted answers are all NOT_ATTEMPTED because:
        - The important information in the gold target is not included in the answer.
        - No statements in the answer contradict the gold target.
        
    Also note the following things:
    - The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
        - For example, consider the question "What episode did Derek and Meredith get legally married in Grey’s Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
    - Do not punish predicted answers if they omit information that would be clearly inferred from the question.
        - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include " California".
        - Consider the question "What award did A pretrainer’s guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL ’24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - Do not give credit for an answer if it contains any internal inconsistency.
        - For example, consider the question: "How many NBA players have scored 60 or more points in a regular season game since 2024?" with the gold answer "8". A response is INCORRECT if it states "8 players" but lists 7 or 9, or if it initially says "8 players" but later contradicts this by concluding 7 or 9.

    Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don’t apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
    ‘‘‘
    Question: {question}
    Gold target: {correct_answer}
    Predicted answer: {response}
    ‘‘‘
    Grade the predicted answer of this new question as one of:
    A: CORRECT
    B: INCORRECT
    C: NOT_ATTEMPTED

    Just return the letters "A", "B", or "C", with no text around it.
    """.strip()
    user_prompt = judge_prompt.format(question=question_text, correct_answer=normalized_gt, response=normalized_answer)
    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            temperature=0.1,  # Lower temperature for more deterministic judgement
        )
        response = chat_response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f" [WARNING] Chat completion request failed: {e}")
        print("Judger Connect Error")
        pt = normalized_answer.lower().strip()
        gt = normalized_gt.lower().strip()
        if gt in pt:
            response = "A"
        else:
            response = "B"
    response = response.replace("\\n", "\n").replace("\\r", "\r")
    # 移除 <think>...</think>、孤立的 <think>、孤立的 </think>
    cleaned = re.sub(r"<think>.*?</think>|<think>|</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    # 去掉多余空行和空格
    cleaned = cleaned.strip()
    max_answer_word_count = 100
    # answer_word_count = len(normalized_answer.split()) if normalized_answer else 0
    answer_word_count = len(normalized_answer)
    if cleaned == "A":
        accuracy_score = 1.0
    elif cleaned == "B":
        accuracy_score = 0.0
    elif cleaned == "C":
        accuracy_score = 0.0
    else:
        accuracy_score = 0.0
    if answer_word_count > max_answer_word_count:
        accuracy_score = 0.0
    return accuracy_score




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


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0
        self.workflow = ""
        self.has_web_image = False
        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)

        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        use_replan = False
        mydata = kwargs["extra_info"]
        ground_truth = mydata["ground_truth"]
        question = mydata["question"]
        
        score = 0.0
        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                agent_data.workflow += f"### Round {agent_data.assistant_turns + 1}:\n"
                state = await self._handle_generating_state(agent_data, sampling_params)
                agent_data.assistant_turns += 1
                if state == AgentState.TERMINATED:
                    score = compute_acc_score(agent_data.workflow, ground_truth, question)
                    if score < 0.5 and (not use_replan): 
                        data = {"data_id": mydata["data_id"], "question": question, "plan": mydata["plan"], "trace": agent_data.workflow, "mem_context1": mydata["mem_context1"], "mem_context2": mydata["mem_context2"]}
                        url = os.getenv("REPLAN_URL")
                        result = await get_user_response_from_url(url, data)
                        try:
                            replan = result["replan"]
                            content = result["content"].replace("<image>", "")
                        except:
                            replan = False
                            content = ""
                        if replan:
                            try:
                                add_messages = [{"role": "user", "content": f"I have a revised plan for you to follow:\n{content}\nPlease continue with this updated guidance:\n"}]
                                if self.processor is not None:
                                    raw_user_response = await self.loop.run_in_executor(
                                        None,
                                        lambda: self.processor.apply_chat_template(
                                            add_messages,
                                            add_generation_prompt=True,
                                            tokenize=False,
                                            **self.apply_chat_template_kwargs,
                                        ),
                                    )
                                    model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
                                    response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
                                else:
                                    response_ids = await self.loop.run_in_executor(
                                        None,
                                        lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
                                    )
                                response_ids = response_ids[len(self.system_prompt) :]
                                model_max_len = int(os.getenv("MODEL_MAX_LEN", 32768)) - 10
                                if len(agent_data.prompt_ids) + len(response_ids) >= model_max_len or len(response_ids) > 500:
                                    state =  AgentState.TERMINATED
                                    break
                                agent_data.prompt_ids += response_ids
                                agent_data.response_mask += [0] * len(response_ids)
                                if agent_data.response_logprobs:
                                    agent_data.response_logprobs += [0.0] * len(response_ids)
                                state = AgentState.GENERATING
                                use_replan = True
                            except:
                                pass
                    elif score > 0.5 and (not use_replan):
                        score += 0.2
    
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
                agent_data.user_turns += 1
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        kwargs["extra_info"]["acc_score"] = score
        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores})
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Check termination conditions
        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)
            agent_data.workflow += f"#### Agent Reasoning and Tool Call:\n{str(assistant_message)}\n"
        else:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids)
            )
            agent_data.workflow += f"#### Agent Reasoning and Tool Call:\n{str(assistant_message)}\n"
        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute
        tasks = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            try:
                tool_name = tool_call.name
                if tool_name == "web_image_to_image_search":
                    if agent_data.has_web_image:
                        tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, pass_tool_call=True))
                    else:
                        tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs))
                    agent_data.has_web_image = True
                else:
                    tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs))
            except:
                tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs))
        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Handle responses for interaction if needed
        if self.interaction_config_file:
            for response in responses:
                if response.text:
                    agent_data.messages.append({"role": "tool", "content": response.text})
                    if response.sub_workflow:
                        agent_data.workflow += response.sub_workflow
        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                content = []
                if tool_response.image:
                    images_to_process = tool_response.image if isinstance(tool_response.image, list) else [tool_response.image]
                    for img in images_to_process:
                        if img is not None:
                            content.append({"type": "image"})

                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                    if tool_response.sub_workflow:
                        agent_data.workflow += tool_response.sub_workflow

                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}
                if tool_response.sub_workflow:
                    agent_data.workflow += tool_response.sub_workflow

            add_messages.append(message)
            agent_data.messages.extend(add_messages)

            # Handle image data
            if tool_response.image:
                if agent_data.image_data is None:
                    agent_data.image_data = []
                elif not isinstance(agent_data.image_data, list):
                    agent_data.image_data = [agent_data.image_data]

                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            agent_data.image_data.append(img)
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        agent_data.image_data.append(tool_response.image)
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

        # Update prompt with tool responses
        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Use only the new images from this turn for processing tool responses
            current_images = new_images_this_turn if new_images_this_turn else None  # Using local variable
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt) :]
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        if self.processor is not None:
            raw_user_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt) :]

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], pass_tool_call=False) -> ToolResponse:
        """Call tool and return tool response."""
        if pass_tool_call:
            text_content = "Error: Skipping duplicate calls to the web_image_to_image_search tool.\n"
            sub_workflow = f"#### Agent Reasoning and Tool Call:\n{text_content}\n"
            tool_response_kwargs = {"text": text_content, "sub_workflow": sub_workflow}
            return ToolResponse(**tool_response_kwargs)
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            
            tool_execution_response, _, _ = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return ToolResponse(
                text=f"Error when executing tool: {e}",
                sub_workflow="",
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)
        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]
        # Create ToolResponse from tool execution result
        text_content = truncate_by_whitespace_words(tool_response_text)
        sub_workflow = f"#### Agent Reasoning and Tool Call:\n{text_content}\n"
        
        tool_response_kwargs = {"text": tool_response_text, "sub_workflow": sub_workflow}
        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value
        return ToolResponse(**tool_response_kwargs)

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map


async def get_user_response_from_url(url, data) -> str:
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: requests.post(url, json=data)
        )
        response.raise_for_status()
        result = response.json()
        return result  
    except Exception as e:
        logger.error(f"Error getting user response from URL: {e}")
        return {"replan": False, "content": ""}
