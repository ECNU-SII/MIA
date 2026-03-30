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
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4
import requests
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def extract_after_think(text):
    last_think_index_close = text.rfind("</think>")
    last_think_index_open = text.rfind("<think>")
    last_think_index = max(last_think_index_open, last_think_index_close)
    if last_think_index == -1:
        return text.replace("<|im_end|>", "").strip()
    start_index = last_think_index + len("</think>")
    result = text[start_index:]
    result = result.replace("<|im_end|>", "").strip()
    return result

class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
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
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
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
        self.current_round = 0  # Track current round of conversation
        self.useful_messages = []
        self.history_messages = []

@register("multi_turn_agent")
class MultiTurnAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level MultiTurnAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
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
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        mydata = kwargs["extra_info"]
        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state, response = await self._handle_generating_state(agent_data, sampling_params)
                agent_data.assistant_turns += 1
                if agent_data.current_round == 1:
                    mydata["plan"] = extract_after_think(response)
                elif agent_data.current_round == 3:
                    mydata["replan"] = extract_after_think(response)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data, mydata)
                agent_data.user_turns += 1
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED
        kwargs["extra_info"]["messages"] = agent_data.useful_messages
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
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the generating state: generate model response."""
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

        assistant_message = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.response_ids)
        )
        # Check termination conditions
        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED, assistant_message

        agent_data.useful_messages.append(assistant_message)
        agent_data.current_round += 1
        if agent_data.current_round == 2:
            # Check if agent response contains "yes"
            if contains_yes(assistant_message):
                # Continue to next round
                return AgentState.INTERACTING, assistant_message
            else:
                # Terminate immediately if no "yes" in response
                return AgentState.TERMINATED, assistant_message
        elif agent_data.current_round >= 4:
            # After 4 rounds, terminate
            return AgentState.TERMINATED, assistant_message
        else:
            # For rounds 1, 3, continue interaction
            return AgentState.INTERACTING, assistant_message


    async def _handle_interacting_state(self, agent_data: AgentData, mydata) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        # Get user response from external API
        if agent_data.current_round == 1:
            data = {"data_id": mydata["data_id"], "question": mydata["question"], "plan": mydata["plan"]}
            url = os.getenv("PLAN_URL")
            result = await get_user_response_from_url(url, data)
            try:
                prompt = result["prompt"]
                trace = result["trace"]
                history_messages = result["history_messages"]
            except:
                prompt = "error"
                trace = "error"
                history_messages = []
            
            agent_data.useful_messages.append(trace)
            agent_data.history_messages = history_messages
            user_response = prompt
        elif agent_data.current_round == 2:
            user_response = mydata["replan_prompt"]
            agent_data.useful_messages.append(user_response)
        else:
            data = {"data_id": mydata["data_id"], "question": mydata["question"], "plan": mydata["plan"], "messages": agent_data.history_messages, "replan": mydata["replan"]}
            url = os.getenv("REPLAN_URL")
            result = await get_user_response_from_url(url, data)
            try:
                prompt = result["prompt"]
                trace = result["trace"]
            except:
                prompt = "error"
                trace = "error"
            agent_data.useful_messages.append(trace)
            user_response = prompt
        add_messages: list[dict[str, Any]] = [{"role": "user", "content": user_response}]
        if agent_data.current_round < 3:
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
            model_max_len = int(os.getenv("MODEL_MAX_LEN", 32786)) - 10
            
            print(len(agent_data.prompt_ids) , len(response_ids), model_max_len - len(agent_data.prompt_ids) - len(response_ids))
            if len(agent_data.prompt_ids) + len(response_ids) >= model_max_len:
                return AgentState.TERMINATED
            
            # Update prompt_ids and response_mask
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)
            
        
        # Check termination condition based on round
        if agent_data.current_round >= 3:
            # After 4 rounds, terminate
            return AgentState.TERMINATED
        else:
            # Continue to next round
            return AgentState.GENERATING

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


def contains_yes(message: str) -> bool:
    def extract_after_think(text):
        last_think_index_close = text.rfind("</think>")
        last_think_index_open = text.rfind("<think>")
        last_think_index = max(last_think_index_open, last_think_index_close)
        if last_think_index == -1:
            return text.replace("<|im_end|>", "").strip()
        start_index = last_think_index + len("</think>")
        result = text[start_index:]
        result = result.replace("<|im_end|>", "").strip()
        return result
    text = extract_after_think(message)
    if not text:
        return False
    return "yes" in text.lower()


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
        return "Default user response due to error"
