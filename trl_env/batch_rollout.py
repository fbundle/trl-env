from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from torch import Tensor
from jaxtyping import Float
import torch

from .environment import Action, Env, Seed
from .model import Model
from .processor import Processor

@dataclass
class RolloutState:
    initial_length: int
    conversation: list[int]
    env_mask: list[int]
    logprobs: list[float]
    total_step_reward: float

    def append_completion(
        self,
        completion_ids: list[int],
        logprobs: list[float] | None,
    ):
        if logprobs is None:
            env_mask = [0] * len(completion_ids)
            logprobs = [0.0] * len(completion_ids)
        else:
            env_mask = [1] * len(completion_ids)

        self.conversation.extend(completion_ids)
        self.env_mask.extend(env_mask)
        self.logprobs.extend(logprobs)

def init_rollout_state(initial_prompt_ids: list[int]) -> RolloutState:
    return RolloutState(
        initial_length=len(initial_prompt_ids),
        conversation=initial_prompt_ids,
        env_mask=[],
        logprobs=[],
        total_step_reward=0,
    )


def batch_rollout(
    model: Model, processor: Processor, env_factory: Callable[[], Env],
    system_prompt: str, max_conversation_length: int,
    seed_list: list[Seed],
    log: Callable[[str], None] | None = None, 
) -> list[RolloutState]:
    def LOG_LINE(s: str):
        if log is not None:
            log(s + "\n")

    LOG_LINE("system>\t" + system_prompt)
    system_prompt_ids = model.tokenizer_encode(processor.init_system_input(system_prompt))
    
    env_list: list[Env] = []
    state_list: list[RolloutState] = []
    for i, seed in enumerate(seed_list):
        env = env_factory()
        initial_delta = env.reset(seed)
        LOG_LINE(f"user_{i}>\t" + initial_delta)

        # assuming tokenizer is additive
        # tok(a ++ b) = tok(a) ++ tok(b)
        initial_prompt_ids = system_prompt_ids + model.tokenizer_encode(processor.append_user_input(initial_delta))

        state = init_rollout_state(initial_prompt_ids=initial_prompt_ids)

        env_list.append(env)
        state_list.append(state)

    # while some environment is still not termininate
    while sum([env.alive for env in env_list]) > 0:
        # MODEL BATCH GENERATE
        # TODO - consider giving `past_key_values`
        # i.e. give the model the last hidden states so that it won't recalculate everything from the beginning
        completion_ids_list, logprobs_list = model.model_batch_generate([state.conversation for state in state_list])

        for env, state, completion_ids, logprobs in zip(env_list, state_list, completion_ids_list, logprobs_list):
            # IF NOT ALIVE
            if not env.alive:
                continue
            # APPEND AGENT COMPLETION
            state.append_completion(
                completion_ids=completion_ids,
                logprobs=logprobs,
            )

            # PARSE ACTION
            completion_text = model.tokenizer_decode(completion_ids)
            reason, action = processor.parse_agent_output(completion_text)
            LOG_LINE(f"agent_{i}>\t" + action)
        
            # INTERACT WITH ENVIRONMENT
            delta = env.step(action)
            LOG_LINE(f"user_{i}>\t" + delta)

            # UPDATE REWARD
            state.total_step_reward += env.last_step_reward

            # IF NOT ALIVE
            if not env.alive:
                continue
            
            # APPEND ENVIRONMENT COMPLETION
            # assuming tokenizer is additive
            # tok(a ++ b) = tok(a) ++ tok(b)
            delta_ids = model.tokenizer_encode(processor.append_user_input(delta))
            state.append_completion(
                completion_ids=delta_ids,
                logprobs=None,
            )

            # TERMINATE ENV
            if len(state.conversation) >= max_conversation_length:
                env.alive = False
                continue
            
            LOG_LINE(f"log_{i}>\t" + f"conversation length {len(state.conversation)}")
    
    return state_list


# for GRPO
from typing import Any
from trl.trainer.grpo_trainer import RolloutFunc, GRPOTrainer, RewardFunc

def make_rollout_func(
    model: Model, processor: Processor, env_factory: Callable[[], Env],
    system_prompt: str, max_conversation_length: int,
) -> tuple[RolloutFunc, RewardFunc]:
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, Any]:
        state_list = batch_rollout(
            model=model, processor=processor, env_factory=env_factory,
            system_prompt=system_prompt, max_conversation_length=max_conversation_length,
            seed_list=prompts,
        )
        return {
            "prompt_ids": [state.conversation[:state.initial_length] for state in state_list],
            "completion_ids": [state.conversation[state.initial_length:] for state in state_list],
            "env_mask": [state.env_mask for state in state_list],
            "logprobs": [state.logprobs for state in state_list],
            "total_step_reward": [state.total_step_reward for state in state_list],
        }

    def reward_func(prompts: list[str], completions: list[str], total_step_reward: list[float], **kwargs) -> list[float]:
        return total_step_reward
    
    return rollout_func, reward_func # type: ignore
