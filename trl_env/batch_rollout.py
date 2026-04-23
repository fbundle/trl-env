from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
from typing import Callable, Protocol

from .environment import Env, Seed
from .model import Model
from .processor import Processor

@dataclass
class RolloutState:
    initial_length: int
    conversation: list[int]
    env_mask: list[int]
    logprobs: list[float]
    reward: float | None

    def append_completion(
        self,
        completion_ids: list[int],
        logprobs: list[float] | None,
    ) -> RolloutState:
        if logprobs is None:
            env_mask = [0] * len(completion_ids)
            logprobs = [0.0] * len(completion_ids)
        else:
            env_mask = [1] * len(completion_ids)

        self.conversation.extend(completion_ids)
        self.env_mask.extend(env_mask)
        self.logprobs.extend(logprobs)
        return self

def init_rollout_state(initial_prompt_ids: list[int]) -> RolloutState:
    return RolloutState(
        initial_length=len(initial_prompt_ids),
        conversation=initial_prompt_ids,
        env_mask=[],
        logprobs=[],
        reward=None,
    )

def batch_rollout(
    model: Model, processor: Processor, env_factory: Callable[[], Env],
    system_prompt: str, max_conversation_length: int,
    seed_list: list[Seed], 
    conversation_logger: Callable[[int, str, str], None] | None = None,
) -> list[RolloutState]:
    _log_lock = threading.Lock()
    def LOG(i: int, role: str, content: str):
        if conversation_logger is not None:
            with _log_lock:
                conversation_logger(i, role, content)

    system_prompt_ids = model.tokenizer_encode(processor.init_system_input(system_prompt))
    
    env_list: list[Env] = []
    state_list: list[RolloutState] = []
    for i, seed in enumerate(seed_list):
        env, initial_delta = env_factory().reset(seed)

        LOG(i, "system", system_prompt)
        LOG(i, "user", initial_delta)

        # assuming tokenizer is additive
        # tok(a ++ b) = tok(a) ++ tok(b)
        initial_prompt_ids = system_prompt_ids + model.tokenizer_encode(processor.append_user_input(initial_delta))

        state = init_rollout_state(initial_prompt_ids=initial_prompt_ids)

        env_list.append(env)
        state_list.append(state)

    with ThreadPoolExecutor(max_workers=len(env_list)) as executor:
        # while some environment is still not termininate
        while sum([env.alive for env in env_list]) > 0:
            # MODEL BATCH GENERATE
            # NOTE - consider giving `past_key_values`
            # i.e. give the model the last hidden states so that it won't recalculate everything from the beginning
            completion_ids_list, logprobs_list = model.model_batch_generate([state.conversation for state in state_list])

            # PROCESS GENERATE
            def process_generate(i: int, env: Env, state: RolloutState, completion_ids: list[int], logprobs: list[float]) -> tuple[int, Env, RolloutState]:
                # precheck env.alive
                if not env.alive:
                    return i, env, state
                # append agent completion
                state = state.append_completion(
                    completion_ids=completion_ids,
                    logprobs=logprobs,
                )
                # parse (reason, action)
                completion_text = model.tokenizer_decode(completion_ids)
                LOG(i, "assistant", completion_text)
                reason, action = processor.parse_agent_output(completion_text)
                # interact with environment
                env, delta = env.step(action)
                LOG(i, "user", delta)
                # save reward
                state.reward = env.reward
                # postcheck env.alive
                if not env.alive:
                    LOG(i, "log", "env terminated")
                    return i, env, state
                # append environment completion
                # assuming tokenizer is additive
                # tok(a ++ b) = tok(a) ++ tok(b)
                delta_ids = model.tokenizer_encode(processor.append_user_input(delta))
                state = state.append_completion(
                    completion_ids=delta_ids,
                    logprobs=None,
                )
                # terminate env if conversation is long
                if len(state.conversation) >= max_conversation_length:
                    env.alive = False
                    LOG(i, "log", "env terminated due to long conversation")
                
                return i, env, state


            # BATCH PROCESS GENERATE
            # NOTE - we wish to use multiprocess here, but it might interfere with torch/accelerate
            output_iter = executor.map(lambda xs: process_generate(*xs), zip(
                range(len(env_list)),
                env_list, state_list,
                completion_ids_list, logprobs_list,
            ))
            for i, env, state in output_iter:
                env_list[i] = env
                state_list[i] = state
        
        return state_list


# for GRPO
from typing import Any
from trl.trainer.grpo_trainer import RolloutFunc, GRPOTrainer, RewardFunc
from .model import TransformerModel
import torch


def make_rollout_func(
    model: TransformerModel, processor: Processor, env_factory: Callable[[], Env],
    system_prompt: str, max_conversation_length: int,
) -> RolloutFunc:
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, Any]:
        try:
            # NOTE - only rollout on eval mode
            model.model.eval()
            with torch.no_grad():
                state_list = batch_rollout(
                    model=model, processor=processor, env_factory=env_factory,
                    system_prompt=system_prompt, max_conversation_length=max_conversation_length,
                    seed_list=prompts,
                )
        finally:
            model.model.train()
        
        return {
            "prompt_ids": [state.conversation[:state.initial_length] for state in state_list],
            "completion_ids": [state.conversation[state.initial_length:] for state in state_list],
            "env_mask": [state.env_mask for state in state_list],
            "logprobs": [state.logprobs for state in state_list],
            "reward": [state.reward for state in state_list],
        }

    return rollout_func

def make_reward_func() -> RewardFunc:
    def reward_func(prompts: list[str], completions: list[str], reward: list[float], **kwargs) -> list[float]:
            return reward
    return reward_func # type: ignore