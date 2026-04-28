from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

from transformers import PreTrainedModel

from .environment import Env, Seed
from .decoder import RolloutDecoder
from .processor import Processor
from .tokenizer import Tokenizer

from trl.trainer.grpo_trainer import RolloutFunc, GRPOTrainer, RewardFunc



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

def rollout(
    processor: Processor, tokenizer: Tokenizer,
    decoder: RolloutDecoder, env: Env,
    system_prompt: str, max_conversation_length: int,
    seed: Seed, 
    conversation_logger: Callable[[str, str], None] | None = None,
) -> RolloutState:
    def LOG(role: str, content: str):
        if conversation_logger is not None:
            conversation_logger(role, content)

    env, initial_delta = env.reset(seed)

    LOG("system", system_prompt)
    LOG("user", initial_delta)
    # assuming tokenizer is additive
    # tok(a ++ b) = tok(a) ++ tok(b)
    system_prompt_ids = tokenizer.encode(processor.init_system_input(system_prompt))
    initial_prompt_ids = system_prompt_ids + tokenizer.encode(processor.append_user_input(initial_delta))

    state = init_rollout_state(initial_prompt_ids=initial_prompt_ids)

    while True:
        # precheck env.alive
        if not env.alive:
            break
        # model generate
        completion_ids, logprobs = decoder.generate(state.conversation)
        # append agent completion
        state = state.append_completion(
            completion_ids=completion_ids,
            logprobs=logprobs,
        )
        # parse (reason, action)
        completion_text = tokenizer.decode(completion_ids)
        LOG("assistant", completion_text)
        reason, action = processor.parse_agent_output(completion_text)
        # interact with environment
        env, delta = env.step(action)
        LOG("user", delta)
        # save reward
        state.reward = env.reward
        # postcheck env.alive
        if not env.alive:
            LOG("log", "env terminated")
            break
        # append environment completion
        # assuming tokenizer is additive
        # tok(a ++ b) = tok(a) ++ tok(b)
        delta_ids = tokenizer.encode(processor.append_user_input(delta))
        state = state.append_completion(
            completion_ids=delta_ids,
            logprobs=None,
        )
        # terminate env if conversation is long
        if len(state.conversation) >= max_conversation_length:
            env.alive = False
            LOG("log", "env terminated due to long conversation")
            break
    return state


def make_rollout_func(
    processor: Processor, tokenizer: Tokenizer,
    make_decoder: Callable[[PreTrainedModel], RolloutDecoder],
    env_factory: Callable[[], Env],    
    system_prompt: str, max_conversation_length: int,
) -> RolloutFunc:
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, Any]:
        state_list = []
        for prompt in prompts:
            # TODO - currently we only do batch_size=1
            # use continuous batching - NOTE: cache can be merged/split
            state = rollout(
                processor=processor, tokenizer=tokenizer,
                decoder=make_decoder(trainer.model), # type: ignore
                env=env_factory(),
                system_prompt=system_prompt, max_conversation_length=max_conversation_length,
                seed=prompt,
            )
            state_list.append(state)

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