from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal, Protocol, Sequence

from .environment import Env, Seed
from .decoder import RolloutDecoder, RolloutDecoderFactory
from .processor import Processor
from .tokenizer import Tokenizer

from trl.trainer.grpo_trainer import RolloutFunc, GRPOTrainer, RewardFunc

class StreamDecoder[T](Protocol):
    def generate(self, input_ids: list[int]) -> Iterator[tuple[int, T]]: ...


@dataclass
class State[T]:
    initial_length: int
    conversation: list[int]
    env_mask: list[int]
    completion_data: list[T | None]
    reward: float | None

    def append_completion(
        self,
        completion_token_list: Sequence[int],
        completion_data_list: Sequence[T | None] | None,
    ) -> State:
        if completion_data_list is None:
            env_mask = [0] * len(completion_token_list)
            completion_data_list = [None] * len(completion_token_list)
        else:
            env_mask = [1] * len(completion_token_list)

        self.conversation.extend(completion_token_list)
        self.env_mask.extend(env_mask)
        self.completion_data.extend(completion_data_list)
        return self

def init_rollout_state(initial_prompt_ids: list[int]) -> State:
    return State(
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
) -> State:
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

from tqdm import tqdm

def make_rollout_func(
    processor: Processor, tokenizer: Tokenizer,
    decoder_factory: RolloutDecoderFactory,
    env_factory: Callable[[], Env],    
    system_prompt: str, max_conversation_length: int,
) -> RolloutFunc:
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, Any]:
        decoder = decoder_factory.make_decoder(trainer)
        env = env_factory()

        state_list = []
        for prompt in tqdm(prompts, desc="rolling_out ..."):
            # TODO batch this - need to make decoder in batch as well
            state = rollout(
                processor=processor, tokenizer=tokenizer,
                decoder=decoder, env=env,
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
