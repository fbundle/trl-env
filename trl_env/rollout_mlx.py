from __future__ import annotations

from typing import Any, Callable

import mlx.nn as nn
import mlx_lm
import mlx_lm.sample_utils

from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from trl_env.engine import Engine

from trl.trainer.grpo_trainer import RolloutFunc, GRPOTrainer, RewardFunc

from trl_env.environment import Env
from trl_env.processor import Processor
from trl_env.rollout import batch_rollout


def collapse_eos_token_id(completion_ids: list[int], eos_token_id: int) -> list[int]:
    try:
        index = completion_ids.index(eos_token_id)
    except ValueError:
        return completion_ids
    return completion_ids[: index + 1]

class MlxEngine(Engine):
    def __init__(
        self,
        model_path: str,
        max_completion_length: int,
        eos_tokens: list[str],
    ) -> None:
        model, tokenizer, _ = mlx_lm.load(  # type: ignore
            path_or_hf_repo=model_path,
            return_config=True,
        )
        self.model: nn.Module = model
        self.tokenizer: TokenizerWrapper = tokenizer

        # 
        self.max_completion_length = max_completion_length
        for eos_token in eos_tokens:
            self.tokenizer.add_eos_token(eos_token)


    def update_weights(self, model: PreTrainedModel):
        print("[WARNING] update_weights is not current implemented")
        return

    def tokenizer_encode(self, input_text: str) -> list[int]:
        return self.tokenizer._tokenizer(input_text).input_ids

    def tokenizer_decode(self, completion_ids: list[int]) -> str:
        output_text = self.tokenizer._tokenizer.decode(completion_ids)
        assert isinstance(output_text, str)
        return output_text
    
    def model_batch_generate(self, input_ids_list: list[list[int]]) -> tuple[list[list[int]], list[list[float]]]:
        completion_ids_list: list[list[int]] = []
        log_probs_list: list[list[float]] = []

        for input_ids in input_ids_list:
            completion_ids: list[int] = []
            log_probs: list[float] = []
            response_generator = mlx_lm.stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=input_ids,
                max_tokens=self.max_completion_length,
            )
            for r in response_generator:
                completion_ids.append(r.token)
                log_probs.append(r.logprobs[r.token].item())
            
            completion_ids_list.append(completion_ids)
            log_probs_list.append(log_probs)





        pass


def make_rollout_func(
    engine: MlxEngine, processor: Processor, env_factory: Callable[[], Env],
    system_prompt: str, max_conversation_length: int,
) -> RolloutFunc:
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, Any]:
        try:
            engine.update_weights(trainer.model)
            state_list = batch_rollout(
                engine=engine, processor=processor, env_factory=env_factory,
                system_prompt=system_prompt, max_conversation_length=max_conversation_length,
                seed_list=prompts,
            )
        finally:
            ...
        
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