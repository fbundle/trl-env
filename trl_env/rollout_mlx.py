


from typing import Union, Any, Callable

import mlx.nn as nn
import mlx_lm
import mlx.core as mx
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from trl_env.engine import Engine

from trl.trainer.grpo_trainer import RolloutFunc, GRPOTrainer, RewardFunc

from trl_env.environment import Env
from trl_env.processor import Processor
from trl_env.rollout import batch_rollout


class MlxEngine(Engine):
    def __init__(self, model_path: str, transformer_model: PreTrainedModel | None) -> None:
        model, _, _ = mlx_lm.load(  # type: ignore
            path_or_hf_repo=model_path,
            return_config=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.transformer_model = transformer_model
        self.model: nn.Module = model
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
    
    def update_weights(self):
        # TODO - get self.transformer_model weights
        file_or_weights: Union[str, list[tuple[str, mx.array]]] = None # type: ignore
        self.model.load_weights(file_or_weights)

    def tokenizer_encode(self, input_text: str) -> list[int]:
        return self.tokenizer(input_text).input_ids

    def tokenizer_decode(self, completion_ids: list[int]) -> str:
        output_text = self.tokenizer.decode(completion_ids)
        assert isinstance(output_text, str)
        return output_text
    
    def model_batch_generate(self, input_ids_list: list[list[int]]) -> tuple[list[list[int]], list[list[float]]]:
        # TODO
        ...


def make_rollout_func(
    engine: MlxEngine, processor: Processor, env_factory: Callable[[], Env],
    system_prompt: str, max_conversation_length: int,
) -> RolloutFunc:
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, Any]:
        try:
            engine.update_weights()
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