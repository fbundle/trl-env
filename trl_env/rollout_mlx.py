from __future__ import annotations

from typing import Any, Callable, Union
import copy

import mlx.nn as nn
import mlx_lm
import mlx.core as mx
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
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        model, _, _ = mlx_lm.load(  # type: ignore
            path_or_hf_repo=model_path,
            return_config=True,
        )
        self.model: nn.Module = model
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path)

        self.generation_kwargs: dict[str, Any] = {
            "max_new_tokens": 256,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if generation_kwargs is not None:
            self.generation_kwargs.update(generation_kwargs)

        for key in ["prompt", "model", "tokenizer", "prompt_cache", "max_tokens"]:
            if key in self.generation_kwargs:
                raise RuntimeError(f"generation_kwargs[{key}] must not be set")

        self._prefix_ids: list[int] | None = None
        self._prefix_cache: list[Any] | None = None
    
    def update_weights(self, model: PreTrainedModel):
        # NOTE: mlx_lm models are not torch models; updating weights from a
        # transformers.PreTrainedModel is not supported here.
        #
        # If the caller passes an MLX model instance, allow swapping it in.
        if isinstance(model, nn.Module):  # type: ignore[unreachable]
            self.model = model
            return
        return

    def tokenizer_encode(self, input_text: str) -> list[int]:
        return self.tokenizer(input_text).input_ids

    def tokenizer_decode(self, completion_ids: list[int]) -> str:
        output_text = self.tokenizer.decode(completion_ids)
        assert isinstance(output_text, str)
        return output_text
    
    def model_batch_generate(self, input_ids_list: list[list[int]]) -> tuple[list[list[int]], list[list[float]]]:
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        eos_token_id = int(self.generation_kwargs["eos_token_id"])
        max_new_tokens = int(self.generation_kwargs["max_new_tokens"])

        sampler = make_sampler(
            temp=float(self.generation_kwargs.get("temp", 0.0)),
            top_p=float(self.generation_kwargs.get("top_p", 1.0)),
            min_p=float(self.generation_kwargs.get("min_p", 0.0)),
            top_k=int(self.generation_kwargs.get("top_k", 0)),
        )

        completion_ids_list: list[list[int]] = []
        logprobs_list: list[list[float]] = []

        for input_ids in input_ids_list:
            if len(input_ids) == 0:
                completion_ids_list.append([])
                logprobs_list.append([])
                continue

            prompt_cache: Any | None = None
            prompt_ids = input_ids

            if (
                self._prefix_ids is not None
                and self._prefix_cache is not None
                and len(prompt_ids) >= len(self._prefix_ids)
                and prompt_ids[: len(self._prefix_ids)] == self._prefix_ids
            ):
                prompt_cache = copy.deepcopy(self._prefix_cache)
                # The cached prefix is built for prefix[:-1]. Re-feed prefix[-1]
                # as the first token to continue from.
                prefix_last = self._prefix_ids[-1]
                prompt_ids = [prefix_last] + prompt_ids[len(self._prefix_ids) :]

            if prompt_cache is None:
                prompt_cache = make_prompt_cache(self.model)

            prompt = mx.array(prompt_ids, dtype=mx.uint32)

            completion_ids: list[int] = []
            completion_logprobs: list[float] = []

            for tok, step_logprobs in generate_step(
                prompt,
                self.model,
                max_tokens=max_new_tokens,
                sampler=sampler,
                prompt_cache=prompt_cache,
            ):
                tok_int = int(tok)
                lp = float(step_logprobs[tok_int].item())
                completion_ids.append(tok_int)
                completion_logprobs.append(lp)
                if tok_int == eos_token_id:
                    break

            completion_ids = collapse_eos_token_id(completion_ids, eos_token_id)
            completion_logprobs = completion_logprobs[: len(completion_ids)]

            completion_ids_list.append(completion_ids)
            logprobs_list.append(completion_logprobs)

        return completion_ids_list, logprobs_list

    def prime_prefix_kv_cache(
        self,
        prefix_ids: list[int],
        *,
        max_kv_size: int | None = None,
        kv_bits: int | None = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
    ) -> None:
        """
        Pre-compute and store a prompt KV cache for a prefix so subsequent
        generations that share the same prefix can skip recomputation.
        """
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache

        if len(prefix_ids) == 0:
            self._prefix_ids = None
            self._prefix_cache = None
            return

        cache = make_prompt_cache(self.model, max_kv_size=max_kv_size)
        # Cache everything except the final token, so generation can continue
        # by re-feeding prefix[-1] as the first token.
        y = mx.array(prefix_ids[:-1], dtype=mx.uint32) if len(prefix_ids) > 1 else mx.array([], dtype=mx.uint32)
        for _ in generate_step(
            y,
            self.model,
            max_tokens=0,
            prompt_cache=cache,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        ):
            pass

        mx.eval([c.state for c in cache])
        self._prefix_ids = list(prefix_ids)
        self._prefix_cache = cache


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