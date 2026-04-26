from __future__ import annotations

from typing import Any, Callable, Generator

import mlx.nn as nn
import mlx.core as mx
import mlx_lm
import mlx_lm.sample_utils
from mlx.utils import tree_flatten

from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from trl_env.engine import Engine

from trl.trainer.grpo_trainer import RolloutFunc, GRPOTrainer, RewardFunc

from trl_env.environment import Env
from trl_env.processor import Processor
from trl_env.rollout import batch_rollout

# feature flags

# rewrite batch_generate in mlx_lm to return logprobs
from typing import List, Optional, Union
from mlx_lm.generate import BatchGenerator, GenerationBatch

Response = GenerationBatch.Response


def _strip_known_state_dict_prefixes(state_dict: dict[str, mx.array]) -> dict[str, mx.array]:
    """
    GRPO training commonly wraps the HF model in PEFT (LoRA), which prefixes keys like:
      - base_model.model.*
      - base_model.model.model.*
    MLX model sanitizers expect the *base* HF key layout, so we strip known wrappers.
    """

    def strip_one(k: str) -> str:
        # PEFT wrappers
        for p in ("base_model.model.", "base_model."):
            if k.startswith(p):
                return k[len(p) :]
        return k

    # Repeatedly strip because we sometimes see nested "base_model.model.model."
    out: dict[str, mx.array] = {}
    for k, v in state_dict.items():
        kk = k
        while True:
            kk2 = strip_one(kk)
            if kk2 == kk:
                break
            kk = kk2
        out[kk] = v
    return out

def stream_batch_generate(
    model,
    tokenizer,
    prompts: List[List[int]],
    prompt_caches: Optional[List[List[Any]]] = None,
    max_tokens: Union[int, List[int]] = 128,
    **kwargs,
) -> Generator[tuple[int, Response], None, None]:


    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        **kwargs,
    )
    if isinstance(max_tokens, int):
        max_tokens = [max_tokens] * len(prompts)

    uids = gen.insert(prompts, max_tokens, caches=prompt_caches)
    uid_to_index: dict[int, int] = {uid: i for i, uid in enumerate(uids)}
    while responses := gen.next_generated():
        for r in responses:
            yield uid_to_index[r.uid], r
    gen.close()

def collapse_eos_token_id(completion_ids: list[int], eos_token_id: int) -> list[int]:
    try:
        index = completion_ids.index(eos_token_id)
    except ValueError:
        return completion_ids
    return completion_ids[: index + 1]

class MlxEngine:
    def __init__(
        self, model_path: str,
        max_completion_length: int = 256,
        temperature: float = 0.0,
        extra_eos_token_list: list[str] | None = None,
    ):
        # the easiest way to load model into MLX is just giving it a HF model directory
        model, tokenizer, _ = mlx_lm.load(  # type: ignore
            path_or_hf_repo=model_path,
            return_config=True,
        )
        self.model: nn.Module = model
        self.tokenizer: TokenizerWrapper = tokenizer

        self.temperature = temperature
        self.max_completion_length = max_completion_length
        if extra_eos_token_list is not None:
            for eos_token in extra_eos_token_list:
                self.tokenizer.add_eos_token(eos_token)
    
    def update_weights(self, state_dict: dict[str, mx.array]):
        state_dict = _strip_known_state_dict_prefixes(state_dict)

        if hasattr(self.model, "sanitize"):
            state_dict = self.model.sanitize(state_dict)

        curr_weights = tree_flatten(self.model.parameters(), destination={})
        new_weights: dict[str, mx.array] = {}

        missing_keys = set()
        for key in curr_weights.keys():
            if key in state_dict:
                val = state_dict[key]
                new_weights[key] = val
            else:
                missing_keys.add(key)
        if len(missing_keys) > 0:
            print("[ERROR] missing keys", len(missing_keys), len(curr_weights))

        self.model.load_weights(file_or_weights=new_weights.items(), strict=False)
    
    def model_batch_generate(
        self, input_ids_list: list[list[int]],
        prompt_cache_list: Optional[List[List[Any]]] = None,
        return_prompt_caches: bool = False,
    ) -> tuple[list[list[int]], list[list[float]], List[Any]]:
    
        completion_ids_list: list[list[int]] = [[] for _ in input_ids_list]
        logprobs_list: list[list[float]] = [[] for _ in input_ids_list]
        new_prompt_cache_list: list[list[Any]] = [[] for _ in input_ids_list]

        response = stream_batch_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=input_ids_list,
            max_tokens=self.max_completion_length,
            sampler=mlx_lm.sample_utils.make_sampler(
                temp=self.temperature,
            ),
            prompt_caches=prompt_cache_list,
        )

        for i, r in response:
            if r.finish_reason is not None:
                if return_prompt_caches:
                    new_prompt_cache_list[i] = r.prompt_cache
            if r.finish_reason != "stop":
                token = r.token
                logprob = r.logprobs[token].item()

                completion_ids_list[i].append(token)
                logprobs_list[i].append(logprob)
        
        return completion_ids_list, logprobs_list, new_prompt_cache_list


class MlxRolloutEngine(Engine):
    def __init__(
        self,
        model_path: str,
        **kwargs,
    ) -> None:
        self.engine = MlxEngine(model_path, **kwargs)
        self.prompt_cache = None

    def update_weights_and_reset_prompt_cache(self, state_dict: dict[str, mx.array]):
        self.engine.update_weights(state_dict)
        self.prompt_cache = None # reset prompt_cache

    def tokenizer_encode(self, input_text: str) -> list[int]:
        return self.engine.tokenizer._tokenizer(input_text).input_ids

    def tokenizer_decode(self, completion_ids: list[int]) -> str:
        output_text = self.engine.tokenizer._tokenizer.decode(completion_ids)
        assert isinstance(output_text, str)
        return output_text
    
    def model_batch_generate(self, input_ids_list: list[list[int]]) -> tuple[list[list[int]], list[list[float]]]:
        completion_ids_list, logprobs_list, new_prompt_cache_list = self.engine.model_batch_generate(
            input_ids_list=input_ids_list,
            prompt_cache_list=self.prompt_cache,
            return_prompt_caches=True,
        )
        self.prompt_cache = new_prompt_cache_list
        return completion_ids_list, logprobs_list


def make_rollout_func(
    engine: MlxEngine, processor: Processor, env_factory: Callable[[], Env],
    system_prompt: str, max_conversation_length: int,
) -> RolloutFunc:
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, Any]:
        try:
            model = trainer.model
            state_dict = {k: mx.array(v.detach().cpu()) for k, v in model.state_dict().items()}
            engine.update_weights_and_reset_prompt_cache(state_dict)
            
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



if __name__ == "__main__":
    from transformers import AutoModelForCausalLM


    path = "Qwen/Qwen3.5-0.8B"
    m = MlxRolloutEngine(
        model_path=path,
        max_completion_length=256,
        temperature=0.6,
        extra_eos_token_list=None,
    )

    model = AutoModelForCausalLM.from_pretrained(path)

    m.update_weights_and_reset_prompt_cache({k: mx.array(v) for k, v in model.state_dict().items()})

    input_text = [
        "hello, this is an example",
        "water is blue"
    ]

    input_text: list[str] = [m.engine.tokenizer.apply_chat_template( # type: ignore
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    ) for text in input_text]

    print(input_text)

    input_ids_list = [m.tokenizer_encode(text) for text in input_text]
    completion_ids_list, logprobs_list = m.model_batch_generate(input_ids_list)
    output_text = [m.tokenizer_decode(completion_ids) for completion_ids in completion_ids_list]
    
    print(output_text)


    input_text = [
        "umm, what was I saying again",
        "umm, what was I saying again",
    ]

    input_text: list[str] = [m.engine.tokenizer.apply_chat_template( # type: ignore
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    ) for text in input_text]

    print(input_text)

    input_ids_list = [m.tokenizer_encode(text) for text in input_text]
    completion_ids_list, logprobs_list = m.model_batch_generate(input_ids_list)
    output_text = [m.tokenizer_decode(completion_ids) for completion_ids in completion_ids_list]
    
    print(output_text)
