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



# rewrite batch_generate in mlx_lm to return logprobs
from typing import List, Optional, Union
from mlx_lm.generate import BatchGenerator, GenerationBatch

Response = GenerationBatch.Response


def batch_generate(
    model,
    tokenizer,
    prompts: List[List[int]],
    prompt_caches: Optional[List[List[Any]]] = None,
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    return_prompt_caches: bool = False,
    **kwargs,
) -> Generator[Response, None, None]:
    """
    Generate responses for the given batch of prompts.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompts (List[List[int]]): The input prompts.
       prompt_caches (List[List[Any]], optional): Pre-computed prompt-caches
          for each input prompt. Note, unlike ``generate_step``, the caches
          won't be updated in-place.
       verbose (bool): If ``True``, print tokens and timing information.
          Default: ``False``.
       max_tokens (Union[int, List[int]): Maximum number of output tokens. This
          can be per prompt if a list is provided.
       return_prompt_caches (bool): Return the prompt caches in the batch
          responses. Default: ``False``.
       kwargs: The remaining options get passed to :obj:`BatchGenerator`.
          See :obj:`BatchGenerator` for more details.
    """

    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        **kwargs,
    )
    num_samples = len(prompts)
    if isinstance(max_tokens, int):
        max_tokens = [max_tokens] * len(prompts)

    uids = gen.insert(prompts, max_tokens, caches=prompt_caches)
    uid_to_index: dict[int, int] = {uid: i for i, uid in enumerate(uids)}
    with gen.stats() as stats:
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

class MlxEngine(Engine):
    def __init__(
        self,
        model_path: str,
        max_completion_length: int = 256,
        temperature: float = 0.0,
        eos_tokens: list[str] | None = None,
    ) -> None:
        model, tokenizer, _ = mlx_lm.load(  # type: ignore
            path_or_hf_repo=model_path,
            return_config=True,
        )
        self.model: nn.Module = model
        self.tokenizer: TokenizerWrapper = tokenizer

        # 
        self.temperature = temperature
        self.max_completion_length = max_completion_length
        if eos_tokens is not None:
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
        logprobs_list: list[list[float]] = []

        response = mlx_lm.batch_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=input_ids_list,
            max_tokens=self.max_completion_length,
            sampler=mlx_lm.sample_utils.make_sampler(
                temp=self.temperature,
            ),
        )

        import pdb; pdb.set_trace()
            
        completion_ids_list.append(completion_ids)
        logprobs_list.append(log_probs)


        return completion_ids_list, logprobs_list


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



if __name__ == "__main__":
    path = "Qwen/Qwen3.5-0.8B"
    m = MlxEngine(
        model_path=path,
        max_completion_length=256,
        temperature=0.6,
        eos_tokens=None,
    )

    input_text = [
        "hello, this is an example",
        "water is blue"
    ]

    input_text: list[str] = [m.tokenizer.apply_chat_template( # type: ignore
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
    
