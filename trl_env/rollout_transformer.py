from typing import Any, Callable

from jaxtyping import Int, Float
import torch
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase
import warnings

import torch
from torch import Tensor
from torch.functional import F # type: ignore

from trl_env.engine import Engine
from trl_env.environment import Env
from trl_env.processor import Processor  

from typing import Any
from trl.trainer.grpo_trainer import RolloutFunc, GRPOTrainer, RewardFunc

from trl_env.rollout import batch_rollout



def collapse_eos_token_id(completion_ids: list[int], eos_token_id: int) ->  list[int]:
    try:
        n = completion_ids.index(eos_token_id)
    except ValueError:
        n = len(completion_ids) - 1
    
    # eos_token found
    # [tok, tok, eos, eos, eos] -> [tok, tok, eos]
    return completion_ids[:n+1]

class TransformerEngine(Engine):
    def __init__(
            self, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel,
            generation_kwargs: dict[str, Any] | None = None,
        ):
        self.tokenizer = tokenizer
        self.model = model
        self.generation_kwargs = {
            "max_new_tokens": 256,
            "eos_token_id": [self.tokenizer.eos_token_id],
        }
        if generation_kwargs is not None:
            self.generation_kwargs.update(generation_kwargs)
        
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # pop hard coded keys
        for key in ["input_ids", "attention_mask", "output_logits", "return_dict_in_generate"]:
            if key in self.generation_kwargs:
                raise RuntimeError(f"generation_kwargs[{key}] must not be set")
        
        # some warning
        import os
        if os.environ.get("PYTORCH_CUDA_ALLOC_CONF", default=None) != "expandable_segments:True":
            warnings.warn("[WARNING] KV cache has not been implemented, set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation")

    def tokenizer_encode(self, input_text: str) -> list[int]:
        return self.tokenizer(input_text).input_ids

    def tokenizer_decode(self, completion_ids: list[int]) -> str:
        output_text = self.tokenizer.decode(completion_ids)
        assert isinstance(output_text, str)
        return output_text


    def model_batch_generate(self, input_ids_list: list[list[int]]) -> tuple[list[list[int]], list[list[float]]]:
        e: BatchEncoding = self.tokenizer.pad(
            {"input_ids": input_ids_list},
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        o = self.model.generate( # type: ignore
            input_ids=e.input_ids,
            attention_mask=e.attention_mask,
            output_logits=True,
            return_dict_in_generate=True,
            **self.generation_kwargs,
        )
        batch_logits: Float[Tensor, "b n d"] = torch.stack(o.logits, dim=1)
        batch_logprobs: Float[Tensor, "b n d"] = F.log_softmax(batch_logits, dim=-1)
        b, n, d = batch_logits.shape
        completion_ids_batch: Int[Tensor, "b n"] = o.sequences[:, -n:]

        completion_ids_list : list[list[int]] = []
        logprobs_list: list[list[float]] = []

        for i in range(b):
            completion_ids: list[int] = collapse_eos_token_id(completion_ids_batch[i, :].tolist(), self.tokenizer.eos_token_id)
            # x[[a, b, c, d], [A, B, C, D]] = [x[a, A], x[b, B], x[c, C], x[d, D]]
            logprobs: list[float] = batch_logprobs[i, range(len(completion_ids)), completion_ids].tolist()
            
            completion_ids_list.append(completion_ids)
            logprobs_list.append(logprobs)
        
        return completion_ids_list, logprobs_list




def make_rollout_func(
    engine: TransformerEngine, processor: Processor, env_factory: Callable[[], Env],
    system_prompt: str, max_conversation_length: int,
) -> RolloutFunc:
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, Any]:
        try:
            engine.model = trainer.model
            # NOTE - only rollout on eval mode
            engine.model.eval()
            with torch.no_grad():
                state_list = batch_rollout(
                    engine=engine, processor=processor, env_factory=env_factory,
                    system_prompt=system_prompt, max_conversation_length=max_conversation_length,
                    seed_list=prompts,
                )
        finally:
            engine.model.train()
        
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
    from transformers import AutoTokenizer, AutoModelForCausalLM
    path = "Qwen/Qwen3.5-0.8B"
    
    m = TransformerEngine(
        tokenizer=AutoTokenizer.from_pretrained(path),
        model=AutoModelForCausalLM.from_pretrained(path).to("mps"),
        generation_kwargs=dict(
            max_new_tokens=256,
        ),
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
    
