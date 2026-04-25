import itertools
from typing import Callable, Generator

from dataclasses import dataclass

import torch
from torch import Tensor
from jaxtyping import Float, Int

from transformers import Cache, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class _BaseModelWithGenerate(PreTrainedModel, GenerationMixin):
    pass

type Sampler = Callable[[Float[Tensor, "b d"]], Int[Tensor, "b"]]

def make_sampler(temperature: float = 0.0) -> Sampler:
    def sampler(logits: Float[Tensor, "b d"]) -> Int[Tensor, "b"]:
        if temperature > 0:
            probs: Float[Tensor, "b d"] = torch.softmax(logits / temperature, dim=-1)
            tokens: Int[Tensor, "b"] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        else:
            tokens: Int[Tensor, "b"] = torch.argmax(logits, dim=-1)
        return tokens
    
    return sampler

type StopCond = Callable[[int, int], bool]

def make_stop_cond(eos_token_set: set[int], max_completion_length: int) -> StopCond:
    def stop_cond(turn: int, token: int) -> bool:
        return (token in eos_token_set) or (turn >= max_completion_length)

    return stop_cond

def stream_generate(
    input_tokens: list[int],
    model: _BaseModelWithGenerate,
    sampler: Sampler,
    stop_cond: StopCond,
    past_key_values: Cache | None = None,
) -> Generator[tuple[int, float, Cache | None], None, None]:
    
    next_input_tokens: list[int] = input_tokens

    for turn in itertools.count():
        # batch_size b = 1
        input_ids: Int[Tensor, "b n"] = torch.tensor([next_input_tokens], device=model.device)
        o: CausalLMOutputWithPast = model.__call__(
            input_ids=input_ids,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=True,
            past_key_values=past_key_values,
        )

        # get logits - o.logits: Float[Tensor, "b n d"]
        assert o.logits is not None
        logits: Float[Tensor, "b d"] = o.logits[:, -1, :]

        # sample next tokens
        sample: Int[Tensor, "b"] = sampler(logits)

        # get logprob of the sample
        # x[[a, b, c], [A, B, C]] = [x[a, A], x[b, B], x[c, C]]
        logprobs: Float[Tensor, "b d"] = torch.log_softmax(logits, dim=-1)
        sample_logprob: Float[Tensor, "b"] = logprobs[range(len(sample)), sample]

        # batch_size b = 1
        next_token: int = sample.tolist()[0]
        next_logprob: float = sample_logprob.tolist()[0]
        assert isinstance(next_token, int) and isinstance(next_logprob, float)

        # prepare next iteration
        next_input_tokens = [next_token]
        past_key_values = o.past_key_values

        # yield output
        yield (next_token, next_logprob, past_key_values)

        # early stop 
        if stop_cond(turn, next_token):
            break

def finalize_cache(past_token: int, past_key_values: Cache | None) -> Cache | None:
    input_ids: Int[Tensor, "b n"] = torch.tensor([[past_token]], device=model.device)
    o: CausalLMOutputWithPast = model.__call__(
        input_ids=input_ids,
        output_logits=True,
        return_dict_in_generate=True,
        use_cache=True,
        past_key_values=past_key_values,
    )
    return o.past_key_values

def generate(*args, **kwargs) -> tuple[list[int], list[float], Cache | None]:
    next_token_list = []
    next_logprob_list = []
    last_past_key_values = None
    for next_token, next_logprob, past_key_values in stream_generate(*args, **kwargs):
        next_token_list.append(next_token)
        next_logprob_list.append(next_logprob)
        last_past_key_values = past_key_values
    
    last_past_key_values = finalize_cache(next_token_list[-1], last_past_key_values)
    return next_token_list, next_logprob_list, last_past_key_values


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "mps"
    model_path = "Qwen/Qwen3.5-0.8B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model: _BaseModelWithGenerate = AutoModelForCausalLM.from_pretrained(model_path).to(device) #type: ignore

    temperature = 0.6

    past_key_values = None

    text1 = f"<|im_start|>user\n the cat is lying on a table <|im_end|>\n<|im_start|>assistant\n"
    input_token_list1: list[int] = tokenizer(text1).input_ids

    completion_list1, _, past_key_values = generate(
        input_tokens=input_token_list1,
        model=model,
        sampler=make_sampler(temperature=temperature),
        stop_cond=make_stop_cond(eos_token_set={tokenizer.eos_token_id}, max_completion_length=64),
        past_key_values=past_key_values,
    )

    assert past_key_values is not None

    text2 = f"<|im_start|>user\n what is the cat lying on?  <|im_end|>\n<|im_start|>assistant\n"
    input_token_list2: list[int] = tokenizer(text2).input_ids
    
    completion_list2, _, past_key_values = generate(
        input_tokens=input_token_list2,
        model=model,
        sampler=make_sampler(temperature=temperature),
        stop_cond=make_stop_cond(eos_token_set={tokenizer.eos_token_id}, max_completion_length=128),
        past_key_values=past_key_values,
    )


    token_list = input_token_list1 + completion_list1 + input_token_list2 + completion_list2

    text = tokenizer.decode(token_list)
    print(text)

