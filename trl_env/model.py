

from typing import Any, Protocol
from jaxtyping import Int, Float

import torch
from torch import Tensor
from torch.functional import F  # type: ignore
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase


class Model(Protocol):
    device: torch.device
    def tokenizer_encode(self, input_text: str) -> list[int]: ...
    def tokenizer_decode(self, completion_ids: list[int]) -> str: ...
    def model_batch_generate(self, input_ids_list: list[list[int]]) -> tuple[list[list[int]], list[list[float]]]: ...


def collapse_eos_token_id(completion_ids: Int[Tensor, "n"], eos_token_id: int) ->  Int[Tensor, "n1"]:
    indices: Int[Tensor, "k"] = (completion_ids == eos_token_id).nonzero(as_tuple=True)[0]
    if len(indices) == 0: # no eos_token found
        return completion_ids
    # eos_token found
    # [tok, tok, eos, eos, eos] -> [tok, tok, eos]
    index: int = int(indices[0]) + 1
    return completion_ids[:index]

class TransformerModel(Model):
    def __init__(
            self, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel,
            generation_kwargs: dict[str, Any] | None = None,
        ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device 
        self.eos_token_id = int(self.tokenizer.eos_token_id) # type: ignore
        self.generation_kwargs = {
            "max_new_tokens": 256,
            "eos_token_id": [self.eos_token_id],
        }
        if generation_kwargs is not None:
            self.generation_kwargs.update(generation_kwargs)
        
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.eos_token_id

        # pop hard coded keys
        for key in ["input_ids", "attention_mask", "output_logits", "return_dict_in_generate"]:
            if key in self.generation_kwargs:
                raise RuntimeError(f"generation_kwargs[{key}] must not be set")
        
    
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
            completion_ids: list[int] = collapse_eos_token_id(completion_ids_batch[i, :], self.eos_token_id).tolist()
            # x[[a, b, c, d], [A, B, C, D]] = [x[a, A], x[b, B], x[c, C], x[d, D]]
            logprobs: list[float] = batch_logprobs[i, range(len(completion_ids)), completion_ids].tolist()
            
            completion_ids_list.append(completion_ids)
            logprobs_list.append(logprobs)
        
        return completion_ids_list, logprobs_list
    
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    path = "Qwen/Qwen3.5-0.8B"
    m = TransformerModel(
        tokenizer=AutoTokenizer.from_pretrained(path),
        model=AutoModelForCausalLM.from_pretrained(path).to("mps"), # type: ignore
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
    import pdb; pdb.set_trace()
    
