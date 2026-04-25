from typing import Iterator
from jaxtyping import Float

import torch
from transformers import Cache

from trl_env.v2.generate import StreamGenerationIteration, Token, stream_generate
from trl_env.v2.model import RolloutModel
from trl_env.v2.generate_transformer import BaseModelWithGenerate, make_model_func, make_sample_func

class TransformerRolloutModel(RolloutModel):
    def __init__(self,
    model: BaseModelWithGenerate,
    temperature: float,
    eos_token_set: set[Token],
    max_completion_length: int,
) -> None:
        self.model: BaseModelWithGenerate = model
        self.cache: Cache | None = None
        self.last_length = 0

        self.temperature = temperature
        self.eos_token_set = eos_token_set
        self.max_completion_length = max_completion_length
    
    def generate(self, input_ids: list[int]) -> tuple[list[int], list[float]]:
        new_input_ids = input_ids[self.last_length:]
        

        i: Iterator[StreamGenerationIteration[Cache | None]] = stream_generate(
            new_token_list=torch.tensor(new_input_ids),
            prev_state=self.cache,
            model_func=make_model_func(model=self.model),
            sample_func=make_sample_func(temperature=self.temperature),
            eos_token_set=self.eos_token_set,
            max_completion_length=self.max_completion_length,
        )

        completion_token_list = []
        logprob_list = []
        new_cache = None
        for o in i:
            logprobs: Float[torch.Tensor, "d"] = torch.log_softmax(o.logits, dim=-1)
            logprob: float = float(logprobs[o.token].item())
            
            completion_token_list.append(o.token)
            logprob_list.append(logprob)
            new_cache = o.state
        
        self.cache = new_cache
        self.last_length = len(input_ids) + len(completion_token_list)

        
        return completion_token_list, logprob_list