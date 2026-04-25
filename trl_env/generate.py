

from typing import Iterator, Protocol, Callable
from jaxtyping import Float, Int

from torch import Tensor
import torch
from transformers import Cache, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

type Token = int
type TokenList = Int[Tensor, "n"]
type Logits = Float[Tensor, "d"]

class Sampler(Protocol):
    def sample(self, logits: Logits) -> Token: ...

class Model(Protocol):
    def consume(self, token_list: TokenList) -> Logits: ...

def stream_generate(
    token_list: TokenList,
    model: Model,
    sampler: Sampler,
    eos_token_set: set[Token],
    max_completion_length: int,
) -> Iterator[tuple[Token, Logits]]:
    token = None
    try:
        for _ in range(max_completion_length):
            logits = model.consume(token_list)
            token = sampler.sample(logits)
            yield token, logits
            if token in eos_token_set:
                break
            # prepare the next iteration
            token_list = torch.tensor([token])
    finally:
        if token is not None:
            token_list = torch.tensor([token])
            _ = model.consume(token_list)

# basic sampler

class BasicSampler(Sampler):
    def __init__(self, temperature: float = 0.0) -> None:
        self.temperature = temperature
    
    def sample(self, logits: Tensor) -> Token:
        if self.temperature > 0:
            dist: Float[Tensor, "d"] = torch.softmax(logits / self.temperature, dim=-1)
            token: int = int(torch.multinomial(dist, num_samples=1).squeeze(dim=-1).item())
        else:
            token: int = int(torch.argmax(logits, dim=-1).item())
        return token
        




# TransformerGenerate

class _BaseModelWithGenerate(PreTrainedModel, GenerationMixin):
    ...

class TransformerModel(Model):
    def __init__(self, model: _BaseModelWithGenerate):
        self.model: _BaseModelWithGenerate = model
        self.cache: Cache | None = None
    
    def consume(self, token_list: TokenList) -> Logits:
        with torch.no_grad():
            batch_input_ids: Int[Tensor, "b n"] = token_list.unsqueeze(dim=0).to(self.model.device)
            o: CausalLMOutputWithPast = self.model.forward(
                input_ids=batch_input_ids,
                output_logits=True,
                return_dict_in_generate=True,
                use_cache=True,
                past_key_values=self.cache,
            )
            assert o.logits is not None
            batch_logits: Float[Tensor, "b n d"] = o.logits
            logits: Float[Tensor, "d"] = batch_logits[0, -1, :]
            self.cache = o.past_key_values
            return logits




