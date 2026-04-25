

from dataclasses import dataclass
from typing import Iterator, Callable
from jaxtyping import Float, Int

from torch import Tensor
import torch


type Token = int
type TokenList = Int[Tensor, "n"]
type Logits = Float[Tensor, "d"]

type SampleFunc = Callable[[Logits], Token]
type ModelFunc[T] = Callable[[T, TokenList], tuple[T, Logits]]

@dataclass
class GenerateIteration[T]:
    state: T
    token: Token
    logits: Logits

def stream_generate[T](
    new_token_list: TokenList,
    prev_state: T,
    model_func: ModelFunc[T],
    sample_func: SampleFunc,
    eos_token_set: set[Token],
    max_completion_length: int,
) -> Iterator[tuple[T, Token, Logits]]:

    # put new_token_list into the model
    # get logits for the first sampling step
    state, logits = model_func(prev_state, new_token_list)

    for _ in range(max_completion_length):
        # sample a new token
        token = sample_func(logits)
        # put the new token into the model 
        state, next_logits = model_func(state, torch.tensor([token]))
        # yield the state after the new token
        yield state, token, logits
        # stop condition
        if token in eos_token_set:
            break
        # prepare for the next sampling step
        logits = next_logits


def make_sample(temperature: float = 0.0) -> SampleFunc:
    def sample(logits: Logits) -> Token:
        if temperature > 0:
            dist: Float[Tensor, "d"] = torch.softmax(logits / temperature, dim=-1)
            token: int = int(torch.multinomial(dist, num_samples=1).squeeze(dim=-1).item())
        else:
            token: int = int(torch.argmax(logits, dim=-1).item())
        return token
    return sample


from transformers import Cache, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class _BaseModelWithGenerate(PreTrainedModel, GenerationMixin):
    ...

def make_model_func(model: _BaseModelWithGenerate) -> ModelFunc[Cache | None]:
    def model_func(prev_cache: Cache | None, token_list: TokenList) -> tuple[Cache | None, Logits]:
        with torch.no_grad():
            # batch_size b = 1
            batch_input_ids: Int[Tensor, "b n"] = token_list.unsqueeze(dim=0).to(model.device)
            o: CausalLMOutputWithPast = model.forward(
                input_ids=batch_input_ids,
                output_logits=True,
                return_dict_in_generate=True,
                use_cache=True,
                past_key_values=prev_cache,
            )
            # get logits - o.logits: Float[Tensor, "b n d"]
            assert o.logits is not None
            batch_logits: Float[Tensor, "b n d"] = o.logits
            logits: Float[Tensor, "d"] = batch_logits[0, -1, :]
            next_cache: Cache | None = o.past_key_values
            return next_cache, logits
    return model_func


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "mps"
    model_path = "Qwen/Qwen3.5-0.8B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model: _BaseModelWithGenerate = AutoModelForCausalLM.from_pretrained(model_path).to(device) #type: ignore