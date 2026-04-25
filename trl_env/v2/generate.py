
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, Callable
from jaxtyping import Float, Int

from torch import Tensor
import torch


type Token = int
type TokenList = Int[Tensor, "n"]
type Logits = Float[Tensor, "d"]

type SampleFunc = Callable[[Logits], Token]
type ModelFunc[T] = Callable[[T, TokenList], tuple[T, Logits]]

@dataclass
class StreamGenerationIteration[T]:
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
) -> Iterator[StreamGenerationIteration[T]]:

    # put new_token_list into the model
    # get logits for the first sampling step
    state, logits = model_func(prev_state, new_token_list)

    for _ in range(max_completion_length):
        # sample a new token
        token = sample_func(logits)
        # put the new token into the model 
        state, next_logits = model_func(state, torch.tensor([token]))
        # yield the state after the new token
        yield StreamGenerationIteration(state=state, token=token, logits=logits)
        # stop condition
        if token in eos_token_set:
            break
        # prepare for the next sampling step
        logits = next_logits


def make_sample_func(temperature: float = 0.0) -> SampleFunc:
    def sample_func(logits: Logits) -> Token:
        if temperature > 0:
            dist: Float[Tensor, "d"] = torch.softmax(logits / temperature, dim=-1)
            token: int = int(torch.multinomial(dist, num_samples=1).squeeze(dim=-1).item())
        else:
            token: int = int(torch.argmax(logits, dim=-1).item())
        return token
    return sample_func


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

    from trl_env.v2.tokenizer import TransformerTokenizer
    from trl_env.v2.processor import *

    device = "mps"
    model_path = "Qwen/Qwen3.5-0.8B"

    t = AutoTokenizer.from_pretrained(model_path)
    m: _BaseModelWithGenerate = AutoModelForCausalLM.from_pretrained(model_path).to(device) #type: ignore

    eos_token: int = t.eos_token_id

    sample_func = make_sample_func(temperature=0.0)
    model_func = make_model_func(model=m)

    tokenizer = TransformerTokenizer(t)
    processor = qwen3_processor

    def dict_transpose[T](ds: Iterable[dict[str, T]]) -> dict[str, list[T]]:
        o: dict[str, list[T]] = {}
        for d in ds:
            for k, v in d.items():
                if k not in o:
                    o[k] = []
                o[k].append(v)
        return o

    def generate_text(state: Cache | None, new_text: str) -> tuple[Cache | None, str]:
        new_token_list: list[int] = tokenizer.encode(new_text)
        
        i: Iterator[StreamGenerationIteration[Cache | None]] = stream_generate(
            new_token_list=torch.tensor(new_token_list),
            prev_state=state,
            model_func=model_func,
            sample_func=sample_func,
            eos_token_set={eos_token},
            max_completion_length=512,
        )

        completion_token_list = []
        new_state = None
        for o in i:
            new_state = o.state
            completion_token_list.append(o.token)
        
        completion_text = tokenizer.decode(completion_token_list)
        return new_state, completion_text

    state: Cache | None = None

    new_text = processor.append_user_input("the cat is lying on the rooftop")
    print(new_text)
    state, completion_text = generate_text(state, new_text)
    print(completion_text)

    new_text = processor.append_user_input("where is the cat?")
    print(new_text)
    state, completion_text = generate_text(state, new_text)
    print(completion_text)


