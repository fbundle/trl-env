from typing import Generator, Literal

import torch
from torch import Tensor
from jaxtyping import Float, Int

from transformers import Cache, dataclass
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto.modeling_auto import _BaseModelWithGenerate

type Token = int

def get_last_token(token_list: list[Token]) -> Token | None:
    if len(token_list) == 0:
        return None
    else:
        return token_list[-1]

type FinishReason = Literal["length", "stop", None]

@dataclass
class GenerateIteration:
    sample: tuple[Token, float] | None
    past_key_values: Cache | None

class TransformerEngine:
    def __init__(self, model: _BaseModelWithGenerate):
        self.model = model
    
    def generate(
        self, input_token_list: list[Token],
        eos_token_set: set[Token],
        max_completion_length: int = 256,
        temperature: float = 0.0,
        past_key_values: Cache | None = None,
    ) -> Generator[GenerateIteration, None, None]:
        input_token_queue: list[Token] = input_token_list

        with torch.no_grad():
            for turn in range(max_completion_length):
                # batch_size b = 1
                input_ids: Int[Tensor, "b n"] = torch.tensor([input_token_queue], device=self.model.device)
                o: CausalLMOutputWithPast = self.model.__call__(
                    input_ids=input_ids,
                    output_logits=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                )


                assert o.logits is not None
                # get logits
                # o.logits: Float[Tensor, "b n d"]
                logits: Float[Tensor, "b d"] = o.logits[:, -1, :]

                # sample token
                # x[[a, b, c], [A, B, C]] = [x[a, A], x[b, B], x[c, C]]
                scaled_probs: Float[Tensor, "b d"] = torch.softmax(logits / temperature, dim=-1)
                sampled_tokens: Int[Tensor, "b"] = torch.multinomial(scaled_probs, num_samples=1).squeeze(dim=-1)

                # original logprobs
                logprobs: Float[Tensor, "b d"] = torch.log_softmax(logits, dim=-1)
                sampled_logprobs: Float[Tensor, "b"] = logprobs[range(len(sampled_tokens)), sampled_tokens]


                sampled_token: Token = sampled_tokens.tolist()[0]
                sampled_logprob: float = sampled_logprobs.tolist()[0]

                yield GenerateIteration(
                    sample=(sampled_token, sampled_logprob),
                    past_key_values=None
                )

                # prepare next generate
                input_token_queue = [sampled_token]
                past_key_values = o.past_key_values

                # early stop 
                if sampled_token in eos_token_set:
                    break
            
            # calculate past_key_values for next user input
            input_ids: Int[Tensor, "b n"] = torch.tensor([input_token_queue], device=self.model.device)
            o: CausalLMOutputWithPast = self.model.__call__(
                input_ids=input_ids,
                output_logits=True,
                return_dict_in_generate=True,
                use_cache=True,
                past_key_values=past_key_values,
            )

            yield GenerateIteration(
                sample=None,
                past_key_values=o.past_key_values,
            )
