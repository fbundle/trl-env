from typing import Generator, Any

from dataclasses import dataclass

import torch
from torch import Tensor
from jaxtyping import Float, Int

from transformers import Cache, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
# from transformers.models.auto.modeling_auto import _BaseModelWithGenerate


class _BaseModelWithGenerate(PreTrainedModel, GenerationMixin):
    pass
type Token = int

@dataclass
class GenerateIteration:
    sample: tuple[Token, float] | None
    past_key_values: Cache | None

class TransformerEngine:
    def __init__(self, model: _BaseModelWithGenerate):
        self.model = model
    
    def stream_generate(
        self, input_token_list: list[Token],
        eos_token_set: set[Token],
        max_completion_length: int = 256,
        temperature: float = 0.0,
        past_key_values: Cache | None = None,
    ) -> Generator[GenerateIteration, None, None]:
        input_token_queue: list[Token] = input_token_list

        with torch.no_grad():
            for _ in range(max_completion_length):
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
                if temperature > 0:
                    scaled_probs: Float[Tensor, "b d"] = torch.softmax(logits / temperature, dim=-1)
                    sampled_tokens: Int[Tensor, "b"] = torch.multinomial(scaled_probs, num_samples=1).squeeze(dim=-1)
                else:
                    sampled_tokens: Int[Tensor, "b"] = torch.argmax(logits, dim=-1)


                # get original logprobs
                logprobs: Float[Tensor, "b d"] = torch.log_softmax(logits, dim=-1)
                sampled_logprobs: Float[Tensor, "b"] = logprobs[range(len(sampled_tokens)), sampled_tokens]

                # yield output
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

    def generate(self, *args, **kwargs) -> tuple[list[tuple[Token, float]], Cache | None]:
        sample_list = []
        past_key_values = None

        for i in self.stream_generate(*args, **kwargs):
            if i.sample is not None:
                sample_list.append(i.sample)
            past_key_values = i.past_key_values
        
        return sample_list, past_key_values
        
        

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "mps"
    model_path = "Qwen/Qwen3.5-0.8B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model: _BaseModelWithGenerate = AutoModelForCausalLM.from_pretrained(model_path).to(device) #type: ignore
    engine = TransformerEngine(model=model)

    past_key_values = None

    text1 = f"<|im_start|>user\n the cat is lying on a table <|im_end|>\n<|im_start|>assistant\n"
    input_token_list1: list[Token] = tokenizer(text1).input_ids

    sample_list1, past_key_values = engine.generate(
        input_token_list=input_token_list1,
        eos_token_set={tokenizer.eos_token_id},
        past_key_values=past_key_values,
    )

    assert past_key_values is not None

    text2 = f"<|im_start|>user\n what is the cat lying on?  <|im_end|>\n<|im_start|>assistant\n"
    input_token_list2: list[Token] = tokenizer(text2).input_ids
    
    sample_list2, past_key_values = engine.generate(
        input_token_list=input_token_list2,
        eos_token_set={tokenizer.eos_token_id},
        past_key_values=past_key_values,
    )

    completion_list1: list[Token] = list(list(zip(*sample_list1))[0]) # type: ignore
    completion_list2: list[Token] = list(list(zip(*sample_list2))[0]) # type: ignore


    token_list = input_token_list1 + completion_list1 + input_token_list2 + completion_list2

    text = tokenizer.decode(token_list)
    print(text)

