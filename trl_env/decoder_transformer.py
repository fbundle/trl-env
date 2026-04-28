from typing import Iterator
from jaxtyping import Float

from peft import PeftModel
import torch
from transformers import Cache, PreTrainedModel, Trainer

from .generate.generate import StreamGenerationIteration, stream_generate
from .decoder import RolloutDecoder, RolloutDecoderFactory
from .generate.generate_transformer import make_model_func, make_sample_func


class TransformerDecoder(RolloutDecoder):
    def __init__(self,
        model: PreTrainedModel | PeftModel,
        temperature: float,
        eos_token_set: set[int],
        max_completion_length: int,
    ) -> None:
        self.model: PreTrainedModel | PeftModel = model
        self.cache: Cache | None = None
        self.last_length = 0

        self.temperature = temperature
        self.eos_token_set = eos_token_set
        self.max_completion_length = max_completion_length

    def generate(self, input_ids: list[int]) -> tuple[list[int], list[float]]:
        was_training = self.model.training
        self.model.eval()
        try:
            new_input_ids = input_ids[self.last_length:]

            i: Iterator[StreamGenerationIteration[Cache | None]] = stream_generate(
                new_token_list=torch.tensor(new_input_ids),
                prev_state=self.cache,
                model_func=make_model_func(model=self.model), # type: ignore
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
        finally:
            self.model.train(was_training)

        return completion_token_list, logprob_list

class TransformerDecoderFactory(RolloutDecoderFactory):
    def __init__(self,
        temperature: float,
        eos_token_set: set[int],
        max_completion_length: int,
    ) -> None:
        self.temperature = temperature
        self.eos_token_set = eos_token_set
        self.max_completion_length = max_completion_length
    
    def make_decoder(self, trainer: Trainer) -> RolloutDecoder:
        return TransformerDecoder(
            model=trainer.model,  # type: ignore
            temperature=self.temperature,
            eos_token_set=self.eos_token_set,
            max_completion_length=self.max_completion_length,
        )