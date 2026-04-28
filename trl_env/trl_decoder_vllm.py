

from .decoder import RolloutDecoder
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin
from peft import PeftModel

from trl.generation.vllm_generation import VLLMGeneration
from accelerate import Accelerator


class VLLMRolloutDecoder(RolloutDecoder):
    def __init__(self,
        model: PreTrainedModel | PeftModel,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        temperature: float,
        eos_token_set: set[int],
        max_completion_length: int,
    ) -> None:
        self.model = model
        self.processing_class = processing_class
        self.temperature = temperature
        self.eos_token_set = eos_token_set
        self.max_completion_length = max_completion_length
        self.vllm = None
    
    def init_vllm(self,
        accelerator: Accelerator,
        is_fsdp_enabled: bool,
        
    ):
        self.vllm = VLLMGeneration(
            model=self.model,
            accelerator=accelerator,
            is_fsdp_enabled=is_fsdp_enabled,
            processing_class=self.processing_class,
            mode="colocate",
            gpu_memory_utilization=0.5,
            max_completion_length=self.max_completion_length,
            temperature=self.temperature,
            generation_kwargs=dict(
                stop_token_ids=list(self.eos_token_set),
                skip_special_tokens=False,
                include_stop_str_in_output=True,
            ),
        )
        

    def sync_weights(self):
        self.vllm.sync_weights()

    def generate(self, input_ids: list[int]) -> tuple[list[int], list[float]]:
        return self.vllm.generate(
            prompts=[input_ids], num_generations=1,
        )

if __name__ == "__main__":
    from typing import Iterable
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from .tokenizer import TransformerTokenizer
    from .processor import *

    device = "cpu"
    model_path = "Qwen/Qwen3.5-0.8B"

    t = AutoTokenizer.from_pretrained(model_path)
    m: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_path).to(device) #type: ignore

    eos_token: int = t.eos_token_id


    tokenizer = TransformerTokenizer(t)
    processor = qwen3_processor

    decoder = VLLMRolloutDecoder(
        model_path=model_path,
        temperature=1.0,
        eos_token_set={eos_token},
        max_completion_length=512,
    )

    decoder.update_weights(m)

    input_ids = tokenizer.encode(processor.append_user_input("the cat is lying on the rooftop"))
    output_ids, logprobs = decoder.generate(input_ids)

    output = tokenizer.decode(output_ids)
    print(logprobs)
    print(output)


    


