from .decoder import RolloutDecoder

import os
# https://github.com/huggingface/trl/issues/3859
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, RequestOutput, SamplingParams
from vllm.config import CompilationConfig
from transformers import PreTrainedModel

class VLLMRolloutDecoder(RolloutDecoder):
    def __init__(self,
        model_path: str,
        temperature: float,
        eos_token_set: set[int],
        max_completion_length: int,
    ) -> None:
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=0.2,
            enable_prefix_caching=True,
            compilation_config=CompilationConfig(mode=0),  # 0 = no compilation
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_completion_length,
            logprobs=1,
            stop_token_ids=list(eos_token_set),
        )
    

    def _fix_param_name_to_vllm(self, name: str, extra_prefixes: list[str] | None = None) -> str:
        """Fix parameter name for vLLM compatibility."""
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def update_weights(self, training_model: PreTrainedModel):
        # stole from trl.generation.vllm_generation
        # 
        for name, param in training_model.named_parameters():
            # When using PEFT, we need to recover the original parameter name
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            # Skip PEFT layers: they don't exist in vLLM, and they are merged already.
            if training_model.prefix in name:
                continue
            # When module to save, remove its prefix and discard the original module
            if "original_module" in name:
                continue
            name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])



            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights([(name, param.data)])

        self.llm.reset_prefix_cache()

    def generate(self, input_ids: list[int]) -> tuple[list[int], list[float]]:
        o_list: list[RequestOutput] = self.llm.generate(
            [input_ids],
            sampling_params=self.sampling_params,
        )
        assert len(o_list) == 1
        o = o_list[0]
        output_ids = o.outputs[0].token_ids
        logprobs = [list(lp.values())[0].logprob for lp in o.outputs[0].logprobs]
        return output_ids, logprobs

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


    


