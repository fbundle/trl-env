from .decoder import RolloutDecoder
from vllm import LLM, RequestOutput, SamplingParams
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
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_completion_length,
            logprobs=1,
            stop_token_ids=list(eos_token_set),
        )
    
    def update_weights(self, training_model: PreTrainedModel):
        named_params = {name: param.data for name, param in training_model.named_parameters()}
        self.llm.collective_rpc(
            "update_weights_from_dict",
            kwargs=dict(named_params=named_params),
        )

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


    


