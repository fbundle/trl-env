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
            gpu_memory_utilization=0.4,
            enable_prefix_caching=True,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_completion_length,
            logprobs=1,
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
        tokens = o.outputs[0].token_ids
        logprobs = [list(lp.values())[0].logprob for lp in o.outputs[0].logprobs]
        return tokens, logprobs
