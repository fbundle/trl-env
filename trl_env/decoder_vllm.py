

from .decoder import RolloutDecoder, RolloutDecoderFactory
from transformers import Trainer

from trl.generation.vllm_generation import VLLMGeneration

class VLLMDecoder(RolloutDecoder):
    def __init__(self,
        vllm: VLLMGeneration,
    ) -> None:
        self.vllm = vllm
    
    def generate(self, input_ids: list[int]) -> tuple[list[int], list[float]]:
        prompt_ids, completion_ids, logprobs, logprob_token_ids = self.vllm.generate(
            prompts=[input_ids], num_generations=1,
            images=None,
        )

        return completion_ids[0], [lp[0] for lp in logprobs[0]]

class VLLMDecoderFactory(RolloutDecoderFactory):
    def __init__(self,
        temperature: float,
        eos_token_set: set[int],
        max_completion_length: int,
        gpu_memory_utilization: float = 0.5,
        enable_sleep_mode: bool = True,
    ) -> None:
        self.vllm: VLLMGeneration | None = None
        self.temperature = temperature
        self.eos_token_set = eos_token_set
        self.max_completion_length = max_completion_length
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_sleep_mode = enable_sleep_mode

        self._last_synced_step: int = -1
    
    def make_decoder(self, trainer: Trainer) -> RolloutDecoder:
        if self.vllm is None:
            self.vllm = VLLMGeneration(
                model=trainer.model, # type: ignore
                accelerator=trainer.accelerator,
                is_fsdp_enabled=trainer.is_fsdp_enabled,
                processing_class=trainer.processing_class, # type: ignore
                mode="colocate",
                gpu_memory_utilization=self.gpu_memory_utilization,
                enable_sleep_mode=self.enable_sleep_mode,
                max_completion_length=self.max_completion_length,
                temperature=self.temperature,
                generation_kwargs=dict(
                    stop_token_ids=list(self.eos_token_set),
                    skip_special_tokens=False,
                    include_stop_str_in_output=True,
                ),
            )
        current_step = trainer.state.global_step
        if current_step != self._last_synced_step:
            self.vllm.sync_weights()
            self._last_synced_step = current_step

        self.vllm.sync_weights()
        return VLLMDecoder(vllm=self.vllm)