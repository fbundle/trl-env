
import sys

import numpy as np
import sympy

from transformers import AutoModelForCausalLM, AutoTokenizer

from trl_env.decoder import RolloutDecoder

from trl_env.rollout import rollout
from trl_env.processor import qwen3_processor, qwen3_instruct_processor

from experiment.examples.discrete_logarithm.discrete_logarithm_env import EXTRA_EOS_TOKEN_LIST, DiscreteLogarithmEnv, DiscreteLogarithmSeed, SYSTEM_PROMPT, generate_seed
from trl_env.tokenizer import TransformerTokenizer


if sys.platform == "darwin":
    import mlx_lm
    import mlx_lm.sample_utils

    class MlxDecoder(RolloutDecoder):
        def __init__(self,
            model_path: str,
            temperature: float,
            eos_token_set: set[int],
            max_completion_length: int,
        ):
            model, tokenizer, config = mlx_lm.load(  # type: ignore
                path_or_hf_repo=model_path,
                return_config=True,
            )
            self.model = model
            self.tokenizer = tokenizer

            self.temperature = temperature
            self.max_completion_length = max_completion_length

            if len(eos_token_set) > 0:
                print("WARNING: eos_token_set is not supported in mlx_lm")
        
        def generate(self, input_ids: list[int]) -> tuple[list[int], list[float]]:
            response_generator = mlx_lm.stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=input_ids,
                max_tokens=self.max_completion_length,
                sampler=mlx_lm.sample_utils.make_sampler(
                    temp=self.temperature,
                ),
                logits_processors=mlx_lm.sample_utils.make_logits_processors(),
            )
            completions_ids: list[int] = []
            logprobs: list[float] = []
            for response in response_generator:
                token = response.token
                logprob = float(response.logprobs[token].item())
                completions_ids.append(token)
                logprobs.append(logprob)
            
            return completions_ids, logprobs
    
    decoder_class = MlxDecoder

else:
    from trl_env.decoder_transformer import TransformerDecoder
    def transformer_decoder(model_path: str, temperature: float, max_completion_length: int):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

        return TransformerDecoder(
            model=model,
            temperature=temperature,
            eos_token_set={tokenizer.eos_token_id},
            max_completion_length=max_completion_length,
        )
    decoder_class = transformer_decoder


def logger(role: str, content: str):
    print(f"## {role.upper()} ##############")
    print(content)


def main(model_path: str, bit_size: int):
    if "instruct" in model_path:
        processor = qwen3_instruct_processor
    else:
        processor = qwen3_processor

    max_turn_length = 32768
    max_conversation_length = 32768

    t = AutoTokenizer.from_pretrained(model_path)

    tokenizer = TransformerTokenizer(t)

    decoder = decoder_class(
        model_path=model_path,
        temperature=0.6,
        max_completion_length=max_turn_length,
    )

    system_prompt = SYSTEM_PROMPT.format(
        max_turn_length=max_turn_length,
        max_conversation_length=max_conversation_length,
    )

    rollout(
        processor=processor,
        tokenizer=tokenizer,
        decoder=decoder,
        env=DiscreteLogarithmEnv(),
        seed=generate_seed(bit_size).model_dump_json(),
        system_prompt=system_prompt,
        max_conversation_length=max_conversation_length,
        conversation_timer=lambda length: f"current conversation length {length}",
        conversation_logger=logger,
    )

if __name__ == "__main__":
    bit_size = 32
    if len(sys.argv) > 2:
        bit_size = int(sys.argv[2])
    main(sys.argv[1], bit_size)