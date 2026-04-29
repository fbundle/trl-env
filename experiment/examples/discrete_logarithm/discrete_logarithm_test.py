
import sys

import numpy as np

from transformers import AutoTokenizer

from trl_env.decoder import RolloutDecoder
from trl_env.rollout import rollout
from trl_env.processor import qwen3_processor

from experiment.examples.discrete_logarithm.discrete_logarithm_env import EXTRA_EOS_TOKEN_LIST, DiscreteLogarithmEnv, DiscreteLogarithmSeed, SYSTEM_PROMPT
from trl_env.tokenizer import TransformerTokenizer

import mlx_lm
import mlx_lm.sample_utils

class MlxDecoder(RolloutDecoder):
    def __init__(self,
        model_path: str,
        temperature: float,
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


def logger(role: str, content: str):
    print(f"## {role.upper()} ##############")
    print(content)

def generate_seed(bit_size: int = 6) -> str:
    # find a prime p
    p: int = np.random.randint(2**(bit_size-1), 2**bit_size)
    p: int = sympy.nextprime(p)             # type: ignore
    # sample g and x
    g = np.random.randint(2, p)
    x = np.random.randint(1, p)
    h = pow(g, x, p)
    return DiscreteLogarithmSeed(g=g, h=h, p=p).model_dump_json()

def main(model_path: str):
    processor = qwen3_processor

    max_turn_length = 8192
    max_conversation_length = 8192

    t = AutoTokenizer.from_pretrained(model_path)

    tokenizer = TransformerTokenizer(t)

    decoder = MlxDecoder(
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
        seed=generate_seed(),
        system_prompt=system_prompt,
        max_conversation_length=max_conversation_length,
        conversation_logger=logger,
    )

if __name__ == "__main__":
    main(sys.argv[1])