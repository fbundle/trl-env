import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl_env.batch_rollout import batch_rollout
from trl_env.model import TransformerModel
from trl_env.processor import qwen3_instruct_processor, qwen3_processor

from experiment.examples.discrete_logarithm_env import DiscreteLogarithmEnv, DiscreteLogarithmSeed, SYSTEM_PROMPT


def logger(i: int, role: str, content: str):
    print(f"{role}> {content}")

def main():
    model_path = "Qwen/Qwen3.5-0.8B"
    processor = qwen3_processor

    max_turn_length = 512
    max_conversation_length = 4096

    model = TransformerModel(
        tokenizer=AutoTokenizer.from_pretrained(model_path),
        model=AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval(),
        generation_kwargs=dict(
            temperature=0.6,
            max_new_tokens=max_turn_length,
        )
    )


    system_prompt = SYSTEM_PROMPT.format(
        max_turn_length=max_turn_length,
        max_conversation_length=max_conversation_length,
    )

    with torch.no_grad():
        o = batch_rollout(
            model=model, processor=processor,
            env_factory=lambda : DiscreteLogarithmEnv(), seed_list=[DiscreteLogarithmSeed(
                g=2, h=3, p=5,
            ).model_dump_json()],
            system_prompt=system_prompt,
            max_conversation_length=max_conversation_length,
            conversation_logger=logger,
        )
        print(o[0].reward)

if __name__ == "__main__":
    main()