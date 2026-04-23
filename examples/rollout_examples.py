import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl_env.batch_rollout import batch_rollout
from trl_env.environment import Action, Delta, Env, GuessEnv, Seed
from trl_env.model import TransformerModel
from trl_env.processor import qwen3_instruct_processor


rule = """
every turn, you can output a maximum number of {max_turn_tokens} tokens
the whole conversation should not last longer than {max_conversation_length} tokens
"""

def main():
    model_path = "Qwen/Qwen3.5-0.8B"
    processor = qwen3_instruct_processor

    max_turn_length = 64
    max_conversation_length = 512

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


    system_prompt = rule.format(
        max_turn_tokens=max_turn_length,
        max_conversation_length=max_conversation_length,
    )

    with torch.no_grad():
        o = batch_rollout(
            model=model, processor=processor,
            env_factory=lambda : GuessEnv(0, 50), seed_list=["36"],
            system_prompt=system_prompt,
            max_conversation_length=max_conversation_length,
            log=sys.stdout.write, # type: ignore
        )
        print(o[0].reward)

if __name__ == "__main__":
    main()