
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl_env.v2.decoder_transformer import TransformerRolloutDecoder
from trl_env.v2.rollout import rollout
from trl_env.v2.processor import qwen3_instruct_processor

from experiment.examples.discrete_logarithm_v2.discrete_logarithm_env import EXTRA_EOS_TOKEN_LIST, DiscreteLogarithmEnv, DiscreteLogarithmSeed, SYSTEM_PROMPT
from trl_env.v2.tokenizer import TransformerTokenizer


def logger(role: str, content: str):
    print(f"{role}> {content}")

def main():
    model_path = "Qwen/Qwen3.5-0.8B"
    processor = qwen3_instruct_processor

    max_turn_length = 1024
    max_conversation_length = 4096

    t = AutoTokenizer.from_pretrained(model_path)

    tokenizer = TransformerTokenizer(t)

    eos_token_set = {t.eos_token_id}
    eos_token_set.update([tokenizer.encode(eos_token)[0] for eos_token in EXTRA_EOS_TOKEN_LIST])

    decoder = TransformerRolloutDecoder(
        model=AutoModelForCausalLM.from_pretrained( # type: ignore
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval(), 
        temperature=0.6,
        eos_token_set=eos_token_set,
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
        seed=DiscreteLogarithmSeed(
            g=2, h=3, p=5,
        ).model_dump_json(),
        system_prompt=system_prompt,
        max_conversation_length=max_conversation_length,
        conversation_logger=logger,
    )



if __name__ == "__main__":
    main()