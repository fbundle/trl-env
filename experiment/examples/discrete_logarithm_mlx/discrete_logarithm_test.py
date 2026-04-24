
import torch
from transformers import AutoModelForCausalLM

from trl_env.rollout import batch_rollout
from trl_env.rollout_mlx import MlxRolloutEngine
from trl_env.processor import qwen3_instruct_processor

from experiment.examples.discrete_logarithm_mlx.discrete_logarithm_env import EXTRA_EOS_TOKEN_LIST, DiscreteLogarithmEnv, DiscreteLogarithmSeed, SYSTEM_PROMPT
import mlx.core as mx

def logger(i: int, role: str, content: str):
    print(f"{role}> {content}")

def main():
    model_path = "Qwen/Qwen3.5-0.8B"
    processor = qwen3_instruct_processor

    max_turn_length = 512
    max_conversation_length = 4096

    engine = MlxRolloutEngine(
        model_path=model_path,
        max_completion_length=max_turn_length,
        temperature=0.6,
        extra_eos_token_list=EXTRA_EOS_TOKEN_LIST,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = model.cpu()
    state_dict = {k: mx.array(v) for k, v in model.state_dict().items()}

    engine.update_weights_and_reset_prompt_cache(state_dict)

    system_prompt = SYSTEM_PROMPT.format(
        max_turn_length=max_turn_length,
        max_conversation_length=max_conversation_length,
    )

    with torch.no_grad():
        o = batch_rollout(
            engine=engine, processor=processor,
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