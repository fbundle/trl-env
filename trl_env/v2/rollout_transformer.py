
from typing import Any, Callable

from trl_env.v2.environment import Env
from trl_env.v2.model import RolloutModel
from trl_env.v2.processor import Processor
from trl_env.v2.rollout import batch_rollout
from trl_env.v2.tokenizer import Tokenizer

from trl.trainer.grpo_trainer import RolloutFunc, GRPOTrainer, RewardFunc

def make_rollout_func(
    processor: Processor, tokenizer: Tokenizer,
    model_factory: Callable[[], RolloutModel], env_factory: Callable[[], Env],
    system_prompt: str, max_conversation_length: int,
) -> RolloutFunc:
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, Any]:
        state_list = batch_rollout(
            processor=processor, tokenizer=tokenizer,
            model_factory=model_factory, env_factory=env_factory,
            system_prompt=system_prompt, max_conversation_length=max_conversation_length,
            seed_list=prompts,
        )

        
        return {
            "prompt_ids": [state.conversation[:state.initial_length] for state in state_list],
            "completion_ids": [state.conversation[state.initial_length:] for state in state_list],
            "env_mask": [state.env_mask for state in state_list],
            "logprobs": [state.logprobs for state in state_list],
            "reward": [state.reward for state in state_list],
        }

    return rollout_func

def make_reward_func() -> RewardFunc:
    def reward_func(prompts: list[str], completions: list[str], reward: list[float], **kwargs) -> list[float]:
            return reward
    return reward_func # type: ignore