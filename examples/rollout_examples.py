import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl_env.batch_rollout import batch_rollout
from trl_env.environment import Action, Delta, Env, Seed
from trl_env.model import TransformerModel
from trl_env.processor import qwen3_instruct_processor


class GuessEnv(Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def reset(self, seed: Seed) -> Delta:
        self.target = int(seed)
        self.reward = 0
        self.alive = True
        return """
I have an integer between 0 and 50 in mind
every turn, you have to take a guess, output
GUESS <number>
I will say if your guess is higher or lower than my number
"""
    
    def step(self, action: Action) -> Delta:
        def helper(action: str) -> tuple[float, float, bool, str]:
            words = action.split()
            if "GUESS" not in words:
                return 0, 0, False, f"can't find your guess"
            
            guess_str = words[words.index("GUESS") + 1]

            try:
                guess = int(guess_str)
            except ValueError:
                guess = None

            if guess is None:
                return 0.5, 0, False, f"can't find the number in your guess"

            f = lambda x: 1 / (1 + x) # map [0, inf) -> [1, 0)
            number_points = f(abs(self.target - guess))
            alive = True
            if guess < self.target:
                state_delta = f"{guess} is too low"
            elif guess > self.target:
                state_delta = f"{guess} is too high"
            else:
                state_delta = f"{guess} is correct"
                alive = False
                  
            return 1.0, number_points, alive, state_delta

        format_points, number_points, alive, state_delta = helper(action)
        self.alive = alive
        points = format_points + number_points
        if points > self.reward: # reward = best
            self.reward = points
        return state_delta

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
            env_factory=GuessEnv, seed_list=["36"],
            system_prompt=system_prompt,
            max_conversation_length=max_conversation_length,
            log=sys.stdout.write, # type: ignore
        )
        print(o[0].reward)

if __name__ == "__main__":
    main()