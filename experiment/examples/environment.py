from trl_env.environment import Action, Delta, Env, Seed

# GuessEnv
# reward must be dense enough in the space of all texts so that it would learn easier

import re

def extract_last_integer(s: str) -> int | None:
    matches = re.findall(r'-?\d+', s)
    return int(matches[-1]) if matches else None

class GuessEnv(Env):
    def __init__(self, MIN: int, MAX: int, **kwargs):
        super().__init__(**kwargs)
        self.initial_delta = f"""
I have an integer between {MIN} and {MAX} in mind
Every turn, you have to take a guess.
I will say if your guess is higher or lower than my number
"""
    
    def reset(self, seed: Seed) -> Delta:
        self.target = int(seed)
        self.turn = 0
        self.best_points = 0
        self.reward = 0
        self.alive = True
        return self.initial_delta
    
    def step(self, action: Action) -> Delta:
        def helper(action: str) -> tuple[float, float, bool, str]:
            guess = extract_last_integer(action)

            if guess is None:
                return 0.0, 0.0, False, f"can't find the number in your guess"

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
        points = format_points + number_points
        
        self.turn += 1
        self.alive = alive
        self.best_points = max(self.best_points, points)
        self.reward = self.best_points * (0.99)**(self.turn)

        return state_delta
    

# PrimeFactorEnv
from pydantic import BaseModel
import re

def parse_tool_call(s: str) -> tuple[str, str] | None:
    match = re.search(r'<tool_call tool="(\w+)">(.*?)</tool_call>', s, re.DOTALL)
    if match is None:
        return None
    return match.group(1), match.group(2).strip()

assert parse_tool_call('<tool_call tool="mini_racer"> code </tool_call>') == ("mini_racer", "code")
assert parse_tool_call('some text <tool_call tool="js"> x + 1 </tool_call> more') == ("js", "x + 1")
assert parse_tool_call("no tool call here") is None

class PrimeFactorSeed(BaseModel):
    pass

class PrimeFactorEnv:
    pass