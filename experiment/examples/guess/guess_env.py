def extract_last_integer(s: str) -> int | None:
    matches = re.findall(r'-?\d+', s)
    if not matches:
        return None
    try:
        last = int(matches[-1])
    except ValueError:
        last = None # integer might be too big for int

    return last

def process_action(seed: int, action: Action) -> tuple[float, bool, str]:
    guess = extract_last_integer(action)
    if guess is None:
        return 0.0, False, "no integer found"

    f = lambda x: 1 / (1 + x)
    format_points = f(jiwer.cer(str(guess), action))

    if guess < 1 or 1000 < guess:
        action_points = 0
        delta = "guess should be within [1, 1000]"
        alive = True
    else:
        action_points = f((guess - seed)**2)
        if guess < seed:
            delta = "too low"
            alive = True
        elif guess > seed:
            delta = "too high"
            alive = True
        else:
            delta = "correct"
            alive = False
    
    points = 0.3 * format_points + 0.7 * action_points
    return points, alive, delta

SYSTEM_PROMPT = """
every turn, you can output a maximum number of {max_turn_length} tokens
the whole conversation should not last longer than {max_conversation_length} tokens
"""

class GuessEnv(Env):
    def reset(self, seed: str) -> tuple[Env, Delta]:
        self.reward = 0
        self.best_points = 0
        self.alive = True
        self.step_count = 0

        self.seed: int = int(seed)
    
        return self, f"""
I have an integer from 1 to 1000 (inclusive) in mind, you have to guess that number
Only respond with the number, no explanation needed. I will let you know if
your guess is higher or lower than my number
"""
    
    def step(self, action: Action) -> tuple[Env, Delta]:
        assert self.seed is not None

        self.step_count += 1

        points, alive, delta = process_action(self.seed, action)
        
        gamma = 0.99
        self.alive = alive
        self.best_points = max(points, self.best_points)
        self.reward = self.best_points * gamma**self.step_count
        
        return self,  delta

import random

def generate_seed() -> int:
    return random.randint(1, 1000)