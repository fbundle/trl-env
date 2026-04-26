
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from py_mini_racer import MiniRacer
import jiwer

from trl_env.environment import Action, Delta, Env, Seed

@dataclass
class ParsedAction:
    action_type: Literal["tool_call", "answer", None]
    action_value: str = ""
    format_points: float = 0 # [0, 1]

f = lambda x: 1 / (1 + x)

def parse_action(action: str) -> ParsedAction:
    parts = action.split("<|box_start|>")
    if len(parts) >= 2:
        action_value = parts[1].split("<|box_end|>")[0]
        format_points = f(jiwer.cer(f"<|box_start|>{action_value}<|box_end|>", action))
        return ParsedAction(
            action_type="answer",
            action_value=action_value,
            format_points=format_points,
        )

    parts = action.split("<tool_call>")
    if len(parts) >= 2:
        action_value = parts[1].split("</tool_call>")[0]
        format_points = f(jiwer.cer(f"<tool_call>{action_value}</tool_call>", action))
        return ParsedAction(
            action_type="tool_call",
            action_value=action_value,
            format_points=format_points,
        )
    
    return ParsedAction(
        action_type=None,
        format_points=0.0,
    )

EXTRA_EOS_TOKEN_LIST = ["</tool_call>", "<|box_end|>"]

f = lambda x: 1 / (1 + x)

def process_action(g: int, h: int, p: int, mini_racer: MiniRacer, action: str) -> tuple[float, bool, str]:
    a = parse_action(action)

    format_points = a.format_points

    if a.action_type == "answer":
        try:
            x = int(a.action_value)
        except ValueError:
            x = None

        if x is None:
            # zero points for no answer
            # stop immediately
            action_points = 0.0
            alive = False
            delta = f"integer not found, found {a.action_value}"
        else:
            h_ans = pow(g, x, p)
            if h_ans != h:
                # 0.5 point for wrong answer
                # stop immediately
                action_points = 0.5
                alive = False
                delta = f"wrong answer expected {h} got {g}^{x} = {h_ans} (mod {p})"
            else:
                # 1.0 point for correct answer
                # stop immediately
                action_points = 1.0
                alive = False
                delta = "correct answer"
    elif a.action_type == "tool_call":
        try:
            result = mini_racer.eval(code=a.action_value, timeout=1000, max_memory=50 * 1024 * 1024)  # 1s, 50MB
            result_str = str(result)
            # 0.3 point for compile ok
            action_points = 0.3
        except Exception as e:
            # 0.2 point for compile error
            result_str = str(e)
            action_points = 0.2
        
        delta = f"## INPUT ##\n{a.action_value}\n## OUTPUT ##\n{result_str[:256]}"
        alive = True
    else:
        # nothing detected
        # zero points for wrong format
        # stop immediately
        action_points = 0.0
        alive = False
        delta = "no tool or answer is detected"

    total_points = 0.3 * format_points + 0.7 * action_points
    return total_points, alive, delta

SYSTEM_PROMPT = """
every turn, you can output a maximum number of {max_turn_length} tokens
the whole conversation should not last longer than {max_conversation_length} tokens
"""

class DiscreteLogarithmSeed(BaseModel):
    """
    find x such that g^x = h (mod p)
    """
    g: int
    h: int
    p: int

class DiscreteLogarithmEnv(Env):
    def __init__(self) -> None:
        super().__init__()
        self.source = open(__file__).read()
        self.reward = 0

        self.best_points = 0
        self.alive = False
        self.step_count = 0

        self.mini_racer: MiniRacer = MiniRacer()
        self.seed: DiscreteLogarithmSeed | None = None
    
    def reset(self, seed: Seed) -> tuple[Env, Delta]:
        self.reward = 0
        self.alive = True
        self.step_count = 0

        self.mini_racer = MiniRacer()
        self.seed = DiscreteLogarithmSeed.model_validate_json(seed)

        # TODO - consider input the source code of the environment into the first prompt
        return self, f"""
Find x such that {self.seed.g}^x = {self.seed.h} (mod {self.seed.p}), this is the discrete logarithm problem
You are allow to use javascript by writing

<tool_call>your javascript code here</tool_call>

I will run that code in a V8 engine with a timeout of 1 seconds and 50 MB max memory.
If you are confident with your answer, write

<|box_start|>answer<|box_end|>

Note that, only the first match is consider. Once the answer is given, the environment is terminated.
"""
    def step(self, action: Action) -> tuple[Env, Delta]:
        assert self.seed is not None
        g, h, p = self.seed.g, self.seed.h, self.seed.p

        self.step_count += 1

        points, alive, delta = process_action(
            g=g,
            h=h,
            p=p,
            mini_racer=self.mini_racer,
            action=action,
        )

        self.alive = alive
        self.best_points = max(points, self.best_points)
        self.reward = self.best_points * 0.999**self.step_count
        
        return self,  delta

if __name__ == "__main__":
    print(open(__file__).read())