
from pydantic import BaseModel
from py_mini_racer import MiniRacer
import jiwer

from trl_env.environment import Action, Delta, Env, Seed

import re

def parse_tool_call(s: str) -> str | None:
    match = re.search(r'<tool_call>(.*?)</tool_call>', s, re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()

def parse_answer(s: str) -> str | None:
    match = re.search(r'<\|box_start\|>(.*?)<\|box_end\|>', s, re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()

def format_tool_call(js_code: str) -> str:
    return f"<tool_call>{js_code}</tool_call>"

def format_answer(answer: str) -> str:
    return f"<|box_start|>{answer}<|box_end|>"

assert parse_tool_call('<tool_call>console.log(1)</tool_call>') == "console.log(1)"
assert parse_tool_call('<tool_call>\nconsole.log(1)\n</tool_call>') == "console.log(1)"
assert parse_tool_call("no tool call") is None
assert parse_answer('<|box_start|> 42 <|box_end|>') == "42"
assert parse_answer("no answer") is None
assert format_tool_call("console.log(1)") == "<tool_call>console.log(1)</tool_call>"
assert format_answer("42") == "<|box_start|>42<|box_end|>"


EXTRA_EOS_TOKEN_LIST = ["</tool_call>", "<|box_end|>"]

f = lambda x: 1 / (1 + x)

def process_action(g: int, h: int, p: int, mini_racer: MiniRacer, cap: int, action: str) -> tuple[float, bool, str]:
    answer_str = parse_answer(action)
    if answer_str is not None:
        format_points = f(jiwer.cer(format_answer(answer_str), action))

        try:
            x = int(answer_str)
        except ValueError:
            x = None

        

        if x is None:
            # zero points for no answer
            # stop immediately
            return 0.0 + format_points, False, f"integer not found, found {answer_str}"
        else:
            h_ans = pow(g, x, p)
            if h_ans != h:
                # 0.5 point for wrong answer
                # stop immediately
                return 0.5 + format_points, False,  f"wrong answer expected {h} got {g}^{x} = {h_ans} (mod {p})"
            else:
                # 1.0 point for correct answer
                # stop immediately
                return 1.0 + format_points, False, f"correct answer"
    
    code_str = parse_tool_call(action)
    if code_str is not None:
        format_points = f(jiwer.cer(format_tool_call(code_str), action))

        try:
            result = mini_racer.eval(code=code_str, timeout=1000, max_memory=50 * 1024 * 1024)  # 1s, 50MB
            result_str = str(result)
        except Exception as e:
            result_str = str(e)
        
        result_str = result_str[:cap] # cap the output

        # 0.3 point for knowing how to use tool
        # keep going
        return 0.3 + format_points, True, f"## INPUT ##\n{code_str}\n## OUTPUT ##\n{result_str}"

    # nothing detected
    # zero points for wrong format
    # stop immediately
    return 0.0, False, "no tool or answer is detected"

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

<tool_call>
your code here
</tool_call>

I will run that code in a V8 engine with a timeout of 1 seconds and 50 MB max memory.
If you are confident with your answer, write
<|box_start|> answer <|box_end|>

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
            cap=256,
            action=action,
        )

        self.alive = alive
        self.best_points = max(points, self.best_points)
        self.reward = self.best_points * 0.999**self.step_count
        
        return self,  delta

if __name__ == "__main__":
    print(open(__file__).read())