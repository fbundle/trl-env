
from dataclasses import dataclass
from typing import Literal, Any

from pydantic import BaseModel
from py_mini_racer import MiniRacer
import jiwer

from trl_env.environment import Action, Delta, Env, Seed

import re

def extract_last_natural(s: str) -> int | None:
    matches = re.findall(r'-?\d+', s)
    if not matches:
        return None
    last = int(matches[-1])
    return last if last >= 0 else None

@dataclass
class ParsedAction:
    action_type: Literal["tool_call", "answer", None]
    action_value: str = ""
    format_points: float = 0 # [0, 1]

f = lambda x: 1 / (1 + x)

def parse_action(action: str) -> ParsedAction:
    parts = action.split("<tool_call>", maxsplit=1)
    if len(parts) >= 2:
        action_value = parts[1]
        format_points = f(jiwer.cer(f"<tool_call>{action_value}", action))
        return ParsedAction(
            action_type="tool_call",
            action_value=action_value,
            format_points=format_points,
        )

    last_natural = extract_last_natural(action)
    if last_natural is not None:
        action_value = str(last_natural)
        format_points = f(jiwer.cer(action_value, action))
        return ParsedAction(
            action_type="answer",
            action_value=action_value,
            format_points=format_points,
        )
    return ParsedAction(
        action_type=None,
        format_points=0.0,
    )

EXTRA_EOS_TOKEN_LIST = []

import multiprocessing as mp
import queue

def _mini_racer_eval(result_queue: mp.Queue, code: str, timeout_sec: float, max_memory_bytes: int):
    try:
        from py_mini_racer import MiniRacer
        mc = MiniRacer()
        result: Any = mc.eval(code=code, timeout_sec=timeout_sec, max_memory=max_memory_bytes)

        ok, result_str = True, str(result)
    except Exception as e:
        ok, result_str = False, str(e)
    finally:
        result_queue.put((ok, result_str))

def safe_eval_js(code: str, timeout_sec: float = 1.0, max_memory_bytes: int = 256 * 1024 * 1024) -> tuple[bool, str]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_mini_racer_eval, args=(result_queue, code, timeout_sec, max_memory_bytes))
    proc.start()
    
    try:
        ok, result_str = result_queue.get(block=True, timeout=timeout_sec + 1) # extra second for process communication
    except queue.Empty:
        ok, result_str = False, "process returned nothing"
    except Exception as e:
        ok, result_str = False, str(e)

    if proc.is_alive():
        proc.kill()
    
    proc.join()

    return ok, result_str

def process_action(g: int, h: int, p: int, action: str) -> tuple[float, bool, str]:
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
        ok, result_str = safe_eval_js(code=a.action_value, timeout_sec=1.0, max_memory_bytes=256 * 1024 * 1024) # 1 second, 256MB
        if ok:
            # 0.3 point for code ok
            action_points = 0.3
        else:
            # 0.2 point for compile error
            action_points = 0.2

        delta = result_str[:256]
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

        self.seed: DiscreteLogarithmSeed | None = None
    
    def reset(self, seed: Seed) -> tuple[Env, Delta]:
        self.reward = 0
        self.alive = True
        self.step_count = 0

        self.seed = DiscreteLogarithmSeed.model_validate_json(seed)

        # TODO - consider input the source code of the environment into the first prompt
        g, h, p = self.seed.g, self.seed.h, self.seed.p
        return self, f"""
Find x such that {g}^x = {h} (mod {p}), this is the discrete logarithm problem
You are allow to use javascript by ending your response by using tool call. For example

<tool_call> function your_function(your_params) {{ your_code }}; your_function(your_args)


I will run that code in a V8 engine with a timeout of 1 seconds and 256 MB max memory and tell you the return value.
If you are confident with your answer, just output the answer without any explanation.
Note that, answer should be in (mod {p}). Once the answer is given, the environment is terminated.
"""
    def step(self, action: Action) -> tuple[Env, Delta]:
        assert self.seed is not None
        g, h, p = self.seed.g, self.seed.h, self.seed.p

        self.step_count += 1

        points, alive, delta = process_action(
            g=g,
            h=h,
            p=p,
            action=action,
        )

        self.alive = alive
        self.best_points = max(points, self.best_points)
        self.reward = self.best_points * 0.95**self.step_count
        
        return self,  delta

if __name__ == "__main__":
    print(open(__file__).read())