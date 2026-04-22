from typing import Protocol

type Action = str
type Delta = str
type Seed = str

class Env(Protocol):
    last_step_reward: float
    alive: bool
    def reset(self, seed: Seed) -> Delta: ...
    def step(self, action: Action) -> Delta: ...
