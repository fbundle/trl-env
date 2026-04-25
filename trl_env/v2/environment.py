from __future__ import annotations
from typing import Protocol

type Action = str
type Delta = str
type Seed = str

class Env(Protocol):
    reward: float
    alive: bool
    def reset(self, seed: Seed) -> tuple[Env, Delta]: ...
    def step(self, action: Action) -> tuple[Env, Delta]: ...
