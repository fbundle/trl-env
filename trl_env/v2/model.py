from __future__ import annotations
from typing import Protocol

class RolloutModel(Protocol):
    def make_context(self) -> RolloutModelContext: ...

class RolloutModelContext(Protocol):
    def generate(self, input_ids: list[int]) -> tuple[list[int], list[float]]: ...
