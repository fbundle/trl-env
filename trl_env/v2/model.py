from __future__ import annotations
from typing import Protocol

class RolloutModel(Protocol):
    def generate(self, input_ids: list[int]) -> tuple[list[int], list[float]]: ...
