from __future__ import annotations
from typing import Protocol

from transformers import Trainer


class RolloutDecoder(Protocol):
    def generate(self, input_ids: list[int]) -> tuple[list[int], list[float]]: ...

class RolloutDecoderFactory(Protocol):
    def make_decoder(self, trainer: Trainer) -> RolloutDecoder: ...
