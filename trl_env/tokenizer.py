from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_list: list[int]) -> str: ...

class TransformerTokenizer(Tokenizer):
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
    
    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text).input_ids

    def decode(self, token_list: list[int]) -> str:
        text = self.tokenizer.decode(token_list)
        assert isinstance(text, str)
        return text



