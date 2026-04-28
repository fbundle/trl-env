from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_list: list[int]) -> str: ...

class TransformerTokenizer(Tokenizer):
    def __init__(self, processing_class) -> None:
        self.processing_class = processing_class
    
    def encode(self, text: str) -> list[int]:
        return self.processing_class(text).input_ids

    def decode(self, token_list: list[int]) -> str:
        text = self.processing_class.decode(token_list)
        assert isinstance(text, str)
        return text



