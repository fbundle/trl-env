



from trl_env.engine import Engine


class MlxEngine(Engine):
    def __init__(self) -> None:
        pass

    def tokenizer_encode(self, input_text: str) -> list[int]: ...
    def tokenizer_decode(self, completion_ids: list[int]) -> str: ...
    def model_batch_generate(self, input_ids_list: list[list[int]]) -> tuple[list[list[int]], list[list[float]]]: ...
