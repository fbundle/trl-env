


from typing import Union

import mlx.nn as nn
import mlx_lm
from mlx_lm.tokenizer_utils import TokenizerWrapper
import mlx.core as mx
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from trl_env.engine import Engine


class MlxEngine(Engine):
    def __init__(self, model_path: str) -> None:
        model, _, _ = mlx_lm.load(  # type: ignore
            path_or_hf_repo=model_path,
            return_config=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model: nn.Module = model
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
    
    def load_weights(self, file_or_weights: Union[str, list[tuple[str, mx.array]]]):
        self.model.load_weights(file_or_weights)

    def tokenizer_encode(self, input_text: str) -> list[int]:
        return self.tokenizer(input_text).input_ids

    def tokenizer_decode(self, completion_ids: list[int]) -> str:
        output_text = self.tokenizer.decode(completion_ids)
        assert isinstance(output_text, str)
        return output_text
    
    def model_batch_generate(self, input_ids_list: list[list[int]]) -> tuple[list[list[int]], list[list[float]]]:
        # TODO
        ...


