import os
import sys
from typing import Iterator

from experiment.examples.llm_engine.engine import MlxEngine, TransformerEngine
from experiment.examples.llm_engine.api import ChatCompletionGenerateConfig, Message

def is_mlx_checkpoint(path: str) -> bool: # type: ignore
    if os.path.exists(os.path.join(path, "README.md")):
        mlx: bool = False
        for line in open(os.path.join(path, "README.md")):
            line = line.strip()
            parts = line.split(":")
            if len(parts) == 2:
                if parts[0].strip() == "library_name" and parts[1].strip() == "mlx":
                    mlx = True
                    break
        return mlx

def split_token(i: Iterator[str], sep: str) -> Iterator[str]:
    for text in i:
        parts = text.split(sep)
        yield parts[0]
        for part in parts[1:]:
            yield sep
            yield part

def main(checkpoint_path: str):
    if is_mlx_checkpoint(checkpoint_path):
        print("LOADING MLX CHECKPOINT ...")
        engine = MlxEngine(checkpoint_path)
    else:
        print("LOADING TRANSFORMER CHECKPOINT ...")
        # full finetuning
        engine = TransformerEngine(checkpoint_path)
        engine.model = engine.model.to("mps") # type: ignore


    messages = []
    while True:
        question = input("user> ")
        messages.append(Message(
            role="user",
            content=question,
        ))

        chat = engine.chat(messages=messages, config=ChatCompletionGenerateConfig(
            max_completion_tokens=131072,
            temperature=0.6,
            top_p=0.95,
            min_p=0.0,
            top_k=20,
            repetition_penalty=1.1,
        ))
        print("agent> ", end="")
        for content in chat:
            print(content, end="", flush=True)
        print()

if __name__ == "__main__":
    main(sys.argv[1])
