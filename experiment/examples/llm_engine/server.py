import os
import sys
from typing import Iterator, Callable

from fastapi import FastAPI, Header, HTTPException
from fastapi.sse import EventSourceResponse
from moka_py import Moka

from .api import ChatCompletionRequest, ChatCompletionChunk, ChatCompletionChoice, GEMMA_CONSUMER, QWEN_CONSUMER, \
    TRANSFORMER_ENGINE, MLX_ENGINE, parse_model_path, GGUF_ENGINE, RAW_CONSUMER
from .consumer import ChatCompletionConsumer, GemmaChatCompletionConsumer, QwenChatCompletionConsumer, \
    RawChatCompletionConsumer
from .engine import Engine, TransformerEngine, MlxEngine, GgufEngine


def split_iter(sep: str, iter: Iterator[str]) -> Iterator[str]:
    for chunk in iter:
        parts = chunk.split(sep)
        yield parts[0]
        for part in parts[1:]:
            yield sep
            yield part



chat_completion_consumer_dict: dict[str, Callable[[], ChatCompletionConsumer]] = {
    RAW_CONSUMER: RawChatCompletionConsumer,
    GEMMA_CONSUMER: GemmaChatCompletionConsumer,
    QWEN_CONSUMER: QwenChatCompletionConsumer,
}

engine_factory_dict: dict[str, Callable[[str], Engine]] = {
    TRANSFORMER_ENGINE: TransformerEngine,
    MLX_ENGINE: MlxEngine,
    GGUF_ENGINE: GgufEngine,
}


class StreamerApp:
    fastapi: FastAPI
    engine_dict: Moka[str, Engine]

    def __init__(self, *args, **kwargs):
        self.fastapi = FastAPI(*args, **kwargs)
        self.engine_dict = Moka(capacity=10)  # maximum 10 models

        self.fastapi.router.api_route(
            path="/v1/chat/completions",
            methods=["POST"],
            response_class=EventSourceResponse,
        )(self.chat_completion)

    def chat_completion(
            self,
            r: ChatCompletionRequest,
            authorization: str | None = Header(default=None),
    ) -> Iterator[ChatCompletionChunk]:
        expected_key = os.environ.get("LEANLM_API_KEY")
        if expected_key:
            if authorization != f"Bearer {expected_key}":
                raise HTTPException(status_code=401, detail="unauthorized")

        if not r.stream:
            raise HTTPException(status_code=400, detail="only support stream=True")

        engine_type, consumer_type, model_path = parse_model_path(r.model)

        if consumer_type not in chat_completion_consumer_dict:
            raise HTTPException(status_code=400, detail=f"consumer type {consumer_type} not supported")

        if engine_type not in engine_factory_dict:
            raise HTTPException(status_code=400, detail=f"engine {engine_type} not supported")

        consumer = chat_completion_consumer_dict[consumer_type]()
        engine = self.engine_dict.get_with(
            key=r.model,
            initializer=lambda: engine_factory_dict[engine_type](model_path),
            tti=60 * 10,  # evict after 10 minutes of inactivity
            ttl=60 * 60 * 24,  # keep models for a maximum of 24 hours
        )

        chunk_iter = engine.chat(
            messages=r.messages,
            config=r.generate_config,
        )

        for token in consumer.split_tokens():
            chunk_iter = split_iter(token, chunk_iter)

        for chunk in chunk_iter:
            print(chunk, end="", flush=True, file=sys.stderr)

            delta, ok = consumer.consume(chunk)
            if not ok:
                break
            if delta is not None:
                yield ChatCompletionChunk(
                    choices=[ChatCompletionChoice(
                        delta=delta,
                        finish_reason=None,
                    )],
                )


if __name__ == "__main__":
    app = StreamerApp(debug=True)

    import uvicorn

    print("docs at http://127.0.0.1:3000/docs")
    uvicorn.run(app.fastapi, host="127.0.0.1", port=3000)
