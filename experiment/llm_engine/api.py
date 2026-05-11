from typing import Literal

from pydantic import BaseModel

# a subset of OpenAI chat completion API streaming mode

# request

type Role = Literal["user", "system", "assistant"]
ROLE_USER: Role = "user"
ROLE_SYSTEM: Role = "system"
ROLE_ASSISTANT: Role = "assistant"


class Message(BaseModel):
    role: Role = ROLE_USER
    content: str  # TODO - make this include other data type like images, videos


class ChatCompletionGenerateConfig(BaseModel):
    max_completion_tokens: int = 4096

    temperature: float = 1.0
    top_p: float = 0.95
    min_p: float = 0.0
    top_k: int = 64

    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.1


type ChatCompletionEngine = Literal["transformer", "mlx", "gguf"]
TRANSFORMER_ENGINE: ChatCompletionEngine = "transformer"
MLX_ENGINE: ChatCompletionEngine = "mlx"
GGUF_ENGINE: ChatCompletionEngine = "gguf"

type ChatCompletionConsumerType = Literal["raw", "gemma", "qwen"]
RAW_CONSUMER: ChatCompletionConsumerType = "raw"
GEMMA_CONSUMER: ChatCompletionConsumerType = "gemma"
QWEN_CONSUMER: ChatCompletionConsumerType = "qwen"


class ChatCompletionRequest(BaseModel):
    messages: list[Message]

    stream: bool = True
    model: str = f"{MLX_ENGINE}:{QWEN_CONSUMER}:mnt/output_mlx/Qwen3.5-0.8B"
    generate_config: ChatCompletionGenerateConfig = ChatCompletionGenerateConfig()


def parse_model_path(model: str) -> tuple[str, str, str]:
    parts = model.split(":", maxsplit=2)

    DEFAULT_ENGINE: ChatCompletionEngine = TRANSFORMER_ENGINE
    DEFAULT_CONSUMER: ChatCompletionConsumerType = GEMMA_CONSUMER

    if len(parts) == 1:
        engine_type, consumer_type, model_path = DEFAULT_ENGINE, DEFAULT_CONSUMER, parts[0]
    elif len(parts) == 2:
        engine_type, consumer_type, model_path = DEFAULT_ENGINE, parts[0], parts[1]
    else:
        engine_type, consumer_type, model_path = parts[0], parts[1], parts[2]
    return engine_type, consumer_type, model_path


# response


class ChatCompletionDelta(BaseModel):
    content: str = ""
    reasoning_content: str = ""

    def is_empty(self) -> bool:
        return len(self.content) == 0 and len(self.reasoning_content) == 0


class ChatCompletionChoice(BaseModel):
    delta: ChatCompletionDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    choices: list[ChatCompletionChoice]
