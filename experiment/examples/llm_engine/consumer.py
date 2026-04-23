from .api import ChatCompletionDelta

type ChatCompletionMode = str
MODE_REASON: ChatCompletionMode = "reason"
MODE_BODY: ChatCompletionMode = "body"
MODE_STOP: ChatCompletionMode = "stop"

class ChatCompletionConsumer:
    def split_tokens(self) -> list[str]:
        raise NotImplementedError
    def consume(self, chunk: str) -> tuple[ChatCompletionDelta | None, bool]:
        raise NotImplementedError

class RawChatCompletionConsumer(ChatCompletionConsumer):
    def split_tokens(self) -> list[str]:
        return []
    def consume(self, chunk: str) -> tuple[ChatCompletionDelta | None, bool]:
        return ChatCompletionDelta(content=chunk, reasoning_content=""), True

class GemmaChatCompletionConsumer(ChatCompletionConsumer):
    # chat completion format
    # <|channel>reasoning<channel|>answer<turn|>

    def __init__(self):
        super().__init__()
        self.mode = MODE_BODY

    def split_tokens(self) -> list[str]:
        return ["<|channel>", "<channel|>", "<turn|>"]

    def consume(self, chunk: str) -> tuple[ChatCompletionDelta | None, bool]:
        if len(chunk) == 0:
            return None, True

        if self.mode == MODE_STOP:
            return None, False

        if chunk == "<|channel>":
            self.mode = MODE_REASON
            return None, True
        elif chunk == "<channel|>":
            self.mode = MODE_BODY
            return None, True
        elif chunk == "<turn|>":
            self.mode = MODE_STOP
            return None, True
        else:
            if self.mode == MODE_REASON:
                return ChatCompletionDelta(content="", reasoning_content=chunk), True
            else:
                return ChatCompletionDelta(content=chunk, reasoning_content=""), True


class QwenChatCompletionConsumer(GemmaChatCompletionConsumer):
    # chat completion format
    # reasoning</think>answer
    def __init__(self):
        super().__init__()
        self.mode = MODE_REASON

    def split_tokens(self) -> list[str]:
        return ["<think>", "</think>", "<|im_end|>"]

    def consume(self, chunk: str) -> tuple[ChatCompletionDelta | None, bool]:
        if len(chunk) == 0:
            return None, True

        if self.mode == MODE_STOP:
            return None, False

        if chunk == "<think>":
            self.mode = MODE_REASON
            return None, True
        elif chunk == "</think>":
            self.mode = MODE_BODY
            return None, True
        elif chunk == "<|im_end|>":
            self.mode = MODE_STOP
            return None, True
        else:
            if self.mode == MODE_REASON:
                return ChatCompletionDelta(content="", reasoning_content=chunk), True
            else:
                return ChatCompletionDelta(content=chunk, reasoning_content=""), True

