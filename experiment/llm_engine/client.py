from __future__ import annotations

import os
import sys
import time
from typing import Iterator

import requests

from .api import ChatCompletionDelta, ChatCompletionRequest, ChatCompletionChunk
from .api import Message, ROLE_USER, ROLE_SYSTEM, ROLE_ASSISTANT


def chat(
        url: str,
        req: ChatCompletionRequest,
        token: str | None = None,
) -> Iterator[ChatCompletionDelta]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with requests.post(
        url=url,
        json=req.model_dump(),
        headers=headers,
        stream=True,
    ) as res:
        res.raise_for_status()
        for b in res.iter_lines():
            line = b.decode("utf-8")
            parts = line.split(":", maxsplit=1)
            if len(parts) != 2:
                continue
            key, val = parts[0].strip(), parts[1].strip()
            if key != "data":
                continue
            
            if val == "[DONE]":
                continue
            
            chunk = ChatCompletionChunk.model_validate_json(val)
            if len(chunk.choices) > 0 and not chunk.choices[0].delta.is_empty():
                yield chunk.choices[0].delta


class Conversation:
    path: str | None
    messages: list[Message]

    def __init__(self, path: str | None = None):
        self.path = path
        self.messages = []

    def load(self):
        if self.path is not None and os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    m = Message.model_validate_json(line)
                    self.append(m, write=False)

    def append(self, m: Message, write: bool = True):
        self.messages.append(m)
        if self.path is not None and write:
            with open(self.path, "a") as f:
                f.write(m.model_dump_json() + "\n")


WELCOME = "type your prompt (type '# <prompt>' to set system prompt)\n"

PROMPT_PREFIX = "> "
SYSTEM_PREFIX = "# "


def main(path: str | None, url: str, req: ChatCompletionRequest, token: str | None = None):
    c = Conversation(path)

    print(WELCOME)
    for message in c.messages:
        if message.role == ROLE_USER:
            print(f"{PROMPT_PREFIX}{message.content}")
        elif message.role == ROLE_SYSTEM:
            print(f"{SYSTEM_PREFIX}{message.content}")
        else:
            print(message.content)

    while True:
        if len(c.messages) > 0 and c.messages[-1].role == ROLE_USER:
            text_list = []
            t0 = time.perf_counter()

            # update req.messages
            req.messages = c.messages

            try:
                for delta in chat(url, req, token=token):
                    if len(delta.content) > 0:
                        text_list.append(delta.content)

                    print(delta.reasoning_content, end="", flush=True, file=sys.stderr)
                    print(delta.content, end="", flush=True)
                print()
            except Exception as e:
                print("ERROR: ", e)
            finally:
                text = "".join(text_list)
                c.append(Message(
                    role=ROLE_ASSISTANT,
                    content=text,
                ))
                t1 = time.perf_counter()
                word_per_sec = len(text.split()) / (t1 - t0)
                print(f"stats: word_per_sec {word_per_sec}", file=sys.stderr)
        else:
            input_text = input(PROMPT_PREFIX)
            if input_text.startswith(SYSTEM_PREFIX):
                system_prompt = input_text.lstrip(SYSTEM_PREFIX)
                c.append(Message(
                    role=ROLE_SYSTEM,
                    content=system_prompt,
                ))
            else:
                user_prompt = input_text
                c.append(Message(
                    role=ROLE_USER,
                    content=user_prompt,
                ))
