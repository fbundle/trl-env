import os
import torch
from threading import Thread
from typing import Iterator, Protocol

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import TextIteratorStreamer

from .api import Message, ChatCompletionGenerateConfig


def apply_chat_template_with_thinking(tokenizer, messages: list[Message]) -> str:
    input_text = tokenizer.apply_chat_template(
        conversation=[message.model_dump() for message in messages],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return input_text


class Engine(Protocol):
    def chat(
            self,
            messages: list[Message] | str,
            config: ChatCompletionGenerateConfig,
    ) -> Iterator[str]: ...


class TransformerEngine:
    def __init__(self, model_path: str, tokenizer_path: str | None = None):
        super().__init__()

        if tokenizer_path is None:
            tokenizer_path = model_path

        print(f"loading transformer {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            import json
            from peft import PeftModel # type: ignore
            
            with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                adapter_config = json.load(f)
            
            base_model_path = adapter_config.get("base_model_name_or_path")
            if base_model_path is None:
                raise RuntimeError("base_model_name_or_path not found in adapter_config.json")
            
            print(f"loading base model {base_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            print(f"loading adapter {model_path}")
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)

    def generate(self, input_text: str, text_streamer: TextIteratorStreamer,
                 generation_config: GenerationConfig):
        # TODO - implement caching for tokenizer
        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        self.model.generate( # type: ignore
            **input_ids,
            streamer=text_streamer,
            generation_config=generation_config,
        ) 

    def chat(
            self,
            messages: list[Message] | str,
            config: ChatCompletionGenerateConfig,
    ) -> Iterator[str]:
        text_streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # skip the prompt, stream the output only
            skip_special_tokens=False,  # pass into tokenizer.decode, skip EOS, for example
        )

        generation_config = GenerationConfig(
            max_new_tokens=config.max_completion_tokens,

            temperature=config.temperature,
            top_p=config.top_p,
            min_p=config.min_p,
            top_k=config.top_k,

            repetition_penalty=config.repetition_penalty,
        )
        if isinstance(messages, str):
            input_text = messages
        else:
            input_text = apply_chat_template_with_thinking(self.tokenizer, messages)

        thread = Thread(
            target=TransformerEngine.generate,
            args=(self, input_text, text_streamer, generation_config),
        )
        thread.start()

        def streamer() -> Iterator[str]:
            yield from text_streamer
            thread.join()

        return streamer()


class MlxEngine(Engine):
    def __init__(self, model_path: str):
        super().__init__()
        import mlx_lm

        print(f"loading mlx {model_path}")
        model, tokenizer, config = mlx_lm.load(  # type: ignore
            path_or_hf_repo=model_path,
            return_config=True,
        )
        self.model = model
        self.tokenizer = tokenizer

    def chat(
            self,
            messages: list[Message] | str,
            config: ChatCompletionGenerateConfig,
    ) -> Iterator[str]:
        if isinstance(messages, str):
            input_text = messages
        else:
            input_text = apply_chat_template_with_thinking(self.tokenizer, messages)

        import mlx_lm.sample_utils

        response_generator = mlx_lm.stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=input_text,
            max_tokens=config.max_completion_tokens,
            sampler=mlx_lm.sample_utils.make_sampler(
                temp=config.temperature,
                top_p=config.top_p,
                min_p=config.min_p,
                top_k=config.top_k,
            ),
            logits_processors=mlx_lm.sample_utils.make_logits_processors(
                presence_penalty=config.presence_penalty,
                frequency_penalty=config.frequency_penalty,
                repetition_penalty=config.repetition_penalty,
            ),
        )

        def streamer() -> Iterator[str]:
            for response in response_generator:
                yield response.text

        return streamer()


class GgufEngine(Engine):
    def __init__(self, model_path: str):
        super().__init__()
        from llama_cpp import Llama

        print(f"loading gguf {model_path}")
        self.llm = Llama(model_path=model_path)

    def chat(
            self,
            messages: list[Message] | str,
            config: ChatCompletionGenerateConfig,
    ) -> Iterator[str]:
        if isinstance(messages, str):
            raise RuntimeError("does not support message type str")

        chunk_iter = self.llm.create_chat_completion(
            messages=[{"role": m.role, "content": m.content} for m in messages], # type: ignore

            stream=True,

            max_tokens=config.max_completion_tokens,

            temperature=config.temperature,
            top_p=config.top_p,
            min_p=config.min_p,
            top_k=config.top_k,

            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            repeat_penalty=config.repetition_penalty,
        )

        for chunk in chunk_iter:
            try:
                yield chunk["choices"][0]["delta"]["content"] # type: ignore
            except:
                pass




