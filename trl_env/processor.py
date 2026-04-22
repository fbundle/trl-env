from typing import Protocol

from pydantic import BaseModel

# THIS SHOULD BE BUILTIN IN TOKENIZER BUT NOONE BOTHER TO DO IT
# check tokenizer.apply_chat_template and tokenizer.parse_response


Language = str

class Processor(Protocol):
    def init_system_input(self, prompt: Language) -> str: ...
    def append_user_input(self, prompt: Language) -> str: ...
    def parse_agent_output(self, completion: Language) -> tuple[str, str]: ...

class Type1ProcessorConfig(BaseModel):
    prefix_system: str
    suffix_system: str
    prefix_user: str
    suffix_user: str
    begin_answer: str
    end_answer: str

class Type1Processor(Processor):
    def __init__(self, config: Type1ProcessorConfig) -> None:
        super().__init__()
        self.config = config
    
    def init_system_input(self, prompt: Language) -> str:
        return self.config.prefix_system + prompt + self.config.suffix_system

    def append_user_input(self, prompt: str) -> Language:
        return self.config.prefix_user + prompt + self.config.suffix_user

    def parse_agent_output(self, completion: Language) -> tuple[str, str]:
        # format
        # reasoning <begin_answer> answer <end_answer> rubbish
        completion = completion.split(self.config.end_answer)[0]
        chunks = completion.split(self.config.begin_answer)
        reason = self.config.begin_answer.join(chunks[:-1])
        answer = chunks[-1]
        return reason, answer
    

qwen3_instruct_processor = Type1Processor(Type1ProcessorConfig(
    prefix_system="<|im_start|>system\n",
    suffix_system="<|im_end|>\n",
    prefix_user="<|im_start|>user\n",
    suffix_user="<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    begin_answer="</think>",
    end_answer="<|im_end|>",
))

qwen3_processor = Type1Processor(Type1ProcessorConfig(
    prefix_system="<|im_start|>system\n",
    suffix_system="<|im_end|>\n",
    prefix_user="<|im_start|>user\n",
    suffix_user="<|im_end|>\n<|im_start|>assistant\n<think>\n",
    begin_answer="</think>",
    end_answer="<|im_end|>",
))

gemma4_instruct_processor = Type1Processor(Type1ProcessorConfig(
    prefix_system="<bos><|turn>system\n",
    suffix_system="<turn|>\n",
    prefix_user="<|turn>user\n",
    suffix_user="<turn|>\n<|turn>model\n",
    begin_answer="<channel|>",
    end_answer="<turn|>",
))

gemma4_processor = Type1Processor(Type1ProcessorConfig(
    prefix_system="<bos><|turn>system\n<|think|>\n",
    suffix_system="<turn|>\n",
    prefix_user="<|turn>user\n",
    suffix_user="<turn|>\n<|turn>model\n",
    begin_answer="<channel|>",
    end_answer="<turn|>",
))

deepseekr1_processor = Type1Processor(Type1ProcessorConfig(
    prefix_system="<｜begin▁of▁sentence｜>",
    suffix_system="",
    prefix_user="<｜User｜>",
    suffix_user="<｜Assistant｜><think>\n",
    begin_answer="</think>",
    end_answer="<｜end▁of▁sentence｜>",
))