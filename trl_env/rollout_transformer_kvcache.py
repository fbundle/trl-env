import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from transformers.modeling_outputs import CausalLMOutputWithPast
from jaxtyping import Float, Int

def collapse_eos_token_id(completion_ids: list[int], eos_token_id: int) ->  list[int]:
    try:
        n = completion_ids.index(eos_token_id)
    except ValueError:
        n = len(completion_ids) - 1
    
    # eos_token found
    # [tok, tok, eos, eos, eos] -> [tok, tok, eos]
    return completion_ids[:n+1]

device = torch.device("mps")

model_id = "Qwen/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval() # type: ignore

tokenizer.padding_side = 'left'
tokenizer.pad_token_id = tokenizer.eos_token_id


def apply_chat_template(text: str) -> str:
    return tokenizer.apply_chat_template( # type: ignore
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def get_last_tokens(completion_ids_list: list[list[int]]) -> list[int | None]:
    last_tokens = []
    for completion_ids in completion_ids_list:
        if len(completion_ids) >= 1:
            last_tokens.append(completion_ids[-1])
        else:
            last_tokens.append(None)
    return last_tokens

def early_stop(last_tokens: list[int | None]) -> bool:
    for token in last_tokens:
        if token != tokenizer.eos_token_id:
            return False
    return True


input_text_list = [
    "hello, this is an example",
    "water is blue"
]

max_completion_length = 256
batch_size = len(input_text_list)

input_text_list: list[str] = [apply_chat_template(text) for text in input_text_list]
print(input_text_list)
input_ids_list = [tokenizer(text).input_ids for text in input_text_list]
e: BatchEncoding = tokenizer.pad(
    {"input_ids": input_ids_list},
    padding=True,
    return_tensors="pt",
).to(device)
completion_ids_list: list[list[int]] = [[] for _ in range(batch_size)]

i = {
    "input_ids": e.input_ids,
    "attention_mask": e.attention_mask,
}

with torch.no_grad():
    past_key_values = None
    for _ in range(max_completion_length):
        last_tokens = get_last_tokens(completion_ids_list)
        
        if early_stop(last_tokens):
            break

        o: CausalLMOutputWithPast = model(
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=True,
            past_key_values=past_key_values,
            **i,
        )

        past_key_values = o.past_key_values
        
        assert o.logits is not None
        logits: Float[Tensor, "b d"] = o.logits[:, -1, :]

        probs = torch.softmax(logits / 0.6, dim=-1)
        completion = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # completion: Int[Tensor, "b"] = torch.argmax(logits, dim=-1)

        new_i = {
            "input_ids": [[] for _ in range(batch_size)],
            "attention_mask": [[] for _ in range(batch_size)],
        }

        completion_list: list[int] = completion.tolist()
        for n in range(batch_size):
            if last_tokens[n] != tokenizer.eos_token_id:
                completion_ids_list[n].append(completion_list[n])
                new_i["input_ids"][n].append(completion_list[n])
                new_i["attention_mask"][n].append(1)
            else:
                new_i["input_ids"][n].append(tokenizer.eos_token_id)
                new_i["attention_mask"][n].append(0)
        
        i = {k: torch.tensor(v).to(device) for k, v in new_i.items()}

completion_ids_list = [collapse_eos_token_id(completion_ids, tokenizer.eos_token_id) for completion_ids in completion_ids_list]

completion_text = [tokenizer.decode(completion_ids) for completion_ids in completion_ids_list]


print(completion_text)

import pdb; pdb.set_trace()