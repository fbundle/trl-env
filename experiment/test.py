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

def get_last_token(completion_ids: list[int]) -> int | None:
    if len(completion_ids) >= 1:
        return completion_ids[-1]
    else:
        return None


def early_stop(completion_ids_list: list[list[int]]) -> bool:
    for completion_ids in completion_ids_list:
        token = get_last_token(completion_ids)
        if token != tokenizer.eos_token_id:
            return False
    return True

temperature = 0.6

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
    # return_tensors="pt",
)

completion_ids_list: list[list[int]] = [[] for _ in range(batch_size)]

input_ids: list[list[int]] = e.input_ids
attention_mask: list[list[int]] = e.attention_mask

with torch.no_grad():
    past_key_values = None
    for _ in range(max_completion_length):

        if early_stop(completion_ids_list):
            break

        o: CausalLMOutputWithPast = model(
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=True,
            past_key_values=past_key_values,
            input_ids=torch.tensor(input_ids).to(device),
            attention_mask=torch.tensor(attention_mask).to(device),
        )

        past_key_values = o.past_key_values
        
        assert o.logits is not None
        logits: Float[Tensor, "b d"] = o.logits[:, -1, :]

        # probs = torch.softmax(logits / temperature, dim=-1)
        # completion = torch.multinomial(probs, num_samples=1).squeeze(-1)

        completion: Int[Tensor, "b"] = torch.argmax(logits, dim=-1)

        for i, token in enumerate(completion.tolist()):
            if get_last_token(completion_ids_list[i]) == tokenizer.eos_token_id:
                input_ids[i] = [tokenizer.eos_token_id]
                attention_mask[i].append(0)
            else:
                completion_ids_list[i].append(token)
                input_ids[i] = [token]
                attention_mask[i].append(1)


completion_text = [tokenizer.decode(completion_ids) for completion_ids in completion_ids_list]


print(completion_text)

import pdb; pdb.set_trace()