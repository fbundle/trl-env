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



tokenizer.pad_token_id = tokenizer.eos_token_id


def apply_chat_template(text: str) -> str:
    return tokenizer.apply_chat_template( # type: ignore
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


input_text_list = [
    "hello, this is an example",
    "water is blue"
]

input_text_list: list[str] = [apply_chat_template(text) for text in input_text_list]

print(input_text_list)

input_ids_list = [tokenizer(text).input_ids for text in input_text_list]
e: BatchEncoding = tokenizer.pad(
    {"input_ids": input_ids_list},
    padding=True,
    return_tensors="pt",
).to(device)

o: CausalLMOutputWithPast = model(
    input_ids=e.input_ids,
    attention_mask=e.attention_mask,
    temperature=0.6,
    max_new_tokens=64,
    output_logits=True,
    return_dict_in_generate=True,
    use_cache=True,
    past_key_values=None,
)

assert o.logits is not None
logits: Float[Tensor, "b n d"] = o.logits # type: ignore
completion: Int[Tensor, "b n"] = torch.argmax(logits, dim=-1)

completion_ids_list: list[list[int]] = completion.tolist()
completion_ids_list = [collapse_eos_token_id(completion_ids, tokenizer.eos_token_id) for completion_ids in completion_ids_list]

output_text_list: list[str] = [tokenizer.decode(completion_ids) for completion_ids in completion_ids_list]

print(output_text_list)






import pdb; pdb.set_trace()
