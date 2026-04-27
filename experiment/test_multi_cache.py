import torch
from transformers import DynamicCache

from trl_env.generate_transformer import TransformerTokenizer

def split_cache(batched_cache: DynamicCache) -> list[DynamicCache]:
    """
    Splits a batched DynamicCache into a list of individual DynamicCache objects.
    
    Args:
        batched_cache: A DynamicCache object with batch_size > 1
        
    Returns:
        A list of DynamicCache objects, each with batch_size = 1
    """
    batch_size = batched_cache.key_cache[0].shape[0]
    individual_caches = []
    
    for b in range(batch_size):
        single_cache = DynamicCache()
        for layer_idx in range(len(batched_cache.key_cache)):
            # Slice the batch dimension [batch, num_heads, seq_len, head_dim]
            k = batched_cache.key_cache[layer_idx][b : b+1]
            v = batched_cache.value_cache[layer_idx][b : b+1]
            single_cache.update(k, v, layer_idx)
        individual_caches.append(single_cache)
        
    return individual_caches

def merge_caches(cache_list: list[DynamicCache]) -> DynamicCache:
    """
    Merges a list of individual DynamicCache objects into a single batched DynamicCache.
    All caches must have the same number of layers and sequence lengths must be 
    compatible for batching (usually they should be padded or you must handle 
    the attention mask separately).
    
    Args:
        cache_list: A list of DynamicCache objects (typically batch_size=1 each)
        
    Returns:
        A single batched DynamicCache object
    """
    if not cache_list:
        return DynamicCache()
        
    combined_cache = DynamicCache()
    num_layers = len(cache_list[0].key_cache)
    
    for layer_idx in range(num_layers):
        k_combined = torch.cat([c.key_cache[layer_idx] for c in cache_list], dim=0)
        v_combined = torch.cat([c.value_cache[layer_idx] for c in cache_list], dim=0)
        combined_cache.update(k_combined, v_combined, layer_idx)
        
    return combined_cache

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from trl_env.processor import qwen3_processor as processor
from transformers.modeling_outputs import CausalLMOutputWithPast
from jaxtyping import Float, Int
from torch import Tensor
from dataclasses import dataclass

@dataclass
class State:
    prev_logits: Float[Tensor, "d"] | None
    next_token_queue: list[int]
    cache: DynamicCache

@dataclass
class Output:
    token: int                          # token output
    logits: Float[Tensor, "d"] | None   # the dist token was sampled from
    

class Decoder:
    def __init__(self) -> None:
        self.pool: dict[str, State] = {}
    
    def insert(self, key: str, tokens: list[int]):
        self.pool[key] = State(
            prev_logits=None,
            next_token_queue=tokens,
            cache=DynamicCache(),
        )
    
    def generate(self, model: PreTrainedModel) -> dict[str, Output]:
        key_list = list(self.pool.keys())
        # sample if necessary
        for key in key_list:
            if len(self.pool[key].next_token_queue) == 0:
                logits = self.pool[key].prev_logits
                assert logits is not None
                temperature = 0.6
                if temperature > 0:
                    dist: Float[Tensor, "d"] = torch.softmax(logits / temperature, dim=-1)
                    token: int = int(torch.multinomial(dist, num_samples=1).squeeze(dim=-1).item())
                else:
                    token: int = int(torch.argmax(logits, dim=-1).item())
                
                self.pool[key].next_token_queue.append(token)
        # pop tokens from pool
        batch_token_list = []
        
        for key in key_list:
            token = self.pool[key].next_token_queue.pop(0)
            batch_token_list.append([token])
        
        batch_input_ids = torch.tensor(batch_token_list, device=model.device)
        
        # merge cache
        cache = merge_caches([self.pool[key].cache for key in key_list])

        # forward
        o: CausalLMOutputWithPast = model.forward(
            input_ids=batch_input_ids,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=True,
            past_key_values=cache,
        )

        # split cache
        cache_list = split_cache(o.past_key_values) # type: ignore
        for i, key in enumerate(key_list):
            self.pool[key].cache = cache_list[i]
        
        # get output
        output = {key: Output(
            token=batch_token_list[i][0],
            logits=self.pool[key].prev_logits,
        ) for i, key in enumerate(key_list)}
        
        # write logits for the next step
        assert o.logits is not None
        batch_logits: Float[Tensor, "b 1 d"] = o.logits
        for i, key in enumerate(key_list):
            logits = batch_logits[i, 0, :]
            self.pool[key].prev_logits = logits

        return output








def main():
    model_path = "Qwen/Qwen3.5-0.8B"
    t = AutoTokenizer.from_pretrained(model_path)
    eos_token: int = t.eos_token_id
    tokenizer = TransformerTokenizer(t)

    model = AutoModelForCausalLM.from_pretrained(model_path)

    
    cache_dict: dict[str, DynamicCache] = {
        "q1": DynamicCache(),
        "q2": DynamicCache(),
    }

    input_pool: dict[str, list[int]] = {
        "q1": tokenizer.encode(processor.append_user_input("the cat is lying on the rooftop")),
        "q2": tokenizer.encode(processor.append_user_input("the turtle is chasing a fire truck")),
    }

    logits_pool: dict[str, Float[Tensor, "d"] | None] = {
        "q1": None,
        "q2": None,
    }

    output_pool: dict[str, list[int]] = {
        "q1": [],
        "q2": [],
    }

    while True:
        key_list = list(input_pool.keys())
        
        # sample if empty





        batch_token_list = []
        cache_list = []
        for key in key_list:
            batch_token_list.append([input_pool[key][0]])
            input_pool[key] = input_pool[key][1:]
            cache_list.append(cache_dict[key])
        
        batch_input_ids = torch.tensor(batch_token_list, device=model.device)
        cache = merge_caches(cache_list)

        o: CausalLMOutputWithPast = model.forward(
            input_ids=batch_input_ids,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=True,
            past_key_values=cache,
        )
        
        assert o.logits is not None
        batch_logits: Float[Tensor, "b 1 d"] = o.logits


        











if __name__ == "__main__":
    main()


