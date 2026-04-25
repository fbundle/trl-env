import itertools
from typing import Any, Callable, Generator

import torch
from torch import Tensor
from jaxtyping import Float, Int

from transformers import Cache, GenerationMixin, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from trl_env.engine import Engine
from trl_env.environment import Env

from trl.trainer.grpo_trainer import RolloutFunc, GRPOTrainer, RewardFunc

from trl_env.processor import Processor
from trl_env.rollout import batch_rollout

class _BaseModelWithGenerate(PreTrainedModel, GenerationMixin):
    pass

type Tokens = Int[Tensor, "n"]
type Logits = Float[Tensor, "d"]
type Generate[T] = Callable[[T, Tokens], tuple[T, Logits]]

def make_model_generate(model: _BaseModelWithGenerate) -> Generate[Cache | None]:
    def model_generate(prev_cache: Cache | None, input_tokens: Tokens) -> tuple[Cache | None, Logits]:
        with torch.no_grad():
            # batch_size b = 1
            batch_input_ids: Int[Tensor, "b n"] = input_tokens.unsqueeze(dim=0).to(model.device)
            o: CausalLMOutputWithPast = model.forward(
                input_ids=batch_input_ids,
                output_logits=True,
                return_dict_in_generate=True,
                use_cache=True,
                past_key_values=prev_cache,
            )
            # get logits - o.logits: Float[Tensor, "b n d"]
            assert o.logits is not None
            batch_logits: Float[Tensor, "b n d"] = o.logits
            logits: Float[Tensor, "d"] = batch_logits[0, -1, :]
            cache: Cache | None = o.past_key_values
            return cache, logits
        
    return model_generate

type Sampler = Callable[[Logits], int]
def make_sampler(temperature: float = 0.0) -> Sampler:
    def sampler(logits: Logits) -> int:
        if temperature > 0:
            dist: Float[Tensor, "d"] = torch.softmax(logits / temperature, dim=-1)
            token: int = int(torch.multinomial(dist, num_samples=1).squeeze(dim=-1).item())
        else:
            token: int = int(torch.argmax(logits, dim=-1).item())
        return token
    
    return sampler

type StopCond = Callable[[int, int], bool]
def make_stop_cond(eos_token_set: set[int], max_completion_length: int) -> StopCond:
    def stop_cond(turn: int, token: int) -> bool:
        return (token in eos_token_set) or (turn >= max_completion_length)

    return stop_cond


class State[T]:
    def __init__(self,
        generate: Generate[T],
        sampler: Sampler,
        stop_cond: StopCond,
        cache: T = None
    ) -> None:
        self._generate = generate
        self._sampler = sampler
        self._stop_cond = stop_cond
        self._cache = cache
    
    def generate(self, input_tokens: Int[Tensor, "n"]) -> Generator[tuple[int, Logits], None, None]:
        token = None
        try:
            for turn in itertools.count():
                # generate
                self._cache, logits = self._generate(self._cache, input_tokens)
                # sample
                token = self._sampler(logits)
                yield token, logits
                if self._stop_cond(turn, token):
                    break
                # prepare next generate
                input_tokens = torch.tensor([token])
        finally:
            # finally, put the last token into the cache
            if token is not None:
                self._cache, _ = self._generate(self._cache, torch.tensor([token]))

class TransformerEngine(Engine):
    def __init__(self,
        tokenizer: PreTrainedTokenizerBase,
        model: _BaseModelWithGenerate,
        temperature: float = 0.6,
        max_completion_length: int = 512,
        extra_eos_tokens: list[int] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.temperature = temperature
        self.max_completion_length = max_completion_length
        self.eos_tokens = {self.tokenizer.eos_token_id} # type: ignore
        if extra_eos_tokens is not None:
            self.eos_tokens.update(extra_eos_tokens)

        self.states: list[State[Cache | None]] = []
        
    
    def tokenizer_encode(self, input_text: str) -> list[int]:
        return self.tokenizer(input_text).input_ids

    def tokenizer_decode(self, completion_ids: list[int]) -> str:
        output_text = self.tokenizer.decode(completion_ids)
        assert isinstance(output_text, str)
        return output_text
    
    def reset_states(self, n: int):
        self.states = [State(
            generate=make_model_generate(self.model),
            sampler=make_sampler(self.temperature),
            stop_cond=make_stop_cond(
                eos_token_set=self.eos_tokens,    # type: ignore
                max_completion_length=self.max_completion_length,
            )
        ) for _ in range(n)]

    def model_batch_generate(self, input_ids_list: list[list[int]]) -> tuple[list[list[int]], list[list[float]]]:
        completion_ids_list = []
        logprobs_list = []
        for i, input_tokens in enumerate(input_ids_list):
            completion_ids = []
            logprobs = []
            assert len(self.states) == len(input_ids_list)
            for token, logits in self.states[i].generate(torch.tensor(input_tokens)):
                completion_ids.append(token)
                dist: Float[Tensor, "d"] = torch.log_softmax(logits, dim=-1)
                logprob: float = float(dist[token].item())
                logprobs.append(logprob)
            
            completion_ids_list.append(completion_ids)
            logprobs_list.append(logprobs)
        
        return completion_ids_list, logprobs_list


def make_rollout_func(
    engine: TransformerEngine, processor: Processor, env_factory: Callable[[], Env],
    system_prompt: str, max_conversation_length: int,
) -> RolloutFunc:
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, Any]:
        try:
            # NOTE - only rollout on eval mode
            engine.model.eval()
            engine.reset_states(len(prompts))
            with torch.no_grad():
                state_list = batch_rollout(
                    engine=engine, processor=processor, env_factory=env_factory,
                    system_prompt=system_prompt, max_conversation_length=max_conversation_length,
                    seed_list=prompts,
                )
        finally:
            engine.model.train()
        
        return {
            "prompt_ids": [state.conversation[:state.initial_length] for state in state_list],
            "completion_ids": [state.conversation[state.initial_length:] for state in state_list],
            "env_mask": [state.env_mask for state in state_list],
            "logprobs": [state.logprobs for state in state_list],
            "reward": [state.reward for state in state_list],
        }

    return rollout_func    

def make_reward_func() -> RewardFunc:
    def reward_func(prompts: list[str], completions: list[str], reward: list[float], **kwargs) -> list[float]:
            return reward
    return reward_func # type: ignore



if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl_env.processor import *
    device = "mps"
    model_path = "Qwen/Qwen3.5-0.8B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model: _BaseModelWithGenerate = AutoModelForCausalLM.from_pretrained(model_path).to(device) #type: ignore

    state = State(
        generate=make_model_generate(model),
        sampler=make_sampler(
            temperature=0.6,
        ),
        stop_cond=make_stop_cond(
            eos_token_set={tokenizer.eos_token_id},
            max_completion_length=512,
        ),
    )

    text1 = qwen3_instruct_processor.append_user_input("the cat is lying on a table")
    input_token_list1: list[int] = tokenizer(text1).input_ids

    completion_list1 = list(list(zip(*state.generate(torch.tensor(input_token_list1))))[0])

    text2 = qwen3_instruct_processor.append_user_input("what is the cat lying on?")
    input_token_list2: list[int] = tokenizer(text2).input_ids
    
    completion_list2 = list(list(zip(*state.generate(torch.tensor(input_token_list2))))[0])

    token_list = input_token_list1 + completion_list1 + input_token_list2 + completion_list2

    text = tokenizer.decode(token_list)
    print(text)

