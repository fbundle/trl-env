from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

from .environment import Env, Seed
from .decoder import RolloutDecoder, RolloutDecoderFactory
from .processor import Processor
from .tokenizer import Tokenizer

from trl.trainer.grpo_trainer import RolloutFunc, GRPOTrainer, RewardFunc



@dataclass
class RolloutState:
    initial_length: int
    conversation: list[int]
    env_mask: list[int]
    logprobs: list[float]
    reward: float | None

    def append_completion(
        self,
        completion_ids: list[int],
        logprobs: list[float] | None,
    ) -> RolloutState:
        if logprobs is None:
            env_mask = [0] * len(completion_ids)
            logprobs = [0.0] * len(completion_ids)
        else:
            env_mask = [1] * len(completion_ids)

        self.conversation.extend(completion_ids)
        self.env_mask.extend(env_mask)
        self.logprobs.extend(logprobs)
        return self

def init_rollout_state(initial_prompt_ids: list[int]) -> RolloutState:
    return RolloutState(
        initial_length=len(initial_prompt_ids),
        conversation=initial_prompt_ids,
        env_mask=[],
        logprobs=[],
        reward=None,
    )

def rollout(
    processor: Processor, tokenizer: Tokenizer,
    decoder: RolloutDecoder, env: Env,
    system_prompt: str, max_conversation_length: int,
    seed: Seed, 
    conversation_logger: Callable[[str, str], None] | None = None,
) -> RolloutState:
    def LOG(role: str, content: str):
        if conversation_logger is not None:
            conversation_logger(role, content)

    env, initial_delta = env.reset(seed)

    LOG("system", system_prompt)
    LOG("user", initial_delta)
    # assuming tokenizer is additive
    # tok(a ++ b) = tok(a) ++ tok(b)
    system_prompt_ids = tokenizer.encode(processor.init_system_input(system_prompt))
    initial_prompt_ids = system_prompt_ids + tokenizer.encode(processor.append_user_input(initial_delta))

    state = init_rollout_state(initial_prompt_ids=initial_prompt_ids)

    while True:
        # precheck env.alive
        if not env.alive:
            break
        # model generate
        completion_ids, logprobs = decoder.generate(state.conversation)
        # append agent completion
        state = state.append_completion(
            completion_ids=completion_ids,
            logprobs=logprobs,
        )
        # parse (reason, action)
        completion_text = tokenizer.decode(completion_ids)
        LOG("assistant", completion_text)
        reason, action = processor.parse_agent_output(completion_text)
        # interact with environment
        env, delta = env.step(action)
        LOG("user", delta)
        # save reward
        state.reward = env.reward
        # postcheck env.alive
        if not env.alive:
            LOG("log", "env terminated")
            break
        # append environment completion
        # assuming tokenizer is additive
        # tok(a ++ b) = tok(a) ++ tok(b)
        delta_ids = tokenizer.encode(processor.append_user_input(delta))
        state = state.append_completion(
            completion_ids=delta_ids,
            logprobs=None,
        )
        # terminate env if conversation is long
        if len(state.conversation) >= max_conversation_length:
            env.alive = False
            LOG("log", "env terminated due to long conversation")
            break
    return state

from tqdm import tqdm

def make_rollout_func(
    processor: Processor, tokenizer: Tokenizer,
    decoder_factory: RolloutDecoderFactory,
    env_factory: Callable[[], Env],    
    system_prompt: str, max_conversation_length: int,
) -> RolloutFunc:
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, Any]:
        decoder = decoder_factory.make_decoder(trainer)
        env = env_factory()

        state_list = []
        for prompt in tqdm(prompts, desc="rolling_out ..."):
            # TODO batch this - need to make decoder in batch as well
            state = rollout(
                processor=processor, tokenizer=tokenizer,
                decoder=decoder, env=env,
                system_prompt=system_prompt, max_conversation_length=max_conversation_length,
                seed=prompt,
            )
            state_list.append(state)

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



import multiprocessing as mp

class ChildDecoder(RolloutDecoder):
    def __init__(self, index: int, qi: mp.Queue, qo: mp.Queue) -> None:
        self.index = index
        self.qi = qi
        self.qo = qo
    
    def close(self):
        self.qi.put({
            "index": self.index,
            "input_ids": None,
        })

    def generate(self, input_ids: list[int]) -> tuple[list[int], list[float]]:
        self.qi.put({
            "index": self.index,
            "input_ids": input_ids,
        })
        return self.qo.get()

def split_decoder(ctx, n: int) -> tuple[mp.Queue, list[ChildDecoder]]:
    qi = ctx.Queue(maxsize=1024)
    child_decoder_list = []
    for index in range(n):
        child_decoder = ChildDecoder(
            index=index,
            qi=qi,
            qo=ctx.Queue(maxsize=1),
        )
        child_decoder_list.append(child_decoder)
    return qi, child_decoder_list

def rollout_then_close_decoder(args):
    qs, processor, tokenizer, decoder, env_factory, system_prompt, max_conversation_length, seed = args
    env = env_factory()
    state, error = None, None
    try:
        state = rollout(
            processor=processor, tokenizer=tokenizer,
            decoder=decoder, env=env,
            system_prompt=system_prompt, max_conversation_length=max_conversation_length,
            seed=seed,
        )
        error = None
    except Exception as e:
        state = None
        error = e
    finally:
        decoder.close()
        qs.put({
            "index": decoder.index,
            "state": state,
            "error": error,
        })

def make_rollout_func_mp(
    processor: Processor, tokenizer: Tokenizer,
    decoder_factory: RolloutDecoderFactory,
    env_factory: Callable[[], Env],    
    system_prompt: str, max_conversation_length: int,
) -> RolloutFunc:
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, Any]:
        decoder = decoder_factory.make_decoder(trainer)
        ctx = mp.get_context('spawn')

        qi, child_decoder_list = split_decoder(ctx, len(prompts))
        qs = ctx.Queue(len(prompts))

        child_list = []
        for index, seed in enumerate(prompts):
            p = ctx.Process(target=rollout_then_close_decoder, args=[(
                qs,
                processor, tokenizer,
                child_decoder_list[index], env_factory,
                system_prompt, max_conversation_length,
                seed,
            )])
            p.start()
            child_list.append(p)
        
        # process messages
        with tqdm(total=len(prompts), desc="rolling_out ...") as pbar:
            finish_count = 0
            while finish_count < len(prompts):
                req = qi.get()
                if req["input_ids"] is None:
                    finish_count += 1
                    pbar.update(1)
                    continue

                res = decoder.generate(req["input_ids"])
                index = req["index"]
                child_decoder_list[index].qo.put(res)

        for child in child_list:
            child.join()

        indexed_state_list = []
        for _ in prompts:
            indexed_state = qs.get()
            error = indexed_state["error"]
            if error is not None:
                raise error
            indexed_state_list.append(indexed_state)
        
        indexed_state_list.sort(key=lambda indexed_state: indexed_state["index"])

        state_list = [indexed_state["state"] for indexed_state in indexed_state_list]

        return {
            "prompt_ids": [state.conversation[:state.initial_length] for state in state_list],
            "completion_ids": [state.conversation[state.initial_length:] for state in state_list],
            "env_mask": [state.env_mask for state in state_list],
            "logprobs": [state.logprobs for state in state_list],
            "reward": [state.reward for state in state_list],
        }

    return rollout_func
