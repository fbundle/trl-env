# TRL-ENV

[TRL](https://github.com/huggingface/trl) is a convenient library to train large language model using reinforcement learning (RL). However,  it is still too new, the interface is not well-developed yet. `rollout_func` is a low-level interface to write your own rollout for RL and `environment_factory` is a high-level interface to train your model with external environemnt, however, how it parse the model output for tool use is uncleared and not documented.



[TRL-ENV](https://github.com/fbundle/trl-env) addresses the middle-level with a very simple environment interface

```python
type Action = str
type Delta = str
type Seed = str

class Env(Protocol):
    last_step_reward: float
    alive: bool
    def reset(self, seed: Seed) -> Delta: ...
    def step(self, action: Action) -> Delta: ...
```

It is similar to tool call if not the same. Moreover, [transformers](https://github.com/huggingface/transformers) despite after 8 years of development (as of 2026) is still not stable. For example, not all models has `Tokenizer.parse_response` which should be a basic function that must be implemented from the beginning. [TRL-ENV](https://github.com/fbundle/trl-env) requires `Tokenizer.parse_response` to be existed by `Processor` interface

```python

Language = str

class Processor(Protocol):
    def init_system_input(self, prompt: Language) -> str: ...
    def append_user_input(self, prompt: Language) -> str: ...
    def parse_agent_output(self, completion: Language) -> tuple[str, str]: ...
```

[TRL-ENV](https://github.com/fbundle/trl-env) also provide a very simple agentic LLM interface. Training code is about 200 lines. See `examples`