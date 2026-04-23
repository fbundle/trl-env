
import random
import sys

from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState


from trl_env.dataset import LazyDataset
from trl_env.environment import Action, Delta, Env, Seed
from trl_env.model import TransformerModel
from trl_env.trainer import train
from trl_env.trainer_config import Mode, TrainConfig
from trl_env.processor import qwen3_instruct_processor, qwen3_processor


def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer
class GuessEnv(Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def reset(self, seed: Seed) -> Delta:
        self.target = int(seed)
        self.turn = 0
        self.best_points = 0
        self.reward = 0
        self.alive = True
        return """
I have an integer between 0 and 50 in mind
every turn, you have to take a guess, output
GUESS <number>
I will say if your guess is higher or lower than my number
"""
    
    def step(self, action: Action) -> Delta:
        def helper(action: str) -> tuple[float, float, bool, str]:
            words = action.split()
            if "GUESS" not in words:
                return 0, 0, False, f"can't find your guess"
            
            guess_str = words[words.index("GUESS") + 1]

            try:
                guess = int(guess_str)
            except ValueError:
                guess = None

            if guess is None:
                return 0.5, 0, False, f"can't find the number in your guess"

            f = lambda x: 1 / (1 + x) # map [0, inf) -> [1, 0)
            number_points = f(abs(self.target - guess))
            alive = True
            if guess < self.target:
                state_delta = f"{guess} is too low"
            elif guess > self.target:
                state_delta = f"{guess} is too high"
            else:
                state_delta = f"{guess} is correct"
                alive = False
                  
            return 1.0, number_points, alive, state_delta

        

        format_points, number_points, alive, state_delta = helper(action)
        points = format_points + number_points
        
        self.turn += 1
        self.alive = alive
        self.best_points = max(self.best_points, points)
        self.reward = self.best_points * (0.99)**(self.turn)

        return state_delta

def main(train_mode: Mode, uuid: str, debug: bool):
    num_processes = PartialState().num_processes

    # model updates every effective_batch_size
    effective_batch_size = 32

    max_turn_length = 256
    # per device memory ~ batch_size x num_generations x max_conversation_length^\alpha
    # alpha = 2 for usual transformer
    # alpha = 1 for flash attention
    per_device_batch_size = 4
    num_generations = 8
    max_conversation_length = 4096

    if debug:
        effective_batch_size = 4
        per_device_batch_size = 1
        gradient_accumulation_steps = 2
        num_generations = 2

        max_turn_length = 64
        max_conversation_length = 512

        push_to_hub = False

    gradient_accumulation_steps = effective_batch_size // (per_device_batch_size * num_processes)

    assert effective_batch_size == per_device_batch_size * gradient_accumulation_steps * num_processes

    # train 1000 batches
    train_size = 100 * effective_batch_size

    # train data generation
    # total_num_steps = train_size x num_generations / effective_batch_size
    #       = 8000
    # no_points_per_step = effective_batch_size / num_generations
    
    def f(i: int) -> str:
        x = random.randint(0, 10000)
        return str(x)
    
    data = LazyDataset[str](n=train_size, f=f)

    model_path = "Qwen/Qwen3.5-4B"
    debug_model_path = "Qwen/Qwen3.5-0.8B"
    output_dir = f"mnt/output/qwen3.5-4b-tl{max_turn_length}-cl{max_conversation_length}-b{effective_batch_size}-{uuid}-lora-guess"
    deepspeed = "conf/ds_zero2.json"

    if debug:
        model_path = debug_model_path
        deepspeed = None


    rule =f"""
every turn, you can output a maximum number of {max_turn_length} tokens
the whole conversation should not last longer than {max_conversation_length} tokens
"""


    model, tokenizer = load_model_and_tokenizer(model_path)

    config = TrainConfig(
        mode=train_mode,
        deepspeed=deepspeed,
        output_dir=output_dir,
        processor=qwen3_processor,
        system_prompt=rule,
        model=TransformerModel(
            tokenizer=tokenizer,
            model=model, # type: ignore
            generation_kwargs=dict(
                max_new_tokens=max_turn_length,
                temperature=1.0,
            ),
        ),
        data=data,
        env_factory=GuessEnv,
        max_turn_length=max_turn_length,

        per_device_batch_size=per_device_batch_size,
        
        num_generations=num_generations,
        max_conversation_length=max_conversation_length,
        gradient_accumulation_steps=gradient_accumulation_steps,

        generation_kwargs=dict(),
        train_config_kwargs=dict(
            learning_rate = 1e-6,
            weight_decay = 0.001,
        ),

        save_every_seconds=3600,
        log_every_seconds=0,

        push_to_hub=push_to_hub,
    )

    train(config)

if __name__ == "__main__":
    MODE = sys.argv[1]
    UUID = "test"
    if len(sys.argv) >= 3:
        UUID = sys.argv[2]
    if MODE not in ["train", "prepare", "debug"]:
        raise RuntimeError("mode")

    if MODE == "debug":
        main("train", UUID, debug=True)
    else:
        main(MODE, UUID, debug=False) # type: ignore
