
import os
import random
import sys

from peft import LoraConfig, get_peft_model
import sympy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState


from experiment.examples.discrete_logarithm.discrete_logarithm_env import DiscreteLogarithmEnv, DiscreteLogarithmSeed, SYSTEM_PROMPT
from experiment.examples.trl_trainer.dataset import LazyDataset
from trl_env.model import TransformerModel
from experiment.examples.trl_trainer.trainer import train
from experiment.examples.trl_trainer.trainer_config import Mode, TrainConfig
from trl_env.processor import deepseekr1_processor, qwen3_processor


def load_model_and_tokenizer(model_path: str):
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_implementation,
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


def main(train_mode: Mode, uuid: str, debug: bool):
    num_processes = PartialState().num_processes

    # model updates every effective_batch_size
    effective_batch_size = 32

    max_turn_length = 512
    # per device memory ~ batch_size x num_generations x max_conversation_length^\alpha
    # alpha = 2 for usual transformer
    # alpha = 1 for flash attention
    per_device_batch_size = 4
    num_generations = 8
    max_conversation_length = 4096
    push_to_hub = True

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
    train_size = 1000 * effective_batch_size


    # train data generation
    # total_num_steps = train_size x num_generations / effective_batch_size
    #       = 8000
    # no_points_per_step = effective_batch_size / num_generations

    def generate_seed(bit_size: int = 10) -> str:
        # find a prime p
        while True:
            p = random.randint(2**(bit_size-1), 2**bit_size)
            if sympy.isprime(p):
                break
        # use g=2 as a simple generator (works for most primes)
        g = 2
        # sample x and compute h
        x = random.randint(1, p - 2)
        h = pow(g, x, p)
        return DiscreteLogarithmSeed(g=g, h=h, p=p).model_dump_json()

    def f(i: int) -> str:
        # make problem progressively harder
        # bit_size 4 -> 16
        proportion: float = i / train_size
        bit_size = int(4 + (16 - 4) * proportion)
        return generate_seed(bit_size)
    
    data = LazyDataset[str](n=train_size, f=f)

    env_factory = lambda: DiscreteLogarithmEnv()

    processor = qwen3_processor
    model_path = "Qwen/Qwen3.5-4B"
    debug_model_path = "Qwen/Qwen3.5-0.8B"


    output_dir = f"mnt/output/discrete-logarithm-{os.path.basename(model_path)}-tl{max_turn_length}-cl{max_conversation_length}-b{effective_batch_size}-{uuid}"
    deepspeed = "conf/ds_zero2.json"
    deepspeed = None # TODO - change to deepspeed for multi GPUs
    
    if debug:
        model_path = debug_model_path
        deepspeed = None

    rule = SYSTEM_PROMPT.format(max_turn_length=max_turn_length, max_conversation_length=max_conversation_length)


    model, tokenizer = load_model_and_tokenizer(model_path)

    config = TrainConfig(
        mode=train_mode,
        deepspeed=deepspeed,
        output_dir=output_dir,
        processor=processor,
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
        env_factory=env_factory,
        max_turn_length=max_turn_length,

        per_device_batch_size=per_device_batch_size,
        
        num_generations=num_generations,
        max_conversation_length=max_conversation_length,
        gradient_accumulation_steps=gradient_accumulation_steps,

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
    from dotenv import load_dotenv
    load_dotenv()

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
