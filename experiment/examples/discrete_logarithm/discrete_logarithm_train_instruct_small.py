
import os
import random
import sys
from typing import Literal

from peft import LoraConfig, get_peft_model
import sympy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState
from transformers.trainer_utils import get_last_checkpoint


from experiment.examples.discrete_logarithm.discrete_logarithm_env import DiscreteLogarithmEnv, DiscreteLogarithmSeed, SYSTEM_PROMPT

from experiment.examples.trl_trainer_util.dataset import LazyDataset
from experiment.examples.trl_trainer_util.trainer_callback import TimeBasedLogSaveCallback

from trl_env.processor import qwen3_instruct_processor


from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig

from trl_env.rollout_transformer import TransformerEngine, make_reward_func, make_rollout_func

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

type Mode = Literal["prepare", "train", "debug"]
ModePrepare: Mode = "prepare"
ModeTrain: Mode = "train"
ModeDebug: Mode = "debug"
all_modes = [ModeTrain, ModePrepare, ModeDebug]

def load_batch_information(mode: Mode):
    num_processes = PartialState().num_processes

    # model updates every effective_batch_size
    # per device memory ~ batch_size x num_generations x max_conversation_length^\alpha
    # alpha = 2 for usual transformer
    # alpha = 1 for flash attention
    effective_batch_size = 16
    per_device_batch_size = 4
    num_generations = 16
    max_conversation_length = 4096
    max_turn_length = 1024

    if mode == ModeDebug:
        effective_batch_size = 4
        per_device_batch_size = 1
        num_generations = 2
        max_conversation_length = 512
        max_turn_length = 64

    gradient_accumulation_steps = effective_batch_size // (per_device_batch_size * num_processes)

    assert effective_batch_size == per_device_batch_size * gradient_accumulation_steps * num_processes
    
    return (
        effective_batch_size,
        per_device_batch_size,
        num_generations,
        max_conversation_length,
        max_turn_length,
        gradient_accumulation_steps,
    )

def load_env_and_data(effective_batch_size: int):
    # train 1000 batches
    train_size = 1000 * effective_batch_size
    # train data generation
    # total_num_steps = train_size x num_generations / effective_batch_size
    #       = 8000
    # no_points_per_step = effective_batch_size / num_generations
    def f(i: int) -> str:
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
        # make problem progressively harder
        # bit_size 4 -> 16
        proportion: float = i / train_size
        bit_size = int(4 + (16 - 4) * proportion)
        return generate_seed(bit_size)
    
    data = LazyDataset[str](n=train_size, f=f)
    env_factory = lambda: DiscreteLogarithmEnv()
    return (
        env_factory,
        data,
    )

def load_model(mode: Mode, max_turn_length: int, max_conversation_length: int):
    processor = qwen3_instruct_processor
    model_path = "Qwen/Qwen3.5-0.8B"
    debug_model_path = "Qwen/Qwen3.5-0.8B"

    deepspeed = None # TODO - change to "conf/ds_zero2.json" for multi GPUs
    
    if mode == ModeDebug:
        model_path = debug_model_path
        deepspeed = None

    model, tokenizer = load_model_and_tokenizer(model_path)
    def apply_chat_template(*args, **kwargs):
        raise RuntimeError("GRPO must not use apply_chat_template")

    # prevent TRL from using apply_chat_template
    tokenizer.apply_chat_template = apply_chat_template


    generation_kwargs=dict(
        max_new_tokens=max_turn_length,
        temperature=1.0,
    )
    # in prepare mode, always generate in full to monitor GPU memory
    if mode == ModePrepare:
        generation_kwargs["min_new_tokens"] = max_conversation_length

    engine = TransformerEngine(
        tokenizer=tokenizer,
        model=model, # type: ignore
        generation_kwargs=generation_kwargs,
    )

    return (
        model_path,
        processor,
        engine,
        deepspeed,
    )

def get_hf_info(output_dir: str) -> tuple[bool, str, str]:
    hf_user = os.environ.get("HF_USER", default=None)
    hf_token = os.environ.get("HF_TOKEN", default=None)
    if hf_user is None or hf_token is None:
        return False, "", ""
    
    hf_model = hf_user + "/" + os.path.basename(output_dir)
    return True, hf_model, hf_token


def main(mode: Mode, uuid: str):
    (
        effective_batch_size,
        per_device_batch_size,
        num_generations,
        max_conversation_length,
        max_turn_length,
        gradient_accumulation_steps,
    ) = load_batch_information(mode=mode)
    (
        env_factory,
        data,
    ) = load_env_and_data(effective_batch_size=effective_batch_size)
    (
        model_path,
        processor,
        model,
        deepspeed,
    ) = load_model(mode=mode, max_turn_length=max_turn_length, max_conversation_length=max_conversation_length)

    output_dir = f"mnt/output/discrete-logarithm-{os.path.basename(model_path)}-instruct-tl{max_turn_length}-cl{max_conversation_length}-b{effective_batch_size}-{uuid}"

    (
        push_to_hub,
        hf_model,
        hf_token,
    ) = get_hf_info(output_dir)
    push_to_hub = push_to_hub and (mode != ModeDebug)

    # TRAIN
    train_dataset = data.map(
        lambda input_text: {"prompt": input_text}
    )

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        deepspeed=deepspeed,

        per_device_train_batch_size=per_device_batch_size,
        num_generations=num_generations,
        max_completion_length=max_conversation_length,  # for padding the output of rollout_func
        gradient_accumulation_steps=gradient_accumulation_steps,

        # floating point precision
        bf16=has_cuda or has_mps,
        tf32=has_cuda,

        # no eval
        eval_strategy="no",

        # log and save - set a big number as we manually save and log
        save_strategy="epoch",
        logging_strategy="epoch",

        # hugging face
        push_to_hub=push_to_hub,
        hub_model_id=hf_model,
        hub_token=hf_token,
        hub_strategy="every_save",
        hub_always_push=True,
        report_to="tensorboard",

        # vllm
        use_vllm=False, # may change to true in the future
        vllm_mode="colocate",

        gradient_checkpointing=True,
    )

    system_prompt = SYSTEM_PROMPT.format(max_turn_length=max_turn_length, max_conversation_length=max_conversation_length)

    rollout_func = make_rollout_func(
        engine=model,
        processor=processor,
        env_factory=env_factory,
        system_prompt=system_prompt,
        max_conversation_length=max_conversation_length,
    )
    reward_func = make_reward_func()


    trainer = GRPOTrainer(
        args=training_args,
        model=model.model,
        processing_class=model.tokenizer,
        rollout_func=rollout_func,
        reward_funcs=reward_func, # type: ignore
        reward_processing_classes=model.tokenizer,
        train_dataset=train_dataset, # type: ignore
        callbacks=[TimeBasedLogSaveCallback(
            save_every_seconds=3600,
            log_every_seconds=0,
        )],
    )

    trainer.train(resume_from_checkpoint=get_last_checkpoint(output_dir))


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    MODE = sys.argv[1]
    UUID = "test"
    if len(sys.argv) >= 3:
        UUID = sys.argv[2]
    if MODE not in all_modes:
        raise RuntimeError("mode")


    main(MODE, UUID) # type: ignore
