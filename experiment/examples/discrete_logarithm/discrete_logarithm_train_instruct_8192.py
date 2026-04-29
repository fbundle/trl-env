
import os
import random
import sys
from typing import Literal

from peft import LoraConfig, get_peft_model
import sympy
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState
from transformers.trainer_utils import get_last_checkpoint

from experiment.examples.trl_trainer_util.dataset import LazyDataset
from experiment.examples.trl_trainer_util.trainer_callback import TimeBasedLogSaveCallback

from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig

from trl_env.rollout import make_reward_func, make_rollout_func
from trl_env.tokenizer import TransformerTokenizer

from transformers import BitsAndBytesConfig



def load_model_and_tokenizer(
    model_path: str,
    load_in_4bit: bool = False,
    attn_implementaion: str = "flash_attention_2",
):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            # device_map="auto",
            attn_implementation=attn_implementaion,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            # device_map="auto",
            attn_implementation=attn_implementaion,
        )

    return model, tokenizer

type Mode = Literal["prepare", "train", "debug"]
ModePrepare: Mode = "prepare"
ModeTrain: Mode = "train"
ModeDebug: Mode = "debug"
all_modes = [ModeTrain, ModePrepare, ModeDebug]

def apply_chat_template(*args, **kwargs):
    raise RuntimeError("GRPO must not use apply_chat_template")

def load_model_for_training(mode: Mode, max_turn_length: int, max_conversation_length: int):
    from trl_env.decoder_vllm import VLLMDecoderFactory
    from trl_env.processor import qwen3_instruct_processor, qwen3_processor
    from experiment.examples.discrete_logarithm.discrete_logarithm_env import EXTRA_EOS_TOKEN_LIST


    processor = qwen3_instruct_processor
    model_path = "Qwen/Qwen3-4B"
    debug_model_path = "Qwen/Qwen3-0.6B"
    deepspeed = None # "conf/ds_zero2.json"
    
    if mode == ModeDebug:
        model_path = debug_model_path
        deepspeed = None

    model, processing_class = load_model_and_tokenizer(model_path)

    # prevent TRL from using apply_chat_template
    processing_class.apply_chat_template = apply_chat_template

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
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
    tokenizer = TransformerTokenizer(processing_class)
    eos_token_set = {processing_class.eos_token_id}
    eos_token_set.update([tokenizer.encode(eos_token)[0] for eos_token in EXTRA_EOS_TOKEN_LIST])

    decoder_factory = VLLMDecoderFactory(
        temperature=1.0,
        eos_token_set=eos_token_set,
        max_completion_length=max_turn_length,
        gpu_memory_utilization=0.3,
    )

    return (
        model_path,
        processor,
        tokenizer,
        decoder_factory,
        model,
        deepspeed,
    )

def load_batch_information(mode: Mode):
    num_processes = PartialState().num_processes

    # model updates every effective_batch_size
    # per device memory ~ batch_size x num_generations x max_conversation_length^\alpha
    # alpha = 2 for usual transformer
    # alpha = 1 for flash attention
    effective_batch_size = 16
    per_device_batch_size = 1
    num_generations = 8
    max_conversation_length = 8192
    max_turn_length = 8192

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
    from experiment.examples.discrete_logarithm.discrete_logarithm_env import DiscreteLogarithmEnv, DiscreteLogarithmSeed, SYSTEM_PROMPT

    # train 100 batches
    train_size = 100 * effective_batch_size
    # train data generation
    # total_num_steps = train_size x num_generations / effective_batch_size
    #       = 8000
    # no_points_per_step = effective_batch_size / num_generations
    def f(i: int) -> str:
        def generate_seed(p_seed: int) -> str:
            # find a prime p
            p: int = sympy.nextprime(p_seed)             # type: ignore
            # sample g and x
            g = np.random.randint(2, p)
            x = np.random.randint(1, p)
            h = pow(g, x, p)
            return DiscreteLogarithmSeed(g=g, h=h, p=p).model_dump_json()
        # make problem progressively harder
        # bit_size 10 -> 30
        MIN, MAX = 10, 30
        proportion: float = i / train_size
        expected_bit_size: float = MIN + (MAX - MIN) * proportion
        p_seed: int = np.random.geometric(1 / 2 ** expected_bit_size)
        return generate_seed(p_seed)
    
    data = LazyDataset[str](n=train_size, f=f)
    env_factory = DiscreteLogarithmEnv
    return (
        env_factory,
        data,
        SYSTEM_PROMPT,
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
        SYSTEM_PROMPT,
    ) = load_env_and_data(effective_batch_size=effective_batch_size)
    (
        model_path,
        processor,
        tokenizer,
        decoder_factory,
        model,
        deepspeed,
    ) = load_model_for_training(mode=mode, max_turn_length=max_turn_length, max_conversation_length=max_conversation_length)

    output_dir = f"mnt/output/discrete-logarithm-instruct-{os.path.basename(model_path)}-tl{max_turn_length}-cl{max_conversation_length}-b{effective_batch_size}-{uuid}-lora"

    (
        push_to_hub,
        hf_model,
        hf_token,
    ) = get_hf_info(output_dir)
    push_to_hub = push_to_hub and (mode != ModeDebug)



    # TRAIN
    train_dataset = data.map(lambda input_text: {"prompt": input_text})

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

        # floating point precision
        bf16=has_cuda or has_mps,
        tf32=has_cuda,
        optim="adamw_bnb_8bit",

    )

    system_prompt = SYSTEM_PROMPT.format(max_turn_length=max_turn_length, max_conversation_length=max_conversation_length)


    rollout_func = make_rollout_func(
        processor=processor,
        tokenizer=tokenizer,
        env_factory=env_factory,
        decoder_factory=decoder_factory,
        system_prompt=system_prompt,
        max_conversation_length=max_conversation_length,
    )
    reward_func = make_reward_func()


    trainer = GRPOTrainer(
        args=training_args,
        model=model, # type: ignore
        processing_class=tokenizer.processing_class,
        rollout_func=rollout_func,
        reward_funcs=reward_func, # type: ignore
        reward_processing_classes=tokenizer.processing_class,
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
