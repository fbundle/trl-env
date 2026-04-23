
import platform

import torch
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from .batch_rollout import make_reward_func, make_rollout_func
from .trainer_config import TrainConfig
from .trainer_util import Callback, get_hf_info

from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig



from dotenv import load_dotenv

def train(config: TrainConfig):
    load_dotenv()
    # train
    if platform.system() == "Linux" and platform.machine() == "x86_64":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training on Linux x86_64 but not found.")

    push_to_hub, hf_model, hf_token = get_hf_info(config.output_dir)
    push_to_hub = push_to_hub and config.push_to_hub

    generation_kwargs = {}
    if config.generation_kwargs is not None:
        generation_kwargs.update(config.generation_kwargs)
    train_config_kwargs = {}
    if config.train_config_kwargs is not None:
        train_config_kwargs.update(config.train_config_kwargs)


    def apply_chat_template(*args, **kwargs):
        raise RuntimeError("GRPO must not use apply_chat_template")

    # prevent TRL from using apply_chat_template
    config.model.tokenizer.apply_chat_template = apply_chat_template

    if config.mode == "prepare":
        # in prepare mode, always generate in full to monitor GPU memory
        generation_kwargs["min_new_tokens"] = config.max_conversation_length

    # DATASET
    train_dataset = config.data.map(
        lambda input_text: {"prompt": input_text}
    )

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    callbacks: list[TrainerCallback] = [Callback(config=config)]

    training_args = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=1,
        deepspeed=config.deepspeed,

        # TODO - since we do our own rollout, I wonder if we still need these

        per_device_train_batch_size=config.per_device_batch_size,
        num_generations=config.num_generations,
        max_completion_length=config.max_conversation_length,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

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

        use_vllm=False, # may change to true in the future
        vllm_mode="colocate",

        gradient_checkpointing=True,

        # others
        generation_kwargs=generation_kwargs,
        **train_config_kwargs,
    )

    rollout_func = make_rollout_func(
        model=config.model,
        processor=config.processor,
        env_factory=config.env_factory,
        system_prompt=config.system_prompt,
        max_conversation_length=config.max_conversation_length,
    )
    reward_func = make_reward_func()

    trainer = GRPOTrainer(
        args=training_args,
        model=config.model.model,
        processing_class=config.model.tokenizer,
        rollout_func=rollout_func,
        reward_funcs=reward_func, # type: ignore
        reward_processing_classes=config.model.tokenizer,
        train_dataset=train_dataset, # type: ignore
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=get_last_checkpoint(config.output_dir))










