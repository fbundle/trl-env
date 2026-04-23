
from dataclasses import dataclass
from typing import Any, Callable, Literal

from .model import TransformerModel

from .dataset import LazyDataset
from .environment import Env
from .processor import Processor


type Mode = Literal["prepare", "train"]
ModePrepare: Mode = "prepare"
ModeTrain: Mode = "train"

@dataclass
class TrainConfig:
    mode: Mode
    output_dir: str
    env_factory: Callable[[], Env]
    system_prompt: str
    processor: Processor
    model: TransformerModel
    data: LazyDataset[str]

    # TRAINING HYPERPARAMS
    # per device memory ~ batch_size x num_generations x max_conversation_length^\alpha
    per_device_batch_size: int
    num_generations: int
    max_turn_length: int
    max_conversation_length: int
    # gradient accumulation in every: num_processes x per_device_batch_size x gradient_accumulation_steps
    gradient_accumulation_steps: int

    # OTHERS
    deepspeed: str | None = None
    generation_kwargs: dict[str, Any] | None = None
    train_config_kwargs: dict[str, Any] | None = None
    push_to_hub: bool = False

    # SAVE AND LOG
    save_every_seconds: int = 1 * 3600    # by default, save every 1 hour
    log_every_seconds: int = 0            # by default, log immediately after step_end

    # LOGGER
    logger: Callable[[str], None] | None = None


    