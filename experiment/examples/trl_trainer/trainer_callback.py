
import time

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


SHOULD_SAVE, SHOULD_LOG = 0, 1
FALSE, TRUE = 0, 1

class TimeBasedLogSaveCallback(TrainerCallback):
    def __init__(self, save_every_seconds: int = 3600, log_every_seconds: int = 0):
        self.save_every_seconds = save_every_seconds
        self.log_every_seconds = log_every_seconds
        
        self.last_save_time = time.time()
        self.last_log_time = time.time()
        self.last_global_step = -1
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step <= self.last_global_step:
            return
        self.last_global_step = state.global_step

        RANK, WORLD_SIZE = args.process_index, args.world_size
        DEVICE = args.device

        # trigger log and save from rank 0 and broadcast to everyone else
        # sync_flags: [SHOULD_SAVE, SHOULD_LOG]
        sync_flags = torch.tensor([FALSE, FALSE], dtype=torch.long, device=DEVICE)
        
        if RANK == 0:
            current_time = time.time()
            if current_time - self.last_save_time >= self.save_every_seconds:
                sync_flags[SHOULD_SAVE] = TRUE
            if current_time - self.last_log_time >= self.log_every_seconds:
                sync_flags[SHOULD_LOG] = TRUE
            
        
        if torch.distributed.is_initialized() and WORLD_SIZE > 1:
            torch.distributed.broadcast(sync_flags, src=0)
        
        if sync_flags[SHOULD_SAVE] == TRUE:
            control.should_save = True
            self.last_save_time = time.time()

        if sync_flags[SHOULD_LOG] == TRUE:
            control.should_log = True
            self.last_log_time = time.time()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if torch.cuda.is_available():
            # Get current and peak memory
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3

            print(
                f"\n[GPU Memory] Step {state.global_step}: "
                f"Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {peak:.2f}GB"
            )

            # Reset peak memory stats for the next step if you want per-step peak
            # torch.cuda.reset_peak_memory_stats()