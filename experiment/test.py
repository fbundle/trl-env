"""
Standalone example: continuous batching (paged attention) generation with Transformers.

This mirrors what GRPOTrainer does in the `use_transformers_paged` branch of
`_generate_single_turn`, but stripped down so the mechanics are easy to follow.

Requirements:
    pip install transformers>=4.46.0 torch accelerate

Key difference from standard generate():
    - Standard path:   left-pad all prompts into a single (B, T) tensor, pass to model.generate()
    - Paged path:      pass a list of variable-length token ID lists to model.generate_batch(),
                       which uses continuous batching internally — no manual padding needed.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


# ---------------------------------------------------------------------------
# 1. Load model and tokenizer
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-0.6B"  # swap for any causal LM

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
model.eval()

device = "cpu"
model = model.to(device)


# ---------------------------------------------------------------------------
# 2. Prepare prompts — intentionally variable length to show the advantage
# ---------------------------------------------------------------------------

raw_prompts = [
    "Explain what a transformer model is in one sentence.",
    "What is 2 + 2?",
    "Write a haiku about the ocean.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
]

# Apply chat template (returns list of strings)
templated = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for p in raw_prompts
]

# Tokenize WITHOUT padding — each element is a plain list of ints.
# This is the key difference: generate_batch accepts ragged sequences.
prompt_ids: list[list[int]] = tokenizer(templated)["input_ids"]

print("Prompt lengths (no padding applied):")
for i, ids in enumerate(prompt_ids):
    print(f"  prompt {i}: {len(ids)} tokens")


# ---------------------------------------------------------------------------
# 3. Build a GenerationConfig
#    (same object GRPOTrainer passes to generate_batch)
# ---------------------------------------------------------------------------

generation_config = GenerationConfig(
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    disable_compile=True,   # avoids recompilation overhead in training loops
)


# ---------------------------------------------------------------------------
# 4. Standard generate() path — for comparison
#    Requires explicit left-padding into a (B, T) tensor.
# ---------------------------------------------------------------------------

def standard_generate(model, tokenizer, prompt_ids, generation_config, device):
    max_len = max(len(ids) for ids in prompt_ids)
    pad_id = tokenizer.pad_token_id

    # Left-pad manually
    input_ids = torch.tensor(
        [[pad_id] * (max_len - len(ids)) + ids for ids in prompt_ids],
        device=device,
    )
    attention_mask = torch.tensor(
        [[0] * (max_len - len(ids)) + [1] * len(ids) for ids in prompt_ids],
        device=device,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

    # Slice off the prompt tokens to get only the completion
    completions = [
        output[max_len:].tolist()
        for output in output_ids
    ]
    return completions


# ---------------------------------------------------------------------------
# 5. Paged / continuous-batching path — generate_batch()
#    Accepts ragged list[list[int]] directly; no padding tensor needed.
# ---------------------------------------------------------------------------

def paged_generate(model, prompt_ids, generation_config):
    with torch.no_grad():
        # generate_batch() forces eval mode internally; restore after if needed.
        all_outputs = model.generate_batch(
            prompt_ids,
            generation_config=generation_config,
            progress_bar=False,
        )
        model.train()  # restore train mode (mirrors GRPOTrainer behaviour)

    # all_outputs is a dict keyed by sequence index; .generated_tokens holds
    # only the new tokens (prompt tokens are NOT included — no slicing needed).
    completion_ids: list[list[int]] = [
        all_outputs[i].generated_tokens for i in range(len(prompt_ids))
    ]
    return completion_ids


# ---------------------------------------------------------------------------
# 6. Run both and decode
# ---------------------------------------------------------------------------

print("\n--- Standard generate() ---")
std_completions = standard_generate(model, tokenizer, prompt_ids, generation_config, device)
for i, ids in enumerate(std_completions):
    text = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"[{i}] {text}\n")

print("\n--- generate_batch() (continuous batching) ---")
paged_completions = paged_generate(model, prompt_ids, generation_config)
for i, ids in enumerate(paged_completions):
    text = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"[{i}] {text}\n")


# ---------------------------------------------------------------------------
# 7. Notes on what generate_batch does NOT return (vs vLLM)
# ---------------------------------------------------------------------------
#
# generate_batch() returns an object with `.generated_tokens` (the new token
# IDs) but does NOT expose per-token log-probabilities. This is why:
#
#   logprobs = None   # in GRPOTrainer's paged branch
#
# Consequence: vLLM importance-sampling correction cannot be used with this
# backend, because there are no sampling logprobs to compare against the
# training model's logprobs.
#
# If you need logprobs for off-policy correction, use vLLM instead.