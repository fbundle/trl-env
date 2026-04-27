import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

model_path = "Qwen/Qwen3.5-0.8B"

# 1. LOAD TRAINING MODEL (PyTorch)

training_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
)
# 2. LOAD VLLM ENGINE
llm = LLM(
    model=model_path,
    dtype="bfloat16",
    gpu_memory_utilization=0.4,
    enable_prefix_caching=True,
)


# 3. GENERATE WITH LOGPROBS
sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=512,
    logprobs=1,
)

outputs = llm.generate(["Find x such that 2^x = 3 (mod 5)"], sampling_params)
for output in outputs:
    tokens = output.outputs[0].token_ids
    logprobs = [list(lp.values())[0].logprob for lp in output.outputs[0].logprobs]
    print(tokens, logprobs)

# 4. UPDATE WEIGHTS
def sync_weights(llm, training_model):
    named_params = {name: param.data for name, param in training_model.named_parameters()}
    llm.collective_rpc(
        "update_weights_from_dict",
        kwargs=dict(named_params=named_params),
    )

sync_weights(llm, training_model)

# 5. GENERATE AGAIN - prefix cache preserved from step 3
prompts = [
    "Find x such that 2^x = 3 (mod 5)",
    "Find x such that 2^x = 5 (mod 7)",
    "Find x such that 2^x = 1 (mod 11)",
]
outputs = llm.generate(prompts, sampling_params)
for o in outputs:
    tokens = o.outputs[0].token_ids
    logprobs = [list(lp.values())[0].logprob for lp in o.outputs[0].logprobs]
    print(o.outputs[0].text)
    print(logprobs)
