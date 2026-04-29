import json
import os
import sys

import shutil

from huggingface_hub import hf_hub_download
import mlx_lm
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

class Context(BaseModel):
    overwrite: bool = False

ctx = Context()

def get_local_path(checkpoint_path: str, name: str) -> str:
    path = os.path.join(checkpoint_path, name)
    if not ctx.overwrite and os.path.exists(path):
        return path
    # download from huggingface
    return hf_hub_download(
        repo_id=checkpoint_path,
        filename=name,
    )

def merge_model(checkpoint_path: str, cache_dir: str = "mnt/model_cache") -> str:
    model_path = os.path.join("mnt/model_cache", checkpoint_path)
    if not ctx.overwrite and os.path.exists(model_path):
        return model_path

    # load base model
    adapter_config_path = get_local_path(checkpoint_path, "adapter_config.json")
    adapter_config = json.loads(open(adapter_config_path).read())
    base_model_path = adapter_config["base_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

    # load adapter model
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    # merge
    model = model.merge_and_unload() # type: ignore

    # save
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

    # overwrite config with base model config
    base_model.config.save_pretrained(model_path)

    return model_path

def main(checkpoint_path: str):
    model_path = merge_model(checkpoint_path)


    mlx_model_path = os.path.join("mnt/output_mlx", checkpoint_path)
    if not ctx.overwrite and os.path.exists(mlx_model_path):
        return mlx_model_path
    if os.path.exists(mlx_model_path):
        shutil.rmtree(mlx_model_path)
    mlx_lm.convert(
        hf_path=model_path,
        mlx_path=mlx_model_path,
        quantize=False,
    )
    print("mlx_model", mlx_model_path)

    mlx_model_path = os.path.join("mnt/output_mlx_quantize", checkpoint_path)
    if not ctx.overwrite and os.path.exists(mlx_model_path):
        return mlx_model_path
    if os.path.exists(mlx_model_path):
        shutil.rmtree(mlx_model_path)
    mlx_lm.convert(
        hf_path=model_path,
        mlx_path=mlx_model_path,
        quantize=True,
    )
    print("mlx_model_quantize", mlx_model_path)

    

if __name__ == "__main__":
    model_path = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "overwrite":
        ctx.overwrite = True
    
    main(model_path)
        
