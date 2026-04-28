import os
import sys

from huggingface_hub import HfApi, login, upload_large_folder
from dotenv import load_dotenv

load_dotenv()

def get_hf_info(output_dir: str) -> tuple[bool, str, str]:
    hf_user = os.environ.get("HF_USER", default=None)
    hf_token = os.environ.get("HF_TOKEN", default=None)
    if hf_user is None or hf_token is None:
        return False, "", ""
    
    hf_model = hf_user + "/" + os.path.basename(output_dir)
    return True, hf_model, hf_token

def upload(output_dir: str):
    push_to_hub, hf_model, hf_token = get_hf_info(output_dir)

    if not push_to_hub:
        raise RuntimeError("HF_USER and HF_TOKEN must be set")
        return

    api = HfApi(token=hf_token)
    api.upload_large_folder(
        folder_path=output_dir,
        repo_id=hf_model,
        repo_type="model",
        ignore_patterns=["checkpoint-*"],
    )


if __name__ == "__main__":
    upload(sys.argv[1].rstrip("/"))
