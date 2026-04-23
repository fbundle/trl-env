import os
import sys

from huggingface_hub import login, upload_large_folder
from dotenv import load_dotenv

def upload(output_dir: str):
    load_dotenv()
    hf_user = os.environ.get("HF_USER", None)
    if hf_user is None:
        raise RuntimeError("HF_USER must be set")

    repo_id = hf_user + "/" + os.path.basename(output_dir)

    login()
    upload_large_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
        # ignore_patterns=["checkpoint-*"],
    )


if __name__ == "__main__":
    upload(sys.argv[1].rstrip("/"))
