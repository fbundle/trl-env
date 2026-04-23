import os
import subprocess
import sys
from dotenv import load_dotenv


JOB_TEMPLATE = """
#!/usr/bin/env bash

#PBS -P {project_name}
#PBS -N log_{job_name}
#PBS -q normal
#PBS -j oe
#PBS -l {pbs_limit}
#PBS -l walltime={walltime}

cd $PBS_O_WORKDIR

module load cuda/12.6.2

export HF_HOME="$HOME/scratch/hf_home"

mkdir -p log
while true; do
    nvidia-smi > log/gpu_{job_name}.log
    sleep 5
done &

export NCCL_TIMEOUT=3600000  # 1 hour
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1

UV="$HOME/miniforge3/envs/test/bin/uv"
$UV run accelerate launch -m {recipe_module} train {uuid} |& tee log/run_{job_name}.log
"""

def write_file(path: str, content: str = ""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

def must_get_env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default=default)
    if value is None:
        raise RuntimeError(f"{name} must be set")
    return value

def get_module_path(file_path: str) -> str:
    file_path = os.path.relpath(file_path, os.getcwd())
    file_path, _ = os.path.splitext(file_path)
    module_path = file_path.replace("/", ".")
    return module_path

def main(recipe_file: str):
    load_dotenv()

    project_name = must_get_env("PBS_PROJECT")
    pbs_limit = must_get_env("PBS_LIMIT", "select=1:ngpus=1")
    walltime = must_get_env("PBS_WALLTIME", "23:50:00")

    recipe_module = get_module_path(recipe_file)
    recipe_name = recipe_module.split(".")[-1]
    uuid = must_get_env("UUID", "debug")
    job_name = f"{recipe_name}_{uuid}"

    

    job_file = f"mnt/job/{job_name}.pbs"
    write_file(
        path=job_file,
        content=JOB_TEMPLATE.format(
            project_name=project_name,
            job_name=job_name,
            pbs_limit=pbs_limit,
            walltime=walltime,
            recipe_module=recipe_module,
            uuid=uuid,
        ),
    )

    write_file(f"log/run_{job_name}.log")
    write_file(f"log/gpu_{job_name}.log")

    result = subprocess.run(["qsub", job_file])
    if result.returncode != 0:
        raise RuntimeError(f"qsub failed with code {result.returncode}")



if __name__ == "__main__":
    main(sys.argv[1])
