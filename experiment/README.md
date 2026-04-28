# EXPERIMENT

## INSTALL PLATFORM DEPENDENT PACKAGES

```shell
uv pip install flash-attn --no-build-isolation
uv pip install vllm --torch-backend=cu126
```

## INSTALL VLLM FOR MACOS

```shell
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
uv pip install -e .
```

## HOW TO SSH INTO A NODE RUNNING JOB

- submit job

```shell
echo "sleep 28800" | qsub -P $PBS_PROJECT -q normal -l select=1:ngpus=1 -l walltime=07:50:00 
```

- get hostname

```shell
qstat -ans <job_id> # or qstat -f <job_id>
```

- ssh

```shell
PBS_JOBID=<job_id> <hostname>
```