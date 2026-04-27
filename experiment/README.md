# EXPERIMENT

## INSTALL FLASH ATTENTION 

- Flash Attention is not cross platform, hence need to use `uv pip`

```shell
uv pip install flash-attn --no-build-isolation
uv pip install vllm --torch-backend=auto
```

## INSTALL VLLM

- linux

```shell
uv pip install vllm --torch-backend=auto
```

- macos

```shell
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
uv pip install -e .
```