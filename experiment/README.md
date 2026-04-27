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