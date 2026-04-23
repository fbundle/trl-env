# EXPERIMENT

## INSTALL FLASH ATTENTION

- Flash Attention is not cross platform, hence need to use `uv pip`

```shell
uv pip install flash-attn --no-build-isolation
```

## INSTALL SGLANG ON MACOS

[19137](https://github.com/sgl-project/sglang/issues/19137)

```
# Install ffmpeg
brew install ffmpeg

# Install uv
brew install uv

# Clone the repository
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Create and activate a virtual environment
uv venv -p 3.11 my-venv
source my-venv/bin/activate

# Install the Python packages
uv pip install --upgrade pip
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
uv pip install -e "python[all_mps]"
```