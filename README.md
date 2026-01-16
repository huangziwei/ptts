# Narrate English books with Pocket-TTS

## Preparation

As I am still using a vintage Intel MacBook Pro, many cool tools that use latest PyTorch or JAX cannot be run directly. I resort to using VMs via Podman:

```bash
git clone <this-repo>
cd ~/pocket
mkdir -p .uv-cache .cache/huggingface

brew install podman
podman --version # tested with 5.7.1
podman machine init --cpus 6 --memory 8192 --disk-size 60 --now pocket-tts

# To use voice cloning, you need to accept the terms via browser at https://huggingface.co/kyutai/pocket-tts
# Then you need to save the access token with correct permissions (I ticked everything in Repositories and Inference)
# This step can be skipped if no need for voice cloning
podman run --rm -it \
  -v "$PWD":/work -w /work \
  -e HF_HOME=/work/.cache/huggingface \
  -e UV_CACHE_DIR=/work/.uv-cache \
  ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
  uvx hf auth login

# Run once and test if it works in the web ui
podman run --rm -it \
  -p 8000:8000 \
  -v "$PWD":/work -w /work \
  -e HF_HOME=/work/.cache/huggingface \
  -e UV_CACHE_DIR=/work/.uv-cache \
  ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
  uvx pocket-tts serve --host 0.0.0.0 --port 8000
```

## TTS a book

```bash
podman run --rm -it \
  -v "$PWD":/work -w /work \
  -e HF_HOME=/work/.cache/huggingface \
  -e UV_CACHE_DIR=/work/.uv-cache \
  ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
  uv run --no-project --with pocket-tts longform_pockettts.py \
    --text book.txt \
    --voice voice/ray.wav \
    --out out/book \
    --max-chars 800 \
    --pad-ms 150
```