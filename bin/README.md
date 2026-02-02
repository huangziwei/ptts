# pmx: Podman wrapper for ptts

For the unfortunate few who still use Intel Mac, I have you covered. Use `pmx` when you need to run `ptts` inside Podman.

## Setup (Podman)

```bash
git clone https://github.com/huangziwei/ptts
cd ~/ptts
mkdir -p .cache/huggingface
chmod +x bin/pmx

brew install podman
podman --version # tested with 5.7.1
podman machine init --cpus 6 --memory 8192 --disk-size 60 --now pocket-tts

# Install project dependencies into .venv (required for `ptts` CLI)
PMX_OPTS="-p 1912:1912" ./bin/pmx uv sync
```

## Run pocket-tts (optional web UI)

```bash
./bin/pmx uv run pocket-tts serve --host 0.0.0.0 --port 1912
```

Open `http://localhost:1912`.

## ptts workflow (prefix with ./bin/pmx)

```bash
./bin/pmx uv run ptts ingest --input books/Some-Book.epub --out out/some-book
./bin/pmx uv run ptts sanitize --book out/some-book --overwrite
./bin/pmx uv run --with pocket-tts ptts synth --book out/some-book --max-chars 400 --pad-ms 150
./bin/pmx uv run ptts merge --book out/some-book --output out/some-book/some-book.m4b
```

### ffmpeg inside Podman

`ptts merge` requires `ffmpeg` on PATH. Install it in the same run (or bake a custom image):

```bash
./bin/pmx bash -lc 'apt-get update && apt-get install -y ffmpeg && uv run ptts merge --book out/some-book --output out/some-book/some-book.m4b'
```

## pmx tips

- `PMX_OPTS` adds container run options (e.g., port mappings).
- `PMX_RESET=1` recreates the named container (useful after changing `PMX_OPTS`).
- `PMX_RM=1` runs a one-off container and removes it on exit.
- `PMX_PRUNE=1` removes all `ptts-*` containers before running.
