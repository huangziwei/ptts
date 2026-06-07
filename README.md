# neb: Narrate English Books with [maneko](https://github.com/huangziwei/maneko)

![screenshot](.github/screenshot/player.png)

## Prerequisites

```bash
git clone https://github.com/huangziwei/neb
cd ~/neb

# neb's default TTS backend is maneko (https://github.com/huangziwei/maneko), a native
# Rust/candle engine installed from GitHub — building it needs a Rust toolchain:
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install project dependencies into .venv (builds maneko from GitHub; required for `neb`)
uv sync

# Model weights download automatically from Hugging Face (public `zwaiwng/maneko`) into
# this repo's .cache/ on first run — no account or token needed.

# Optional: also install the legacy torch/pocket-tts backend (select via NEB_TTS_BACKEND=torch)
# uv sync --extra torch
```

## TTS a book

### via the local web app

```bash
uv run neb play \
  --root out \
  --host 0.0.0.0 \
  --port 1912
```

Open `http://localhost:1912`.

### via CLI

#### 1) Ingest EPUB or TXT into raw chapters
```bash
uv run neb ingest \
  --input books/Some-Book.epub \
  --out out/some-book
```

Plain text input works the same way:
```bash
uv run neb ingest \
  --input books/Some-Book.txt \
  --out out/some-book
```

#### 2) Sanitize (clean) chapters
```bash
uv run neb sanitize \
  --book out/some-book \
  --overwrite
```

#### 3) Synthesize audio (TTS)
```bash
uv run neb synth \
  --book out/some-book \
  --max-chars 400 \
  --pad-ms 300
```

Voice cloning is wav-only — pass a local `.wav` clone source with `--voice` (create one
with `neb clone`). The default is `voices/ray.wav`:
```bash
uv run neb synth --book out/some-book --voice voices/ray.wav
```

To use the legacy torch/pocket-tts engine instead, install it (`uv sync --extra torch`)
and set `NEB_TTS_BACKEND=torch`.

Optional: add per-book pronunciation overrides at
`out/some-book/reading-overrides.json`:

```json
{
  "global": [
    { "base": "sutta", "reading": "soot-ta" },
    { "base": "sati", "reading": "sah-tee" },
    { "base": "satipatthana", "reading": "sah-tee-pat-ta-na" }
  ]
}
```

`base` uses whole-word matching by default (case-insensitive). Chapter-specific
overrides are also supported under `"chapters": { "<chapter-id>": { "replacements": [...] } }`.

#### 4) Merge to M4B
```bash
uv run neb merge \
  --book out/some-book \
  --output out/some-book/some-book.m4b
```

Merge auto-splits if the book is longer than 8 hours, keeping parts near-equal
and splitting only at chapter boundaries.

To override the split threshold:
```bash
uv run neb merge \
  --book out/some-book \
  --output out/some-book/some-book.m4b \
  --split-hours 8
```

`neb merge` requires `ffmpeg` on PATH (for macOS: `brew install ffmpeg`).
