# neb: Narrate English Books with [Pocket-TTS](https://huggingface.co/kyutai/pocket-tts)

![screenshot](.github/screenshot/player.png)

## Prerequisites

```bash
git clone https://github.com/huangziwei/neb
cd ~/neb

# Install project dependencies into .venv (required for `neb` CLI)
uv sync

## To use voice cloning, you need to accept the terms via browser at https://huggingface.co/kyutai/pocket-tts
## Then you need to save the access token with correct permissions (I ticked everything in Repositories and Inference)
## This step can be skipped if you don't need voice cloning
# uvx hf auth login
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
uv run --with pocket-tts neb synth \
  --book out/some-book \
  --max-chars 400 \
  --pad-ms 300
```

By default, `neb synth` uses the built-in voice `alba`. To choose a built-in voice
explicitly (or use a cloned wav), pass `--voice`:
```bash
uv run --with pocket-tts neb synth --book out/some-book --voice alba
uv run --with pocket-tts neb synth --book out/some-book --voice voices/ray.wav
```

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
