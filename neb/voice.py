from __future__ import annotations

from pathlib import Path
from typing import Optional

# Voice cloning is wav-only: the narrator is always a local `.wav` clone source
# (under voices/) or an `hf://…/foo.wav` URL. pocket-tts's hosted stock voices
# have been removed — there are no built-in named voices.
DEFAULT_VOICE = "voices/ray.wav"


def resolve_voice_prompt(
    voice: Optional[str], base_dir: Optional[Path] = None
) -> str:
    if not voice:
        voice = DEFAULT_VOICE

    voice = voice.strip()
    if not voice:
        voice = DEFAULT_VOICE

    if voice.lower() == "default":
        voice = DEFAULT_VOICE

    if voice.startswith("hf://"):
        return voice

    candidate = Path(voice)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = (base_dir / candidate).resolve()

    if candidate.exists():
        return str(candidate)

    raise ValueError(
        f"Voice prompt not found: {voice}. Pass a wav file path (e.g. voices/ray.wav) "
        "or an hf:// URL. Create a clone source with `neb clone`."
    )
