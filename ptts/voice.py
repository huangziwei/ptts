from __future__ import annotations

from pathlib import Path
from typing import Optional

BUILTIN_VOICES = {
    "alba": "hf://kyutai/tts-voices/alba-mackenna/casual.wav",
    "marius": "hf://kyutai/tts-voices/voice-donations/Selfie.wav",
    "javert": "hf://kyutai/tts-voices/voice-donations/Butter.wav",
    "jean": "hf://kyutai/tts-voices/ears/p010/freeform_speech_01.wav",
    "fantine": "hf://kyutai/tts-voices/vctk/p244_023.wav",
    "cosette": "hf://kyutai/tts-voices/expresso/ex04-ex02_confused_001_channel1_499s.wav",
    "eponine": "hf://kyutai/tts-voices/vctk/p262_023.wav",
    "azelma": "hf://kyutai/tts-voices/vctk/p303_023.wav",
}
DEFAULT_VOICE = "alba"


def resolve_voice_prompt(
    voice: Optional[str], base_dir: Optional[Path] = None
) -> str:
    if not voice:
        voice = DEFAULT_VOICE

    voice = voice.strip()
    if not voice:
        voice = DEFAULT_VOICE

    lowered = voice.lower()
    if lowered == "default":
        lowered = DEFAULT_VOICE

    if lowered in BUILTIN_VOICES:
        return BUILTIN_VOICES[lowered]

    if voice.startswith("hf://"):
        return voice

    candidate = Path(voice)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = (base_dir / candidate).resolve()

    if candidate.exists():
        return str(candidate)

    choices = ", ".join(sorted(BUILTIN_VOICES))
    raise ValueError(
        f"Voice prompt not found: {voice}. Use a built-in voice ({choices}), "
        "a wav file path, or an hf:// URL."
    )
