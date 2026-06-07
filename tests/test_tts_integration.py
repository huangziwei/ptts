"""End-to-end integration tests that drive neb's synth pipeline with synthetic data.

These tests do not mock the TTS engine. For each importable backend — maneko (the
default) and torch/pocket-tts (the optional extra) — they exercise the whole neb
pipeline: manifest prep, chunking, voice resolution, model load, audio generation,
wav writing. A backend is skipped when it (or its weights) isn't installed, so the
suite stays green in environments where only one (or neither) engine is present.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from neb import tts

_BACKENDS: list[str] = []
if tts.maneko is not None:
    _BACKENDS.append("maneko")
if tts.TTSModel is not None and tts.torch is not None:
    _BACKENDS.append("torch")

# When no engine is installed, still emit a visible skip (rather than 0 collected items).
_BACKEND_PARAMS = [pytest.param(name) for name in _BACKENDS] or [
    pytest.param(
        "maneko",
        marks=pytest.mark.skip(reason="no TTS backend installed (maneko or torch)"),
    )
]


def _make_clone_source(path: Path, *, seconds: float = 2.0, sr: int = 24000) -> None:
    """Write a short synthetic mono wav to use as a voice-clone source.

    Voice cloning is wav-only now; a synthetic tone keeps the test hermetic (no
    dependency on the gitignored voices/ directory being populated).
    """
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    tone = 0.2 * np.sin(2.0 * np.pi * 150.0 * t).astype(np.float32)
    tts.write_wav_mono_16k_or_24k(path, tts.floats_to_int16(tone), sr)


@pytest.mark.parametrize("backend_name", _BACKEND_PARAMS)
@pytest.mark.parametrize(
    "language, layers, text",
    [
        ("english", 6, "Hello world."),
        ("german", 24, "Guten Tag."),
    ],
)
def test_synthesize_produces_wav_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
    language: str,
    layers: int,
    text: str,
) -> None:
    monkeypatch.setenv("NEB_TTS_BACKEND", backend_name)

    voice_path = tmp_path / "voices" / "tone.wav"
    _make_clone_source(voice_path)
    voice = voice_path.relative_to(tmp_path).as_posix()  # "voices/tone.wav"

    chapter = tts.ChapterInput(
        index=1,
        id="0001-smoke",
        title="Smoke",
        text=text,
        path=None,
    )
    out_dir = tmp_path / "tts"

    rc = tts.synthesize(
        chapters=[chapter],
        voice=voice,
        out_dir=out_dir,
        max_chars=200,
        pad_ms=100,
        chunk_mode="sentence",
        rechunk=True,
        base_dir=tmp_path,
        language=language,
        layers=layers,
    )
    assert rc == 0

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["language"] == language
    assert manifest["layers"] == layers
    assert manifest["voice"] == voice

    seg_path = out_dir / "segments" / chapter.id / "000001.wav"
    assert seg_path.exists(), f"expected wav at {seg_path}"
    assert seg_path.stat().st_size > 0
    with wave.open(str(seg_path), "rb") as wf:
        assert wf.getnframes() > 0

    durations = manifest["chapters"][0]["durations_ms"]
    assert durations and durations[0] and durations[0] > 0
