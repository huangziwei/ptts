#!/usr/bin/env python3
"""
Stop/restartable long-form Pocket-TTS generator.

Outputs:
  out/
    manifest.json
    segments/000001.wav ...
    concat.txt
    chapters.ffmeta

Resume behavior:
  - If manifest.json exists, it is reused (chunking is stable).
  - Any segment WAV that already exists and is valid is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from pocket_tts import TTSModel  # per upstream README :contentReference[oaicite:1]{index=1}


# ----------------------------
# Text chunking
# ----------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def read_text(path: Path) -> str:
    s = path.read_text(encoding="utf-8", errors="strict")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Normalize whitespace but keep paragraph breaks.
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip() + "\n"


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def split_paragraphs(text: str) -> List[str]:
    # Split on blank lines (one or more).
    paras = re.split(r"\n\s*\n+", text.strip())
    return [p.strip().replace("\n", " ") for p in paras if p.strip()]


def split_sentences(paragraph: str) -> List[str]:
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    parts = _SENT_SPLIT_RE.split(paragraph)
    return [p.strip() for p in parts if p.strip()]


def pack_into_chunks(units: List[str], max_chars: int) -> List[str]:
    """
    Pack a list of sentence-like units into chunks <= max_chars when possible.
    Falls back to word splitting if a single unit is too large.
    """
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append(" ".join(buf).strip())
            buf = []
            buf_len = 0

    for u in units:
        u = u.strip()
        if not u:
            continue

        if len(u) > max_chars:
            # Flush current buffer, then hard-split this long unit by words.
            flush()
            words = u.split()
            wbuf: List[str] = []
            wlen = 0
            for w in words:
                add = (1 if wbuf else 0) + len(w)
                if wbuf and (wlen + add) > max_chars:
                    chunks.append(" ".join(wbuf))
                    wbuf = [w]
                    wlen = len(w)
                else:
                    if wbuf:
                        wlen += 1 + len(w)
                    else:
                        wlen = len(w)
                    wbuf.append(w)
            if wbuf:
                chunks.append(" ".join(wbuf))
            continue

        add = (1 if buf else 0) + len(u)
        if buf and (buf_len + add) > max_chars:
            flush()
            buf = [u]
            buf_len = len(u)
        else:
            if buf:
                buf_len += 1 + len(u)
            else:
                buf_len = len(u)
            buf.append(u)

    flush()
    return chunks


def make_chunks(text: str, max_chars: int) -> List[str]:
    chunks: List[str] = []
    for para in split_paragraphs(text):
        sents = split_sentences(para)
        if not sents:
            continue
        chunks.extend(pack_into_chunks(sents, max_chars=max_chars))
    # Ensure each chunk ends with punctuation/newline-like boundary for prosody.
    cleaned: List[str] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if c[-1] not in ".!?":
            c += "."
        cleaned.append(c)
    return cleaned


# ----------------------------
# WAV IO utilities
# ----------------------------

def tensor_to_int16(audio: torch.Tensor) -> torch.Tensor:
    """
    Pocket-TTS README says returned audio is PCM data in a 1D torch tensor. :contentReference[oaicite:2]{index=2}
    Make this robust to float or int tensors.
    """
    a = audio.detach().cpu().flatten().contiguous()

    if a.dtype in (torch.float16, torch.float32, torch.float64):
        # Heuristic: if values look like [-1, 1], scale to int16.
        max_abs = float(a.abs().max().item()) if a.numel() else 0.0
        if max_abs <= 1.5:
            a = torch.clamp(a, -1.0, 1.0)
            a = torch.round(a * 32767.0).to(torch.int16)
        else:
            a = torch.round(a).to(torch.int16)
    elif a.dtype != torch.int16:
        a = a.to(torch.int16)

    return a


def write_wav_mono_16k_or_24k(path: Path, samples_i16: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    arr = samples_i16.numpy()  # requires numpy via torch; typical torch installs include it
    data = arr.tobytes()

    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(data)

    tmp.replace(path)


def wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    if rate <= 0:
        return 0
    return int(round(frames * 1000.0 / rate))


def is_valid_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getnframes() > 0
    except Exception:
        return False


# ----------------------------
# Manifest + outputs
# ----------------------------

def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_concat_file(segment_paths: List[Path], concat_path: Path, base_dir: Path) -> None:
    lines = []
    for p in segment_paths:
        rel = p.relative_to(base_dir).as_posix()
        # ffmpeg concat demuxer format
        lines.append(f"file '{rel}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_chapters_ffmeta(durations_ms: List[int], ffmeta_path: Path, chapter_title_prefix: str = "Chunk") -> None:
    """
    Generates a simple FFMETADATA1 file with one chapter per chunk.
    ffmpeg can import chapters using -map_chapters (see ffmpeg docs/patch history). :contentReference[oaicite:3]{index=3}
    """
    out = [";FFMETADATA1"]
    t = 0
    for i, d in enumerate(durations_ms, start=1):
        start = t
        end = t + max(d, 1)
        out.append("")
        out.append("[CHAPTER]")
        out.append("TIMEBASE=1/1000")
        out.append(f"START={start}")
        out.append(f"END={end}")
        out.append(f"title={chapter_title_prefix} {i}")
        t = end
    ffmeta_path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, type=Path, help="Input UTF-8 .txt file")
    ap.add_argument("--voice", required=True, help="Voice prompt: local wav path or hf://... URL")
    ap.add_argument("--out", required=True, type=Path, help="Output directory, e.g. out/book1")
    ap.add_argument("--max-chars", type=int, default=800, help="Max characters per chunk (default: 800)")
    ap.add_argument("--pad-ms", type=int, default=150, help="Silence to append to each chunk in ms (default: 150)")
    ap.add_argument("--rechunk", action="store_true", help="Ignore existing manifest and rechunk the input text")
    args = ap.parse_args()

    out_dir: Path = args.out
    seg_dir = out_dir / "segments"
    manifest_path = out_dir / "manifest.json"
    concat_path = out_dir / "concat.txt"
    chapters_path = out_dir / "chapters.ffmeta"

    out_dir.mkdir(parents=True, exist_ok=True)

    text = read_text(args.text)
    text_hash = sha256_str(text)

    if manifest_path.exists() and not args.rechunk:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("text_sha256") != text_hash:
            sys.stderr.write(
                "manifest.json exists but input text hash differs. "
                "Run with --rechunk to regenerate manifest.\n"
            )
            return 2
        chunks: List[str] = manifest["chunks"]
        sample_rate = int(manifest.get("sample_rate", 24000))
        pad_ms = int(manifest.get("pad_ms", args.pad_ms))
    else:
        chunks = make_chunks(text, max_chars=args.max_chars)
        # Load model once to read authoritative sample_rate.
        tts_model = TTSModel.load_model()
        sample_rate = int(tts_model.sample_rate)
        pad_ms = int(args.pad_ms)
        manifest = {
            "created_unix": int(time.time()),
            "text_path": str(args.text),
            "text_sha256": text_hash,
            "voice": args.voice,
            "max_chars": int(args.max_chars),
            "pad_ms": pad_ms,
            "sample_rate": sample_rate,
            "chunks": chunks,
            "durations_ms": [None] * len(chunks),
        }
        atomic_write_json(manifest_path, manifest)

    # Load model + voice state once (recommended upstream). :contentReference[oaicite:4]{index=4}
    tts_model = TTSModel.load_model()
    voice_state = tts_model.get_state_for_audio_prompt(args.voice)

    pad_samples = int(round(sample_rate * (pad_ms / 1000.0)))
    pad_tensor = torch.zeros(pad_samples, dtype=torch.int16) if pad_samples > 0 else None

    segment_paths: List[Path] = []
    durations_ms: List[int] = []

    for idx, chunk_text in enumerate(chunks, start=1):
        seg_path = seg_dir / f"{idx:06d}.wav"
        segment_paths.append(seg_path)

        if seg_path.exists() and is_valid_wav(seg_path):
            dms = wav_duration_ms(seg_path)
            durations_ms.append(dms)
            continue

        audio = tts_model.generate_audio(voice_state, chunk_text)
        a16 = tensor_to_int16(audio)

        if pad_tensor is not None and pad_tensor.numel() > 0:
            a16 = torch.cat([a16, pad_tensor], dim=0)

        write_wav_mono_16k_or_24k(seg_path, a16, sample_rate=sample_rate)
        dms = int(round(a16.numel() * 1000.0 / sample_rate))
        durations_ms.append(dms)

        # Persist progress for restartability.
        manifest["durations_ms"][idx - 1] = dms
        atomic_write_json(manifest_path, manifest)

    build_concat_file(segment_paths, concat_path, base_dir=out_dir)
    build_chapters_ffmeta(durations_ms, chapters_path, chapter_title_prefix="Chunk")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
