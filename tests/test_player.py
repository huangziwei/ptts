import shutil
from pathlib import Path
import json

from fastapi.testclient import TestClient
import pytest

from neb import player


def _make_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='neb'\n", encoding="utf-8")
    root_dir = repo_root / "out"
    root_dir.mkdir()
    return repo_root, root_dir


def test_parse_clone_time_accepts_seconds_and_timecode() -> None:
    assert player._parse_clone_time("0", "start", allow_zero=True) == "0"
    assert player._parse_clone_time("1.250", "start", allow_zero=True) == "1.25"
    assert player._parse_clone_time("01:02", "start", allow_zero=True) == "62"
    assert player._parse_clone_time("00:01:02.5", "start", allow_zero=True) == "62.5"


def test_normalize_reading_overrides() -> None:
    raw = [
        {"base": "sutta", "reading": "soot-ta"},
        {"base": "", "reading": "ignored"},
        ["sati", "sah-tee"],
        {"base": "re:satipatth?ana", "reading": "sah-tee-pat-ta-na"},
        {"pattern": r"samyutta", "reading": "samyukta", "mode": "first"},
        {"base": "Sati", "reading": "sah-tee", "case_sensitive": True},
        "invalid",
    ]
    cleaned = player._normalize_reading_overrides(raw)
    assert cleaned == [
        {"base": "sutta", "reading": "soot-ta"},
        {"base": "sati", "reading": "sah-tee"},
        {"pattern": "satipatth?ana", "reading": "sah-tee-pat-ta-na"},
        {"pattern": r"samyutta", "reading": "samyukta", "mode": "first"},
        {"base": "Sati", "reading": "sah-tee", "case_sensitive": True},
    ]


@pytest.mark.parametrize(
    "raw,allow_zero,message",
    [
        ("", False, "required"),
        ("-1", True, ">= 0"),
        ("0", False, "> 0"),
        ("00:61", True, "below 60"),
        ("a:b:c", True, "timecode"),
    ],
)
def test_parse_clone_time_rejects_invalid_values(
    raw: str,
    allow_zero: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        player._parse_clone_time(raw, "duration", allow_zero=allow_zero)


def test_clone_preview_and_save_reuses_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root, root_dir = _make_repo(tmp_path)
    source = repo_root / "voice-source.mp3"
    source.write_bytes(b"source")
    source_path = str(source)
    calls: list[tuple[Path, Path, str, str]] = []

    def fake_run_clone_ffmpeg(
        input_path: Path,
        output_path: Path,
        start: str,
        duration: str,
    ) -> None:
        calls.append((input_path, output_path, start, duration))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFFFAKEWAVE")

    real_which = shutil.which

    def fake_which(name: str) -> str | None:
        if name == "ffmpeg":
            return "/usr/bin/ffmpeg"
        return real_which(name)

    monkeypatch.setattr(player, "_run_clone_ffmpeg", fake_run_clone_ffmpeg)
    monkeypatch.setattr(player.shutil, "which", fake_which)

    app = player.create_app(root_dir)
    client = TestClient(app)

    preview = client.post(
        "/api/voices/clone/preview",
        json={"source": source_path, "start": "1", "duration": "3.5"},
    )
    assert preview.status_code == 200
    preview_payload = preview.json()
    assert preview_payload["status"] == "ready"
    assert preview_payload["suggested_name"] == "voice-source"
    assert preview_payload["start"] == "1"
    assert preview_payload["duration"] == "3.5"
    assert len(calls) == 1

    preview_audio = client.get("/api/voices/clone/preview-audio")
    assert preview_audio.status_code == 200
    assert preview_audio.headers["content-type"].startswith("audio/wav")

    save = client.post(
        "/api/voices/clone/save",
        json={
            "source": source_path,
            "start": "1",
            "duration": "3.5",
            "name": "NARRATOR",
            "gender": "female",
            "language": "german",
        },
    )
    assert save.status_code == 200
    save_payload = save.json()
    assert save_payload["status"] == "saved"
    assert save_payload["used_preview"] is True
    assert save_payload["gender"] == "female"
    assert save_payload["language"] == "german"
    assert save_payload["display_name"] == "Narrator"
    assert save_payload["voice"] == {"label": "Narrator", "value": "voices/narrator.wav"}
    assert len(calls) == 1
    assert (repo_root / "voices" / "narrator.wav").exists()
    voices_payload = client.get("/api/voices").json()
    local_entry = next(
        item for item in voices_payload["local"] if item["value"] == "voices/narrator.wav"
    )
    assert local_entry["gender"] == "female"
    assert local_entry["language"] == "german"
    assert local_entry["label"] == "Narrator"

    conflict = client.post(
        "/api/voices/clone/save",
        json={
            "source": source_path,
            "start": "1",
            "duration": "3.5",
            "name": "narrator",
        },
    )
    assert conflict.status_code == 409

    overwrite = client.post(
        "/api/voices/clone/save",
        json={
            "source": source_path,
            "start": "1",
            "duration": "3.5",
            "name": "narrator",
            "overwrite": True,
        },
    )
    assert overwrite.status_code == 200
    overwrite_payload = overwrite.json()
    assert overwrite_payload["overwrote"] is True


def test_clone_preview_rejects_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, root_dir = _make_repo(tmp_path)
    real_which = shutil.which

    def fake_which(name: str) -> str | None:
        if name == "ffmpeg":
            return "/usr/bin/ffmpeg"
        return real_which(name)

    monkeypatch.setattr(player.shutil, "which", fake_which)

    app = player.create_app(root_dir)
    client = TestClient(app)
    missing = str(repo_root / "missing-file.mp3")
    response = client.post(
        "/api/voices/clone/preview",
        json={"source": missing, "start": "0", "duration": "3"},
    )
    assert response.status_code == 400
    assert "Input file not found" in response.json()["detail"]


def test_clone_preview_rejects_relative_local_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root_dir = _make_repo(tmp_path)
    real_which = shutil.which

    def fake_which(name: str) -> str | None:
        if name == "ffmpeg":
            return "/usr/bin/ffmpeg"
        return real_which(name)

    monkeypatch.setattr(player.shutil, "which", fake_which)

    app = player.create_app(root_dir)
    client = TestClient(app)
    response = client.post(
        "/api/voices/clone/preview",
        json={"source": "relative-file.mp3", "start": "0", "duration": "3"},
    )
    assert response.status_code == 400
    assert "must be absolute" in response.json()["detail"]


def test_voice_metadata_update_and_clear(tmp_path: Path) -> None:
    repo_root, root_dir = _make_repo(tmp_path)
    voices_dir = repo_root / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    voice_path = voices_dir / "sample.wav"
    voice_path.write_bytes(b"RIFFFAKEWAVE")

    app = player.create_app(root_dir)
    client = TestClient(app)

    save = client.post(
        "/api/voices/metadata",
        json={
            "voice": "voices/sample.wav",
            "gender": "male",
            "language": "german",
            "name": "SamplePrime",
        },
    )
    assert save.status_code == 200
    assert save.json()["gender"] == "male"
    assert save.json()["language"] == "german"
    assert save.json()["name"] == "Sampleprime"
    assert save.json()["voice"] == "voices/sampleprime.wav"
    assert not voice_path.exists()
    assert (voices_dir / "sampleprime.wav").exists()

    listed = client.get("/api/voices")
    assert listed.status_code == 200
    local_entry = next(
        item
        for item in listed.json()["local"]
        if item["value"] == "voices/sampleprime.wav"
    )
    assert local_entry["gender"] == "male"
    assert local_entry["language"] == "german"
    assert local_entry["label"] == "Sampleprime"

    rename = client.post(
        "/api/voices/metadata",
        json={"voice": "voices/sampleprime.wav", "name": "Sample2"},
    )
    assert rename.status_code == 200
    rename_payload = rename.json()
    assert rename_payload["gender"] == "male"
    assert rename_payload["language"] == "german"
    assert rename_payload["name"] == "Sample2"
    assert rename_payload["voice"] == "voices/sample2.wav"
    assert not (voices_dir / "sampleprime.wav").exists()
    assert (voices_dir / "sample2.wav").exists()

    listed_renamed = client.get("/api/voices")
    local_entry_renamed = next(
        item
        for item in listed_renamed.json()["local"]
        if item["value"] == "voices/sample2.wav"
    )
    assert local_entry_renamed["gender"] == "male"
    assert local_entry_renamed["language"] == "german"
    assert local_entry_renamed["label"] == "Sample2"

    clear = client.post(
        "/api/voices/metadata",
        json={"voice": "voices/sample2.wav", "gender": None},
    )
    assert clear.status_code == 200
    assert clear.json()["gender"] is None
    assert clear.json()["language"] == "german"
    assert clear.json()["name"] == "Sample2"
    assert clear.json()["voice"] == "voices/sample2.wav"

    listed_after = client.get("/api/voices")
    local_entry_after = next(
        item
        for item in listed_after.json()["local"]
        if item["value"] == "voices/sample2.wav"
    )
    assert "gender" not in local_entry_after
    assert local_entry_after["language"] == "german"
    assert local_entry_after["label"] == "Sample2"

    clear_language = client.post(
        "/api/voices/metadata",
        json={"voice": "voices/sample2.wav", "language": None},
    )
    assert clear_language.status_code == 200
    assert clear_language.json()["language"] is None

    metadata = player._load_json(repo_root / "voices" / "metadata.json")
    assert "voices/sample2.wav" not in metadata


def test_voice_metadata_rejects_invalid_gender(tmp_path: Path) -> None:
    repo_root, root_dir = _make_repo(tmp_path)
    voices_dir = repo_root / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    (voices_dir / "sample.wav").write_bytes(b"RIFFFAKEWAVE")

    app = player.create_app(root_dir)
    client = TestClient(app)

    response = client.post(
        "/api/voices/metadata",
        json={"voice": "voices/sample.wav", "gender": "robot"},
    )
    assert response.status_code == 400
    assert "Gender must be" in response.json()["detail"]


def test_voice_metadata_normalizes_and_rejects_language(tmp_path: Path) -> None:
    repo_root, root_dir = _make_repo(tmp_path)
    voices_dir = repo_root / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    (voices_dir / "sample.wav").write_bytes(b"RIFFFAKEWAVE")

    app = player.create_app(root_dir)
    client = TestClient(app)

    iso = client.post(
        "/api/voices/metadata",
        json={"voice": "voices/sample.wav", "language": "de-DE"},
    )
    assert iso.status_code == 200
    assert iso.json()["language"] == "german"

    rejected = client.post(
        "/api/voices/metadata",
        json={"voice": "voices/sample.wav", "language": "klingon"},
    )
    assert rejected.status_code == 400
    assert "not supported by pocket-tts" in rejected.json()["detail"]


def test_voices_listing_exposes_language_options(tmp_path: Path) -> None:
    repo_root, root_dir = _make_repo(tmp_path)
    voices_dir = repo_root / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    (voices_dir / "hannah.wav").write_bytes(b"RIFFFAKEWAVE")
    (voices_dir / "kate.wav").write_bytes(b"RIFFFAKEWAVE")

    app = player.create_app(root_dir)
    client = TestClient(app)
    tagged = client.post(
        "/api/voices/metadata",
        json={"voice": "voices/hannah.wav", "gender": "female", "language": "german"},
    )
    assert tagged.status_code == 200

    payload = client.get("/api/voices").json()
    assert payload["default_language"] == "english"
    assert payload["languages"] == sorted(player.language_util.POCKET_TTS_LANGUAGES)
    assert payload["language_display_names"]["german"] == "German"

    hannah = next(
        item for item in payload["local"] if item["value"] == "voices/hannah.wav"
    )
    assert hannah["language"] == "german"
    # Untagged voices stay untagged; the player treats them as the default language.
    kate = next(item for item in payload["local"] if item["value"] == "voices/kate.wav")
    assert "language" not in kate


def test_voice_delete_removes_file_and_metadata(tmp_path: Path) -> None:
    repo_root, root_dir = _make_repo(tmp_path)
    voices_dir = repo_root / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    voice_path = voices_dir / "sample.wav"
    voice_path.write_bytes(b"RIFFFAKEWAVE")

    app = player.create_app(root_dir)
    client = TestClient(app)

    tagged = client.post(
        "/api/voices/metadata",
        json={"voice": "voices/sample.wav", "gender": "female"},
    )
    assert tagged.status_code == 200
    assert tagged.json()["voice"] == "voices/sample.wav"
    assert tagged.json()["gender"] == "female"

    deleted = client.post(
        "/api/voices/delete",
        json={"voice": "voices/sample.wav"},
    )
    assert deleted.status_code == 200
    assert deleted.json() == {"status": "deleted", "voice": "voices/sample.wav"}
    assert not voice_path.exists()

    listed = client.get("/api/voices")
    assert listed.status_code == 200
    assert all(item["value"] != "voices/sample.wav" for item in listed.json()["local"])

    metadata_path = repo_root / "voices" / "metadata.json"
    metadata = player._load_json(metadata_path)
    assert "voices/sample.wav" not in metadata

    missing = client.post(
        "/api/voices/delete",
        json={"voice": "voices/sample.wav"},
    )
    assert missing.status_code == 404


def _make_book(root_dir: Path, book_id: str) -> Path:
    book_dir = root_dir / book_id
    (book_dir / "clean").mkdir(parents=True, exist_ok=True)
    (book_dir / "clean" / "toc.json").write_text(
        '{"metadata": {"title": "Book"}, "chapters": []}\n',
        encoding="utf-8",
    )
    return book_dir


def test_book_details_includes_pause_multipliers(tmp_path: Path) -> None:
    repo_root, root_dir = _make_repo(tmp_path)
    book_dir = _make_book(root_dir, "book-a")
    tts_dir = book_dir / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "voice": "voices/ray.wav",
        "pad_ms": 300,
        "chapters": [
            {
                "id": "0001-intro",
                "title": "Intro",
                "chunk_spans": [[0, 10], [11, 20], [21, 30]],
                "pause_multipliers": [3, 1, 5],
                "durations_ms": [None, None, None],
            }
        ],
    }
    (tts_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    details = player._book_details(book_dir, repo_root)
    assert details["chapters"] == [
        {
            "id": "0001-intro",
            "title": "Intro",
            "chunk_spans": [[0, 10], [11, 20], [21, 30]],
            "pause_multipliers": [3, 1, 5],
            "chunk_count": 3,
        }
    ]


def test_reading_overrides_save_and_get(tmp_path: Path) -> None:
    _repo_root, root_dir = _make_repo(tmp_path)
    book_dir = _make_book(root_dir, "book-a")
    app = player.create_app(root_dir)
    client = TestClient(app)

    save = client.post(
        "/api/reading-overrides",
        json={
            "book_id": "book-a",
            "overrides": [
                {"base": "sutta", "reading": "soot-ta"},
                {"base": "re:sati", "reading": "sah-tee"},
                {"pattern": "satipatth?ana", "reading": "sah-tee-pat-ta-na"},
            ],
        },
    )
    assert save.status_code == 200
    payload = save.json()
    assert payload["status"] == "ok"
    assert payload["tts_cleared"] is False
    assert payload["overrides"] == [
        {"base": "sutta", "reading": "soot-ta"},
        {"pattern": "sati", "reading": "sah-tee"},
        {"pattern": "satipatth?ana", "reading": "sah-tee-pat-ta-na"},
    ]

    stored = player._load_json(book_dir / "reading-overrides.json")
    assert stored.get("global") == payload["overrides"]

    loaded = client.get("/api/reading-overrides", params={"book_id": "book-a"})
    assert loaded.status_code == 200
    assert loaded.json()["overrides"] == [
        {"base": "sutta", "reading": "soot-ta", "mode": "word", "case_sensitive": False},
        {"pattern": "sati", "reading": "sah-tee", "mode": "all", "case_sensitive": False},
        {
            "pattern": "satipatth?ana",
            "reading": "sah-tee-pat-ta-na",
            "mode": "all",
            "case_sensitive": False,
        },
    ]


def _make_book_with_language(root_dir: Path, book_id: str, lang_tag: str) -> Path:
    book_dir = root_dir / book_id
    (book_dir / "clean").mkdir(parents=True, exist_ok=True)
    (book_dir / "clean" / "toc.json").write_text(
        json.dumps(
            {
                "metadata": {"title": "Book", "language": lang_tag},
                "chapters": [],
            }
        ),
        encoding="utf-8",
    )
    return book_dir


def test_model_config_defaults_from_toc_language(tmp_path: Path) -> None:
    _repo_root, root_dir = _make_repo(tmp_path)
    book_dir = _make_book_with_language(root_dir, "book-de", "de-DE")
    app = player.create_app(root_dir)
    client = TestClient(app)

    resp = client.get("/api/books/book-de/model-config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["config"]["language"] == "german"
    assert body["config"]["layers"] == 24
    assert body["options"]["layers"] == [6, 24]
    assert "german" in body["options"]["languages"]
    assert body["options"]["language_display_names"]["german"] == "German"
    assert not (book_dir / "model-config.json").exists()


def test_model_config_persists_overrides(tmp_path: Path) -> None:
    _repo_root, root_dir = _make_repo(tmp_path)
    book_dir = _make_book_with_language(root_dir, "book-de", "de-DE")
    app = player.create_app(root_dir)
    client = TestClient(app)

    resp = client.post(
        "/api/books/book-de/model-config",
        json={"language": "german", "layers": 6},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["config"]["language"] == "german"
    assert body["config"]["layers"] == 6

    stored = json.loads((book_dir / "model-config.json").read_text(encoding="utf-8"))
    assert stored == {"language": "german", "layers": 6}

    reload = client.get("/api/books/book-de/model-config")
    assert reload.json()["config"] == body["config"]


def test_model_config_coerces_unsupported_layers(tmp_path: Path) -> None:
    _repo_root, root_dir = _make_repo(tmp_path)
    _make_book_with_language(root_dir, "book-en", "en")
    app = player.create_app(root_dir)
    client = TestClient(app)

    resp = client.post(
        "/api/books/book-en/model-config",
        json={"layers": 24},
    )
    assert resp.status_code == 200
    assert resp.json()["config"]["layers"] == 6


def test_model_config_hides_single_layer_for_french(tmp_path: Path) -> None:
    _repo_root, root_dir = _make_repo(tmp_path)
    _make_book_with_language(root_dir, "book-fr", "fr-CA")
    app = player.create_app(root_dir)
    client = TestClient(app)

    resp = client.get("/api/books/book-fr/model-config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["config"]["language"] == "french"
    assert body["config"]["layers"] == 24
    assert body["options"]["layers"] == [24]


def test_model_config_clears_cache_when_config_differs(tmp_path: Path) -> None:
    _repo_root, root_dir = _make_repo(tmp_path)
    book_dir = _make_book_with_language(root_dir, "book-de", "de-DE")
    tts_dir = book_dir / "tts"
    seg_dir = tts_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "dummy.wav").write_bytes(b"x")
    manifest_path = tts_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "voice": "voices/ray.wav",
                "language": "german",
                "layers": 24,
                "chapters": [],
            }
        ),
        encoding="utf-8",
    )
    app = player.create_app(root_dir)
    client = TestClient(app)

    resp = client.post(
        "/api/books/book-de/model-config",
        json={"language": "german", "layers": 6},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["cache_cleared"] is True
    assert not seg_dir.exists()
    assert not manifest_path.exists()


def test_model_config_preserves_cache_when_unchanged(tmp_path: Path) -> None:
    _repo_root, root_dir = _make_repo(tmp_path)
    book_dir = _make_book_with_language(root_dir, "book-de", "de-DE")
    tts_dir = book_dir / "tts"
    seg_dir = tts_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "dummy.wav").write_bytes(b"x")
    manifest_path = tts_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "voice": "voices/ray.wav",
                "language": "german",
                "layers": 24,
                "chapters": [],
            }
        ),
        encoding="utf-8",
    )
    app = player.create_app(root_dir)
    client = TestClient(app)

    resp = client.post(
        "/api/books/book-de/model-config",
        json={"language": "german", "layers": 24},
    )
    assert resp.status_code == 200
    assert resp.json()["cache_cleared"] is False
    assert seg_dir.exists()
    assert manifest_path.exists()


def test_book_details_surfaces_model_config_and_manifest_model(tmp_path: Path) -> None:
    repo_root, root_dir = _make_repo(tmp_path)
    book_dir = _make_book_with_language(root_dir, "book-de", "de-DE")
    tts_dir = book_dir / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "voice": "voices/ray.wav",
        "pad_ms": 300,
        "language": "german",
        "layers": 24,
        "chapters": [],
    }
    (tts_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    details = player._book_details(book_dir, repo_root)
    book = details["book"]
    assert book["language"] == "german"
    assert book["language_display"] == "German"
    assert book["model_config"]["language"] == "german"
    assert book["model_config"]["layers"] == 24
    assert book["manifest_model"] == {
        "language": "german",
        "layers": 24,
    }
    assert book["model_config_options"]["layers"] == [6, 24]


def test_read_log_tail_returns_last_nonempty_line(tmp_path: Path) -> None:
    log = tmp_path / "synth.log"
    log.write_text(
        "starting\nprogress 1\n\nValueError: voice cloning unsupported\n\n",
        encoding="utf-8",
    )
    assert player._read_log_tail(log) == "ValueError: voice cloning unsupported"


def test_read_log_tail_missing_file_and_long_logs(tmp_path: Path) -> None:
    assert player._read_log_tail(tmp_path / "missing.log") == ""
    log = tmp_path / "long.log"
    log.write_text("x" * 20000 + "\nfinal line\n", encoding="utf-8")
    assert player._read_log_tail(log) == "final line"
