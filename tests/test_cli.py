from ptts import cli


def _entry(
    delta_ms: float,
    output_latency_ms: float | None = None,
    *,
    trigger: str = "gapless",
    preloaded: bool = True,
    playback_rate: float = 1.0,
    pad_ms: int = 280,
) -> dict:
    payload = {
        "trigger": trigger,
        "preloaded": preloaded,
        "playback_rate": playback_rate,
        "delta_ms": delta_ms,
        "pad_ms": pad_ms,
    }
    if output_latency_ms is not None:
        payload["output_latency_ms"] = output_latency_ms
    return payload


def test_boundary_report_requires_output_latency_coverage() -> None:
    entries = [
        _entry(20.0),
        _entry(24.0),
        _entry(30.0),
        _entry(35.0),
        _entry(40.0),
        _entry(45.0),
    ]
    payload = cli._boundary_report_payload(entries, min_samples=5)
    recommendation = payload["recommendation"]

    assert payload["samples_filtered"] == 6
    assert payload["samples_with_output_latency"] == 0
    assert recommendation["recommended_pad_adjust_ms"] is None
    assert recommendation["basis"] == "insufficient_data"
    assert "coverage" in recommendation["reason"].lower()


def test_boundary_report_recommends_pad_adjustment_with_latency_data() -> None:
    entries = [
        _entry(10.0, output_latency_ms=20.0),
        _entry(12.0, output_latency_ms=20.0),
        _entry(14.0, output_latency_ms=20.0),
        _entry(16.0, output_latency_ms=20.0),
        _entry(18.0, output_latency_ms=20.0),
        _entry(999.0, output_latency_ms=20.0, trigger="ended"),
    ]
    payload = cli._boundary_report_payload(entries, min_samples=5)
    recommendation = payload["recommendation"]

    assert payload["samples_filtered"] == 5
    assert payload["samples_with_output_latency"] == 5
    assert payload["output_latency_coverage"] == 1.0
    assert recommendation["basis"] == "p50(delta_ms + output_latency_ms)"
    assert recommendation["recommended_pad_adjust_ms"] == 34
