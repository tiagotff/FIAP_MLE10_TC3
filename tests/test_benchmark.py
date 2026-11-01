import pytest

from triage.benchmark.latency import measure, summarize


def test_summarize_basic_statistics():
    summary = summarize([1.0, 2.0, 3.0, 4.0])
    assert summary.n == 4
    assert summary.mean_ms == pytest.approx(2.5)
    assert summary.min_ms == 1.0
    assert summary.max_ms == 4.0
    assert summary.p50_ms == pytest.approx(2.5)
    assert summary.throughput_rps == pytest.approx(400.0)


def test_summarize_percentiles_are_ordered():
    summary = summarize(list(range(1, 101)))
    assert summary.p50_ms <= summary.p95_ms <= summary.p99_ms <= summary.max_ms


def test_summarize_empty_raises():
    with pytest.raises(ValueError, match="ao menos uma"):
        summarize([])


def test_to_dict_rounds_values():
    data = summarize([1.23456]).to_dict()
    assert data["mean_ms"] == 1.235
    assert set(data) >= {"p95_ms", "p99_ms", "throughput_rps"}


def test_measure_calls_function_for_each_text():
    seen = []
    samples = measure(seen.append, ["a", "b", "c"])
    assert seen == ["a", "b", "c"]
    assert len(samples) == 3
    assert all(s >= 0 for s in samples)
