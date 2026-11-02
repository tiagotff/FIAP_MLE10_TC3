import random

import pytest
from scripts.generate_traffic import (
    ENGLISH_SAMPLES,
    PORTUGUESE_SAMPLES,
    Stats,
    load_english_texts,
    parse_args,
    parse_plan,
    predicted_labels,
    scenario_factory,
)


def test_parse_plan():
    assert parse_plan("normal:120, drift:30") == [("normal", 120.0), ("drift", 30.0)]


@pytest.mark.parametrize("plan", ["normal", "normal:", "invasao:10"])
def test_parse_plan_rejects_invalid_phase(plan):
    with pytest.raises(ValueError):
        parse_plan(plan)


def test_drift_scenario_sends_portuguese_reports():
    next_request = scenario_factory("drift", ENGLISH_SAMPLES, batch_share=0.0)
    request = next_request(random.Random(0))
    assert request.path == "/predict"
    assert request.payload["text"] in PORTUGUESE_SAMPLES


def test_batch_share_routes_to_batch_endpoint():
    next_request = scenario_factory("normal", ENGLISH_SAMPLES, batch_share=1.0)
    request = next_request(random.Random(0))
    assert request.path == "/predict/batch"
    assert len(request.payload["texts"]) == 8


def test_errors_scenario_mixes_invalid_payloads():
    next_request = scenario_factory("errors", ENGLISH_SAMPLES, batch_share=0.0)
    rng = random.Random(1)
    payloads = [next_request(rng).payload for _ in range(200)]
    invalid = [p for p in payloads if p.get("text") not in ENGLISH_SAMPLES]
    assert 20 <= len(invalid) <= 60


def test_predicted_labels_for_single_and_batch():
    assert predicted_labels({"label": "urgente"}) == ["urgente"]
    batch = {"predictions": [{"label": "normal"}, {"label": "atencao"}]}
    assert predicted_labels(batch) == ["normal", "atencao"]
    assert predicted_labels({}) == []


def test_stats_summary_reports_p99():
    stats = Stats()
    for ms in range(1, 101):
        stats.record(200, float(ms), ["urgente"])
    summary = stats.summary()
    assert "Requisições: 100" in summary
    assert "p99=99.0 ms" in summary
    assert "{'urgente': 100}" in summary


def test_load_english_texts_falls_back_to_samples(tmp_path):
    assert load_english_texts(tmp_path / "ausente.csv") == ENGLISH_SAMPLES


def test_load_english_texts_reads_csv(tmp_path):
    path = tmp_path / "test.csv"
    path.write_text("medical_abstract\nlaudo um\nlaudo dois\n", "utf-8")
    assert load_english_texts(path) == ["laudo um", "laudo dois"]


def test_parse_args_rejects_non_positive_rate():
    with pytest.raises(SystemExit):
        parse_args(["--rps", "0"])
