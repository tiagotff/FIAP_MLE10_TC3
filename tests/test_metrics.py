import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from triage.api.main import create_app
from triage.inference.predictor import Prediction, SklearnPredictor
from triage.monitoring.metrics import ApiMetrics


def scrape(client: TestClient) -> dict[tuple[str, tuple], float]:
    """Lê /metrics e indexa as amostras por (nome, rótulos)."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
    }


def value(samples, name, **labels) -> float:
    return samples.get((name, tuple(sorted(labels.items()))), 0.0)


@pytest.fixture
def client(fake_predictor):
    with TestClient(create_app(lambda: fake_predictor)) as test_client:
        yield test_client


def test_metrics_exposes_model_version(client):
    samples = scrape(client)
    assert (
        value(
            samples,
            "triage_model_info",
            model_version="test-version",
            backend="fake",
        )
        == 1.0
    )


def test_predict_updates_request_and_prediction_metrics(client):
    client.post("/predict", json={"text": "dor torácica"})
    client.post("/predict/batch", json={"texts": ["a", "b", "c"]})
    samples = scrape(client)
    assert (
        value(
            samples,
            "triage_http_requests_total",
            method="POST",
            route="/predict",
            status="200",
        )
        == 1
    )
    assert value(samples, "triage_predictions_total", label="urgente") == 4
    assert (
        value(samples, "triage_inference_duration_seconds_count", endpoint="predict")
        == 1
    )
    assert value(samples, "triage_batch_size_sum", endpoint="predict_batch") == 3
    assert value(samples, "triage_input_text_length_chars_count") == 4


def test_validation_errors_are_counted_by_route_template(client):
    client.post("/predict", json={"text": ""})
    client.get("/nao-existe")
    samples = scrape(client)
    assert (
        value(
            samples,
            "triage_http_requests_total",
            method="POST",
            route="/predict",
            status="422",
        )
        == 1
    )
    assert (
        value(
            samples,
            "triage_http_requests_total",
            method="GET",
            route="unmatched",
            status="404",
        )
        == 1
    )


def test_metrics_endpoint_is_not_tracked(client):
    scrape(client)
    samples = scrape(client)
    assert not any(
        name == "triage_http_requests_total" and ("route", "/metrics") in labels
        for name, labels in samples
    )


def test_unhandled_error_is_counted_as_500(fake_predictor):
    def explode(texts):
        raise RuntimeError("falha")

    fake_predictor.predict = explode
    app = create_app(lambda: fake_predictor)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        test_client.post("/predict", json={"text": "laudo"})
        samples = scrape(test_client)
    assert (
        value(
            samples,
            "triage_http_requests_total",
            method="POST",
            route="/predict",
            status="500",
        )
        == 1
    )
    assert value(samples, "triage_http_requests_in_progress") == 0


def test_vocabulary_coverage_is_recorded_for_capable_backend(trained_pipeline):
    predictor = SklearnPredictor(trained_pipeline, metadata={"model_version": "v1"})
    with TestClient(create_app(lambda: predictor)) as test_client:
        test_client.post("/predict", json={"text": "acute stroke hemiparesis"})
        test_client.post("/predict", json={"text": "acidente vascular cerebral"})
        samples = scrape(test_client)
    assert value(samples, "triage_vocabulary_coverage_ratio_count") == 2
    # Um texto com cobertura total e outro sem termo conhecido algum.
    assert value(samples, "triage_vocabulary_coverage_ratio_sum") == 1.0
    assert value(samples, "triage_vocabulary_coverage_ratio_bucket", le="0.1") == 1


def test_set_model_replaces_previous_version():
    metrics = ApiMetrics.create()
    metrics.set_model("v1", "sklearn")
    metrics.set_model("v2", "onnx")
    exposed = metrics.render().decode()
    assert 'model_version="v2"' in exposed
    assert 'model_version="v1"' not in exposed


def test_observe_predictions_tracks_confidence():
    metrics = ApiMetrics.create()
    probs = {"normal": 0.7, "atencao": 0.2, "urgente": 0.1}
    metrics.observe_predictions(
        [Prediction(label="normal", confidence=0.7, probabilities=probs)]
    )
    exposed = metrics.render().decode()
    assert 'triage_prediction_confidence_bucket{label="normal",le="0.7"} 1.0' in (
        exposed
    )


def test_each_app_has_isolated_registry(fake_predictor):
    first = create_app(lambda: fake_predictor)
    second = create_app(lambda: fake_predictor)
    assert first.state.metrics.registry is not second.state.metrics.registry
