import pytest
from fastapi.testclient import TestClient

from triage.api.main import create_app
from triage.api.schemas import MAX_BATCH_SIZE, MAX_TEXT_LENGTH


@pytest.fixture
def client(fake_predictor):
    with TestClient(create_app(lambda: fake_predictor)) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "test-version",
    }


def test_ready_with_model(client):
    assert client.get("/ready").json() == {
        "status": "ready",
        "model_version": "test-version",
    }


def test_model_info(client):
    body = client.get("/model").json()
    assert body["backend"] == "fake"
    assert body["classifier"] == "logistic_regression"
    assert body["decision"]["policy"] == "threshold"
    assert body["test_metrics"] == {"accuracy": 0.6, "f1_macro": 0.5}


def test_predict_returns_classification(client, fake_predictor):
    response = client.post("/predict", json={"text": "  dor torácica aguda  "})
    body = response.json()
    assert response.status_code == 200
    assert body["label"] == "urgente"
    assert body["model_version"] == "test-version"
    assert body["inference_ms"] >= 0
    assert fake_predictor.calls == [["dor torácica aguda"]]


def test_predict_batch_preserves_order(client, fake_predictor):
    response = client.post("/predict/batch", json={"texts": ["a", " b "]})
    body = response.json()
    assert response.status_code == 200
    assert len(body["predictions"]) == 2
    assert fake_predictor.calls == [["a", "b"]]


@pytest.mark.parametrize(
    "payload",
    [{"texts": []}, {"texts": ["ok", "  "]}, {"texts": ["x"] * (MAX_BATCH_SIZE + 1)}],
)
def test_predict_batch_rejects_invalid_payload(client, payload):
    assert client.post("/predict/batch", json=payload).status_code == 422


@pytest.mark.parametrize(
    "payload",
    [{"text": ""}, {"text": "   "}, {}, {"text": "x" * (MAX_TEXT_LENGTH + 1)}],
)
def test_predict_rejects_invalid_payload(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_endpoints_unavailable_without_model():
    with TestClient(create_app(lambda: None)) as test_client:
        assert test_client.get("/health").json()["model_loaded"] is False
        assert test_client.get("/ready").status_code == 503
        assert test_client.get("/model").status_code == 503
        response = test_client.post("/predict", json={"text": "laudo"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Modelo indisponível no momento."


def test_openapi_documents_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    expected = {"/health", "/ready", "/model", "/predict", "/predict/batch"}
    assert expected <= set(paths)
