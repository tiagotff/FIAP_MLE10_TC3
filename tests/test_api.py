import pytest
from fastapi.testclient import TestClient

from triage.api.main import create_app
from triage.api.schemas import MAX_TEXT_LENGTH


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


def test_predict_returns_classification(client, fake_predictor):
    response = client.post("/predict", json={"text": "  dor torácica aguda  "})
    body = response.json()
    assert response.status_code == 200
    assert body["label"] == "urgente"
    assert body["model_version"] == "test-version"
    assert body["inference_ms"] >= 0
    assert fake_predictor.calls == [["dor torácica aguda"]]


def test_predict_sets_process_time_header(client):
    response = client.post("/predict", json={"text": "laudo"})
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


@pytest.mark.parametrize(
    "payload",
    [{"text": ""}, {"text": "   "}, {}, {"text": "x" * (MAX_TEXT_LENGTH + 1)}],
)
def test_predict_rejects_invalid_payload(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_unavailable_without_model():
    with TestClient(create_app(lambda: None)) as test_client:
        assert test_client.get("/health").json()["model_loaded"] is False
        response = test_client.post("/predict", json={"text": "laudo"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Modelo indisponível no momento."


def test_openapi_documents_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/predict"} <= set(paths)
