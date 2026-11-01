import logging

import pytest
from fastapi.testclient import TestClient

from triage.api.logging_config import configure_logging
from triage.api.main import create_app
from triage.api.middleware import resolve_request_id
from triage.api.schemas import MAX_TEXT_LENGTH


@pytest.fixture
def client(fake_predictor):
    with TestClient(create_app(lambda: fake_predictor)) as test_client:
        yield test_client


def test_validation_error_does_not_echo_input(client):
    secret = "x" * (MAX_TEXT_LENGTH + 1)
    response = client.post("/predict", json={"text": secret})
    body = response.json()
    assert body["detail"] == "Requisição inválida."
    assert body["errors"][0]["loc"] == ["body", "text"]
    assert secret not in response.text
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_request_id_is_propagated(client):
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_resolve_request_id_rejects_invalid_values():
    assert resolve_request_id("ok_1.2-3") == "ok_1.2-3"
    assert len(resolve_request_id("tem espaço")) == 32
    assert len(resolve_request_id("x" * 65)) == 32
    assert len(resolve_request_id(None)) == 32


def test_access_log_never_contains_text(client, caplog):
    with caplog.at_level(logging.INFO, logger="triage.access"):
        client.post("/predict", json={"text": "laudo sigiloso do paciente"})
    assert "POST /predict 200" in caplog.text
    assert "sigiloso" not in caplog.text


def test_unexpected_error_returns_generic_500(fake_predictor):
    def explode(texts):
        raise RuntimeError("detalhe interno")

    fake_predictor.predict = explode
    app = create_app(lambda: fake_predictor)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/predict", json={"text": "laudo"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Erro interno ao processar a requisição."
    assert "detalhe interno" not in response.text


def test_configure_logging_sets_level():
    configure_logging("warning")
    assert logging.getLogger().level == logging.WARNING
    configure_logging("INFO")


def test_configure_logging_rejects_invalid_level():
    with pytest.raises(ValueError, match="inválido"):
        configure_logging("verboso")
