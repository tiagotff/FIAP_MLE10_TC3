"""Endpoints HTTP do serviço de triagem."""

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from triage.api.schemas import (
    BatchRequest,
    BatchResponse,
    ErrorResponse,
    HealthResponse,
    LaudoRequest,
    ModelInfoResponse,
    PredictionItem,
    PredictionResponse,
    ReadinessResponse,
)
from triage.inference.predictor import Prediction, UrgencyPredictor

router = APIRouter()

_UNAVAILABLE = {503: {"model": ErrorResponse, "description": "Modelo indisponível"}}
_SUMMARY_KEYS = ("accuracy", "f1_macro", "recall_urgente", "precision_urgente")


def get_predictor(request: Request) -> UrgencyPredictor:
    """Recupera o preditor carregado na inicialização da aplicação.

    Raises:
        HTTPException: 503 se o modelo não estiver disponível.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo indisponível no momento.",
        )
    return predictor


PredictorDep = Annotated[UrgencyPredictor, Depends(get_predictor)]


def timed_predict(
    predictor: UrgencyPredictor, texts: list[str]
) -> tuple[list[Prediction], float]:
    """Executa a predição e mede o tempo de inferência em ms."""
    start = time.perf_counter()
    predictions = predictor.predict(texts)
    return predictions, round((time.perf_counter() - start) * 1000, 3)


def to_item(prediction: Prediction) -> PredictionItem:
    """Converte a predição de domínio no schema de resposta."""
    return PredictionItem(
        label=prediction.label,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
    )


def test_metrics_summary(metadata: dict[str, Any]) -> dict[str, float]:
    """Extrai as métricas de teste da política de decisão em uso."""
    test = metadata.get("metrics", {}).get("test", {}).get("decision_policy", {})
    return {k: round(float(test[k]), 4) for k in _SUMMARY_KEYS if k in test}


@router.get("/health", response_model=HealthResponse, tags=["infra"])
def health(request: Request) -> HealthResponse:
    """Liveness: indica que o processo responde."""
    predictor = getattr(request.app.state, "predictor", None)
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        model_version=getattr(predictor, "model_version", "none"),
    )


@router.get(
    "/ready", response_model=ReadinessResponse, responses=_UNAVAILABLE, tags=["infra"]
)
def ready(predictor: PredictorDep) -> ReadinessResponse:
    """Readiness: 200 somente com o modelo carregado."""
    return ReadinessResponse(status="ready", model_version=predictor.model_version)


@router.get(
    "/model", response_model=ModelInfoResponse, responses=_UNAVAILABLE, tags=["infra"]
)
def model_info(predictor: PredictorDep) -> ModelInfoResponse:
    """Versão, regra de decisão e métricas de teste do modelo em uso."""
    metadata = predictor.metadata
    return ModelInfoResponse(
        model_version=predictor.model_version,
        classifier=str(metadata.get("classifier", "unknown")),
        labels=list(metadata.get("labels", [])),
        decision={k: v for k, v in metadata.get("decision", {}).items()},
        test_metrics=test_metrics_summary(metadata),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses=_UNAVAILABLE,
    tags=["triagem"],
)
def predict(payload: LaudoRequest, predictor: PredictorDep) -> PredictionResponse:
    """Classifica a urgência de um laudo (normal / atencao / urgente)."""
    [prediction], elapsed_ms = timed_predict(predictor, [payload.text])
    return PredictionResponse(
        **to_item(prediction).model_dump(),
        model_version=predictor.model_version,
        inference_ms=elapsed_ms,
    )


@router.post(
    "/predict/batch",
    response_model=BatchResponse,
    responses=_UNAVAILABLE,
    tags=["triagem"],
)
def predict_batch(payload: BatchRequest, predictor: PredictorDep) -> BatchResponse:
    """Classifica um lote de laudos, preservando a ordem de entrada."""
    predictions, elapsed_ms = timed_predict(predictor, payload.texts)
    return BatchResponse(
        predictions=[to_item(p) for p in predictions],
        model_version=predictor.model_version,
        inference_ms=elapsed_ms,
    )
