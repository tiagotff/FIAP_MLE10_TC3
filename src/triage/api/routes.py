"""Endpoints HTTP do serviço de triagem."""

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST

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
from triage.inference.predictor import CoverageAware, Prediction, UrgencyPredictor
from triage.monitoring.metrics import ApiMetrics

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


def get_metrics(request: Request) -> ApiMetrics:
    """Recupera os coletores de métricas criados com a aplicação."""
    return request.app.state.metrics


PredictorDep = Annotated[UrgencyPredictor, Depends(get_predictor)]
MetricsDep = Annotated[ApiMetrics, Depends(get_metrics)]


def timed_predict(
    predictor: UrgencyPredictor, texts: list[str]
) -> tuple[list[Prediction], float]:
    """Executa a predição e mede o tempo de inferência em ms."""
    start = time.perf_counter()
    predictions = predictor.predict(texts)
    return predictions, round((time.perf_counter() - start) * 1000, 3)


def monitored_predict(
    predictor: UrgencyPredictor,
    metrics: ApiMetrics,
    texts: list[str],
    endpoint: str,
) -> tuple[list[Prediction], float]:
    """Predição com registro de tempo, rótulos e sinais de drift.

    A cobertura de vocabulário é calculada fora do tempo de inferência
    reportado e só quando o backend oferece essa capacidade.
    """
    predictions, elapsed_ms = timed_predict(predictor, texts)
    metrics.observe_inference(endpoint, elapsed_ms / 1000, len(texts))
    metrics.observe_predictions(predictions)
    coverages = (
        predictor.vocabulary_coverage(texts)
        if isinstance(predictor, CoverageAware)
        else None
    )
    metrics.observe_inputs(texts, coverages)
    return predictions, elapsed_ms


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
    """Versão, backend, regra de decisão e métricas de teste do modelo em uso."""
    metadata = predictor.metadata
    return ModelInfoResponse(
        model_version=predictor.model_version,
        backend=predictor.backend,
        classifier=str(metadata.get("classifier", "unknown")),
        labels=list(metadata.get("labels", [])),
        decision={k: v for k, v in metadata.get("decision", {}).items()},
        test_metrics=test_metrics_summary(metadata),
    )


@router.get("/metrics", include_in_schema=False)
def metrics_endpoint(metrics: MetricsDep) -> Response:
    """Métricas no formato de exposição do Prometheus."""
    return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses=_UNAVAILABLE,
    tags=["triagem"],
)
def predict(
    payload: LaudoRequest, predictor: PredictorDep, metrics: MetricsDep
) -> PredictionResponse:
    """Classifica a urgência de um laudo (normal / atencao / urgente)."""
    [prediction], elapsed_ms = monitored_predict(
        predictor, metrics, [payload.text], "predict"
    )
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
def predict_batch(
    payload: BatchRequest, predictor: PredictorDep, metrics: MetricsDep
) -> BatchResponse:
    """Classifica um lote de laudos, preservando a ordem de entrada."""
    predictions, elapsed_ms = monitored_predict(
        predictor, metrics, payload.texts, "predict_batch"
    )
    return BatchResponse(
        predictions=[to_item(p) for p in predictions],
        model_version=predictor.model_version,
        inference_ms=elapsed_ms,
    )
