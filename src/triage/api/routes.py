"""Endpoints HTTP do serviço de triagem."""

import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from triage.api.schemas import HealthResponse, LaudoRequest, PredictionResponse
from triage.inference.predictor import UrgencyPredictor

router = APIRouter()


def get_predictor(request: Request) -> UrgencyPredictor:
    """Recupera o preditor carregado na inicialização da aplicação.

    Raises:
        HTTPException: 503 se o modelo ainda não estiver disponível.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo indisponível no momento.",
        )
    return predictor


PredictorDep = Annotated[UrgencyPredictor, Depends(get_predictor)]


@router.get("/health", response_model=HealthResponse, tags=["infra"])
def health(request: Request) -> HealthResponse:
    """Verifica se o serviço está de pé e se o modelo foi carregado."""
    predictor = getattr(request.app.state, "predictor", None)
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        model_version=getattr(predictor, "model_version", "none"),
    )


@router.post("/predict", response_model=PredictionResponse, tags=["triagem"])
def predict(payload: LaudoRequest, predictor: PredictorDep) -> PredictionResponse:
    """Classifica a urgência de um laudo (normal / atencao / urgente)."""
    start = time.perf_counter()
    prediction = predictor.predict([payload.text])[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return PredictionResponse(
        label=prediction.label,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        model_version=predictor.model_version,
        inference_ms=round(elapsed_ms, 3),
    )
