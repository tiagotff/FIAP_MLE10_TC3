"""Contratos de entrada e saída da API (Pydantic)."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

UrgencyLabel = Literal["normal", "atencao", "urgente"]

MAX_TEXT_LENGTH = 20_000
MAX_BATCH_SIZE = 64

LaudoText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LENGTH),
]

_EXAMPLE_TEXT = (
    "Patient admitted with acute chest pain radiating to the left arm, "
    "ST-segment elevation and elevated troponin levels."
)


class LaudoRequest(BaseModel):
    """Texto de um laudo a ser triado."""

    model_config = ConfigDict(json_schema_extra={"example": {"text": _EXAMPLE_TEXT}})

    text: LaudoText


class BatchRequest(BaseModel):
    """Lote de laudos a serem triados em uma única chamada."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"texts": [_EXAMPLE_TEXT, "Routine follow-up."]}}
    )

    texts: list[LaudoText] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class PredictionItem(BaseModel):
    """Classificação de urgência de um laudo."""

    label: UrgencyLabel
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class PredictionResponse(PredictionItem):
    """Resposta de `/predict`."""

    model_version: str
    inference_ms: float = Field(ge=0.0, description="Tempo de inferência do modelo")


class BatchResponse(BaseModel):
    """Resposta de `/predict/batch` (mesma ordem da requisição)."""

    predictions: list[PredictionItem]
    model_version: str
    inference_ms: float = Field(ge=0.0, description="Tempo de inferência do lote")


class HealthResponse(BaseModel):
    """Liveness: o processo está de pé."""

    status: Literal["ok"]
    model_loaded: bool
    model_version: str


class ReadinessResponse(BaseModel):
    """Readiness: o serviço pode receber tráfego."""

    status: Literal["ready"]
    model_version: str


class ModelInfoResponse(BaseModel):
    """Informações do modelo em produção."""

    model_version: str
    classifier: str
    labels: list[str]
    decision: dict[str, Any]
    test_metrics: dict[str, float]


class ErrorResponse(BaseModel):
    """Formato padrão de erro."""

    detail: str
    request_id: str | None = None
