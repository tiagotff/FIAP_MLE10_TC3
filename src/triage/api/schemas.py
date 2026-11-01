"""Contratos de entrada e saída da API (Pydantic)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

UrgencyLabel = Literal["normal", "atencao", "urgente"]

MAX_TEXT_LENGTH = 20_000


class LaudoRequest(BaseModel):
    """Texto do laudo a ser triado."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": (
                    "Patient admitted with acute chest pain radiating to the left arm, "
                    "ST-segment elevation and elevated troponin levels."
                )
            }
        }
    )

    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("O texto do laudo não pode ser vazio.")
        return stripped


class PredictionResponse(BaseModel):
    """Classificação de urgência retornada pela API."""

    label: UrgencyLabel
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model_version: str
    inference_ms: float = Field(ge=0.0, description="Tempo de inferência do modelo")


class HealthResponse(BaseModel):
    """Estado do serviço."""

    status: Literal["ok"]
    model_loaded: bool
    model_version: str
