"""Modelo tipado do protocolo de treino (configs/training.yaml)."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from triage.config.loader import load_yaml


class SplitConfig(BaseModel):
    """Frações de validação e teste sobre o corpus deduplicado.

    O treino recebe o restante (ex.: 0.15 + 0.15 -> 70/15/15).
    """

    validation: float = Field(gt=0.0, lt=1.0)
    test: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _leaves_room_for_train(self) -> "SplitConfig":
        if self.validation + self.test >= 1.0:
            raise ValueError("Validação + teste devem somar menos que 1.")
        return self

    @property
    def train(self) -> float:
        """Fração destinada ao treino."""
        return round(1.0 - self.validation - self.test, 6)


class CalibrationConfig(BaseModel):
    """Validação cruzada usada para calibrar a regra de decisão."""

    cv_folds: int = Field(default=5, ge=2)


class DecisionConfig(BaseModel):
    """Regra de decisão aplicada às probabilidades do modelo."""

    policy: Literal["argmax", "threshold"] = "argmax"
    positive_label: str = "urgente"
    target_recall: float = Field(default=0.85, gt=0.0, le=1.0)


class TrainingConfig(BaseModel):
    """Configuração completa do protocolo de treino."""

    split: SplitConfig
    calibration: CalibrationConfig = CalibrationConfig()
    decision: DecisionConfig = DecisionConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        """Cria a configuração a partir de um arquivo YAML."""
        return cls.model_validate(load_yaml(path))
