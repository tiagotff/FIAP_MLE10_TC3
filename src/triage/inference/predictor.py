"""Preditores de urgência (padrão Strategy).

A API depende apenas do protocolo `UrgencyPredictor`; o backend concreto
(scikit-learn hoje, ONNX Runtime na etapa de otimização) é escolhido na
inicialização, sem alterar os endpoints.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Prediction:
    """Resultado da classificação de um laudo."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class UrgencyPredictor(Protocol):
    """Contrato mínimo de qualquer backend de inferência."""

    model_version: str

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Classifica uma lista de textos."""
        ...


def to_predictions(probas: np.ndarray, classes: list[str]) -> list[Prediction]:
    """Converte a matriz de probabilidades em objetos `Prediction`."""
    results: list[Prediction] = []
    for row in probas:
        best = int(np.argmax(row))
        results.append(
            Prediction(
                label=classes[best],
                confidence=float(row[best]),
                probabilities={c: float(p) for c, p in zip(classes, row, strict=True)},
            )
        )
    return results


class SklearnPredictor:
    """Backend baseado no pipeline scikit-learn serializado com joblib."""

    def __init__(self, pipeline: Pipeline, model_version: str = "unknown") -> None:
        self._pipeline = pipeline
        self._classes = [str(c) for c in pipeline.classes_]
        self.model_version = model_version

    @classmethod
    def from_path(
        cls, path: Path, model_version: str = "unknown"
    ) -> "SklearnPredictor":
        """Carrega o pipeline a partir do disco.

        Raises:
            FileNotFoundError: Se o artefato do modelo não existir.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {path}. "
                "Execute: python -m triage.training.train"
            )
        return cls(joblib.load(path), model_version)

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Classifica uma lista de textos."""
        return to_predictions(self._pipeline.predict_proba(texts), self._classes)
