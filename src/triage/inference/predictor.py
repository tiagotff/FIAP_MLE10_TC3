"""Preditores de urgência (padrão Strategy).

A API depende apenas do protocolo `UrgencyPredictor`; o backend concreto
(scikit-learn ou ONNX Runtime, ver `onnx_predictor`) é escolhido na
inicialização, sem alterar os endpoints. A regra de decisão
(`DecisionPolicy`) é injetada, e não embutida no backend.
"""

from collections.abc import Callable, Container, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from triage.inference.policy import ArgmaxPolicy, DecisionPolicy


@dataclass(frozen=True)
class Prediction:
    """Resultado da classificação de um laudo."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class UrgencyPredictor(Protocol):
    """Contrato mínimo de qualquer backend de inferência."""

    backend: str
    model_version: str
    metadata: dict[str, Any]

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Classifica uma lista de textos."""
        ...


@runtime_checkable
class CoverageAware(Protocol):
    """Capacidade opcional: medir quanto do texto o modelo reconhece.

    Separada de `UrgencyPredictor` (segregação de interfaces): um backend
    que não exponha o vocabulário continua válido, apenas sem esse sinal.
    """

    def vocabulary_coverage(self, texts: list[str]) -> list[float]:
        """Fração dos termos de cada texto presentes no vocabulário (0 a 1)."""
        ...


def token_coverage(tokens: Sequence[str], vocabulary: Container[str]) -> float:
    """Fração de `tokens` presentes em `vocabulary` (0.0 se não houver tokens)."""
    if not tokens:
        return 0.0
    return sum(token in vocabulary for token in tokens) / len(tokens)


def column_order(classes: Sequence[str], labels: Sequence[str]) -> list[int]:
    """Índices que reordenam as colunas de `classes` na ordem de `labels`.

    Raises:
        ValueError: Se os conjuntos de classes divergirem.
    """
    if set(classes) != set(labels):
        raise ValueError(
            f"Classes do modelo {sorted(classes)} diferem das esperadas "
            f"{sorted(labels)}."
        )
    return [list(classes).index(label) for label in labels]


def to_predictions(
    probas: np.ndarray, labels: Sequence[str], chosen: np.ndarray
) -> list[Prediction]:
    """Monta as predições a partir das probabilidades e dos índices escolhidos."""
    return [
        Prediction(
            label=labels[idx],
            confidence=float(row[idx]),
            probabilities={c: float(p) for c, p in zip(labels, row, strict=True)},
        )
        for row, idx in zip(probas, chosen, strict=True)
    ]


class SklearnPredictor:
    """Backend baseado no pipeline scikit-learn serializado com joblib."""

    backend = "sklearn"

    def __init__(
        self,
        pipeline: Pipeline,
        policy: DecisionPolicy | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.metadata = metadata or {}
        self.model_version = str(self.metadata.get("model_version", "unknown"))
        classes = [str(c) for c in pipeline.classes_]
        self._labels = list(self.metadata.get("labels") or classes)
        self._order = column_order(classes, self._labels)
        self._pipeline = pipeline
        self._policy = policy or ArgmaxPolicy()
        self._analyzer, self._vocabulary = _text_vocabulary(pipeline)

    @classmethod
    def from_path(
        cls,
        path: Path,
        policy: DecisionPolicy | None = None,
        metadata: dict[str, Any] | None = None,
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
        return cls(joblib.load(path), policy, metadata)

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Classifica uma lista de textos (probabilidades na ordem canônica)."""
        probas = self._pipeline.predict_proba(texts)[:, self._order]
        chosen = self._policy.decide(probas, self._labels)
        return to_predictions(probas, self._labels, chosen)

    def vocabulary_coverage(self, texts: list[str]) -> list[float]:
        """Fração dos termos de cada texto que o TF-IDF reconhece.

        Usa o mesmo analisador do treino (minúsculas, tokenização e stop
        words). Valores baixos indicam texto fora do domínio — por exemplo,
        laudos em português —, um sinal de drift observável sem rótulos.
        """
        return [
            token_coverage(self._analyzer(text), self._vocabulary) for text in texts
        ]


def _text_vocabulary(
    pipeline: Pipeline,
) -> tuple[Callable[[str], list[str]], Container[str]]:
    """Analisador e vocabulário do vetorizador (primeira etapa do pipeline).

    Raises:
        ValueError: Se a primeira etapa não for um vetorizador de texto ajustado.
    """
    vectorizer = pipeline.steps[0][1]
    if not hasattr(vectorizer, "vocabulary_"):
        raise ValueError("A primeira etapa do pipeline não é um vetorizador ajustado.")
    return vectorizer.build_analyzer(), vectorizer.vocabulary_
