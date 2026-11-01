"""Regras de decisão: probabilidades -> rótulo (padrão Strategy).

Separar a regra de decisão do modelo permite ajustar o compromisso entre
recall e precisão da classe crítica sem retreinar: basta trocar a
política (ou o limiar) registrada nos metadados do modelo.
"""

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np


class DecisionPolicy(Protocol):
    """Contrato de uma regra de decisão sobre a matriz de probabilidades."""

    def decide(self, probas: np.ndarray, classes: Sequence[str]) -> np.ndarray:
        """Retorna o índice da classe escolhida para cada linha."""
        ...

    def describe(self) -> dict[str, Any]:
        """Descreve a política (para metadados e para a API)."""
        ...


class ArgmaxPolicy:
    """Escolhe a classe de maior probabilidade."""

    def decide(self, probas: np.ndarray, classes: Sequence[str]) -> np.ndarray:
        """Retorna o índice da maior probabilidade de cada linha."""
        return np.argmax(probas, axis=1)

    def describe(self) -> dict[str, Any]:
        """Descreve a política."""
        return {"policy": "argmax"}


class ThresholdPolicy:
    """Prioriza a classe positiva quando sua probabilidade atinge o limiar.

    Abaixo do limiar, escolhe a maior probabilidade entre as demais classes.
    """

    def __init__(self, positive_label: str, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Limiar deve estar entre 0 e 1: {threshold}")
        self.positive_label = positive_label
        self.threshold = threshold

    def decide(self, probas: np.ndarray, classes: Sequence[str]) -> np.ndarray:
        """Aplica o limiar à classe positiva e argmax às demais."""
        positive = self._positive_index(classes)
        others = probas.copy()
        others[:, positive] = -np.inf
        fallback = np.argmax(others, axis=1)
        return np.where(probas[:, positive] >= self.threshold, positive, fallback)

    def describe(self) -> dict[str, Any]:
        """Descreve a política e o limiar em uso."""
        return {
            "policy": "threshold",
            "positive_label": self.positive_label,
            "threshold": self.threshold,
        }

    def _positive_index(self, classes: Sequence[str]) -> int:
        if self.positive_label not in classes:
            raise ValueError(
                f"Classe positiva '{self.positive_label}' ausente no modelo: "
                f"{list(classes)}"
            )
        return list(classes).index(self.positive_label)


def _threshold_from_spec(spec: dict[str, Any]) -> DecisionPolicy:
    return ThresholdPolicy(spec["positive_label"], float(spec["threshold"]))


POLICY_REGISTRY: dict[str, Callable[[dict[str, Any]], DecisionPolicy]] = {
    "argmax": lambda _spec: ArgmaxPolicy(),
    "threshold": _threshold_from_spec,
}


def build_policy(spec: dict[str, Any] | None) -> DecisionPolicy:
    """Cria a política descrita em `spec` (padrão: argmax).

    Raises:
        ValueError: Se a política for desconhecida ou faltarem parâmetros.
    """
    spec = spec or {"policy": "argmax"}
    builder = POLICY_REGISTRY.get(spec.get("policy", ""))
    if builder is None:
        available = ", ".join(sorted(POLICY_REGISTRY))
        raise ValueError(f"Política '{spec.get('policy')}' desconhecida: {available}")
    try:
        return builder(spec)
    except KeyError as missing:
        raise ValueError(f"Parâmetro ausente na política: {missing}") from missing
