"""Quality gate: decide se um modelo retreinado pode ser publicado."""

from dataclasses import dataclass
from typing import Any


class QualityGateError(RuntimeError):
    """Modelo reprovado nos critérios mínimos de qualidade."""


@dataclass(frozen=True)
class GateCheck:
    """Resultado da comparação de uma métrica com seu mínimo."""

    metric: str
    value: float
    minimum: float

    @property
    def passed(self) -> bool:
        """Indica se a métrica atingiu o mínimo."""
        return self.value >= self.minimum


def evaluate_gate(
    metrics: dict[str, Any], minimums: dict[str, float]
) -> list[GateCheck]:
    """Compara cada métrica exigida com o mínimo configurado.

    Raises:
        KeyError: Se uma métrica exigida não existir nas métricas avaliadas.
    """
    missing = sorted(set(minimums) - set(metrics))
    if missing:
        raise KeyError(f"Métricas exigidas pelo quality gate ausentes: {missing}")
    return [
        GateCheck(metric=name, value=float(metrics[name]), minimum=minimum)
        for name, minimum in minimums.items()
    ]


def enforce_gate(checks: list[GateCheck]) -> None:
    """Interrompe o pipeline se alguma métrica ficou abaixo do mínimo.

    Raises:
        QualityGateError: Com a lista de métricas reprovadas.
    """
    failed = [c for c in checks if not c.passed]
    if failed:
        details = "; ".join(
            f"{c.metric}={c.value:.4f} < {c.minimum:.4f}" for c in failed
        )
        raise QualityGateError(f"Modelo reprovado no quality gate: {details}")
