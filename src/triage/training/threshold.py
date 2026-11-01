"""Calibração do limiar de decisão da classe crítica."""

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import precision_recall_curve


@dataclass(frozen=True)
class ThresholdResult:
    """Limiar escolhido e o desempenho que ele produz na validação."""

    threshold: float
    precision: float
    recall: float

    def to_dict(self) -> dict[str, float]:
        """Serializa o resultado com valores arredondados."""
        return {k: round(v, 6) for k, v in asdict(self).items()}


def calibrate_threshold(
    is_positive: np.ndarray, scores: np.ndarray, target_recall: float
) -> ThresholdResult:
    """Escolhe o maior limiar cujo recall atinge `target_recall`.

    Entre os limiares que atendem ao recall mínimo, o maior é o que
    preserva mais precisão (menos falsos alarmes).

    Args:
        is_positive: Vetor booleano indicando se a amostra é da classe positiva.
        scores: Probabilidade prevista da classe positiva.
        target_recall: Recall mínimo desejado, em (0, 1].

    Raises:
        ValueError: Se não houver amostras positivas na validação.
    """
    if not np.any(is_positive):
        raise ValueError("A validação não contém amostras da classe positiva.")
    precision, recall, thresholds = precision_recall_curve(is_positive, scores)
    eligible = np.flatnonzero(recall[:-1] >= target_recall)
    best = int(eligible[-1]) if eligible.size else 0
    return ThresholdResult(
        threshold=float(thresholds[best]),
        precision=float(precision[best]),
        recall=float(recall[best]),
    )
