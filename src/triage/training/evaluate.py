"""Métricas de avaliação do classificador de urgência."""

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from triage.data.dataset import TextDataset

URGENT_LABEL = "urgente"


def evaluate(
    model: Pipeline, dataset: TextDataset, labels: list[str]
) -> dict[str, Any]:
    """Avalia o modelo e retorna métricas globais e por classe.

    O recall da classe `urgente` é reportado à parte por ser a métrica
    clinicamente crítica: um laudo urgente classificado como normal
    (falso negativo) é o erro mais caro da triagem.
    """
    predicted = model.predict(dataset.texts)
    return {
        "accuracy": float(accuracy_score(dataset.labels, predicted)),
        "f1_macro": float(f1_score(dataset.labels, predicted, average="macro")),
        "recall_urgente": float(
            recall_score(
                dataset.labels, predicted, labels=[URGENT_LABEL], average="macro"
            )
        ),
        "per_class": classification_report(
            dataset.labels, predicted, labels=labels, output_dict=True, zero_division=0
        ),
    }
