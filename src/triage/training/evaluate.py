"""Métricas de avaliação do classificador de urgência."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from triage.data.dataset import TextDataset
from triage.inference.policy import DecisionPolicy


def labels_from_probas(
    probas: np.ndarray, classes: Sequence[str], policy: DecisionPolicy
) -> list[str]:
    """Converte probabilidades em rótulos segundo a política de decisão."""
    return [classes[i] for i in policy.decide(probas, classes)]


def predict_labels(
    model: Pipeline, texts: Sequence[str], policy: DecisionPolicy
) -> list[str]:
    """Aplica a política de decisão às probabilidades do modelo."""
    classes = [str(c) for c in model.classes_]
    return labels_from_probas(model.predict_proba(list(texts)), classes, policy)


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: list[str],
    positive_label: str,
) -> dict[str, Any]:
    """Métricas globais, por classe e matriz de confusão.

    O recall da classe positiva (`urgente`) é destacado por ser a métrica
    clinicamente crítica: um laudo urgente classificado como não urgente
    é o erro mais caro da triagem.
    """
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        f"recall_{positive_label}": float(report[positive_label]["recall"]),
        f"precision_{positive_label}": float(report[positive_label]["precision"]),
        "per_class": {label: report[label] for label in labels},
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
    }


def evaluate(
    model: Pipeline,
    dataset: TextDataset,
    policy: DecisionPolicy,
    labels: list[str],
    positive_label: str = "urgente",
) -> dict[str, Any]:
    """Avalia o modelo sob uma política de decisão."""
    predicted = predict_labels(model, dataset.texts, policy)
    return classification_metrics(dataset.labels, predicted, labels, positive_label)
