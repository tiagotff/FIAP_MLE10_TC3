"""Probabilidades out-of-fold e calibração da regra de decisão.

O limiar da classe crítica é calibrado sobre probabilidades out-of-fold
(validação cruzada estratificada em treino + validação): cada texto é
pontuado por um modelo que não o viu no ajuste. Isso evita o viés de
calibrar na mesma validação usada pelo Optuna para escolher os
hiperparâmetros, que tende a produzir probabilidades otimistas.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from triage.data.dataset import TextDataset
from triage.inference.policy import ArgmaxPolicy, DecisionPolicy, ThresholdPolicy
from triage.models.factory import ModelConfig, build_pipeline
from triage.training.config import DecisionConfig
from triage.training.threshold import calibrate_threshold


@dataclass(frozen=True)
class OutOfFoldPredictions:
    """Probabilidades out-of-fold, ordem das classes e rótulos verdadeiros."""

    probas: np.ndarray
    classes: list[str]
    labels: list[str]


def out_of_fold_probabilities(
    dataset: TextDataset, config: ModelConfig, folds: int, seed: int
) -> OutOfFoldPredictions:
    """Gera probabilidades out-of-fold com validação cruzada estratificada."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probas = cross_val_predict(
        build_pipeline(config, seed),
        dataset.texts,
        dataset.labels,
        cv=splitter,
        method="predict_proba",
    )
    classes = sorted(set(dataset.labels))
    return OutOfFoldPredictions(probas=probas, classes=classes, labels=dataset.labels)


def build_decision(
    oof: OutOfFoldPredictions, config: DecisionConfig
) -> tuple[DecisionPolicy, dict[str, Any]]:
    """Cria a política de decisão, calibrando o limiar nas predições OOF."""
    if config.policy == "argmax":
        return ArgmaxPolicy(), {}
    scores = oof.probas[:, oof.classes.index(config.positive_label)]
    is_positive = np.asarray(oof.labels) == config.positive_label
    result = calibrate_threshold(is_positive, scores, config.target_recall)
    policy = ThresholdPolicy(config.positive_label, result.threshold)
    return policy, {"target_recall": config.target_recall, **result.to_dict()}
