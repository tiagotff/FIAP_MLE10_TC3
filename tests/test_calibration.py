import numpy as np
import pytest

from triage.data.dataset import build_dataset
from triage.inference.policy import ArgmaxPolicy, ThresholdPolicy
from triage.training.calibration import (
    OutOfFoldPredictions,
    build_decision,
    out_of_fold_probabilities,
)
from triage.training.config import DecisionConfig
from triage.training.evaluate import labels_from_probas

CLASSES = ["atencao", "normal", "urgente"]


def _oof() -> OutOfFoldPredictions:
    probas = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.2, 0.4, 0.4],
            [0.5, 0.2, 0.3],
            [0.1, 0.8, 0.1],
        ]
    )
    labels = ["urgente", "urgente", "atencao", "normal"]
    return OutOfFoldPredictions(probas=probas, classes=CLASSES, labels=labels)


def test_out_of_fold_probabilities_cover_every_sample(
    model_config, data_config, toy_frame
):
    dataset = build_dataset(toy_frame, data_config)
    oof = out_of_fold_probabilities(dataset, model_config, folds=3, seed=42)
    assert oof.probas.shape == (len(dataset), 3)
    assert oof.classes == CLASSES
    assert np.allclose(oof.probas.sum(axis=1), 1.0)


def test_build_decision_argmax_has_no_calibration():
    policy, calibration = build_decision(_oof(), DecisionConfig(policy="argmax"))
    assert isinstance(policy, ArgmaxPolicy)
    assert calibration == {}


def test_build_decision_threshold_reaches_target_recall():
    config = DecisionConfig(policy="threshold", target_recall=1.0)
    policy, calibration = build_decision(_oof(), config)
    assert isinstance(policy, ThresholdPolicy)
    assert policy.threshold == pytest.approx(0.4)
    assert calibration["recall"] == pytest.approx(1.0)
    assert calibration["target_recall"] == 1.0


def test_labels_from_probas_applies_policy():
    oof = _oof()
    assert labels_from_probas(oof.probas, oof.classes, ArgmaxPolicy()) == [
        "urgente",
        "normal",
        "atencao",
        "normal",
    ]
    policy = ThresholdPolicy("urgente", 0.4)
    assert labels_from_probas(oof.probas, oof.classes, policy)[1] == "urgente"
