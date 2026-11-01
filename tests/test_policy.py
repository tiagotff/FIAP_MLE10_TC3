import numpy as np
import pytest

from triage.inference.policy import (
    ArgmaxPolicy,
    ThresholdPolicy,
    build_policy,
)

CLASSES = ["normal", "atencao", "urgente"]
PROBAS = np.array(
    [
        [0.5, 0.2, 0.3],  # urgente abaixo de 0.35 -> maior entre as demais
        [0.3, 0.3, 0.4],  # urgente acima do limiar
        [0.2, 0.5, 0.3],  # urgente abaixo -> atencao
    ]
)


def test_argmax_policy():
    assert ArgmaxPolicy().decide(PROBAS, CLASSES).tolist() == [0, 2, 1]
    assert ArgmaxPolicy().describe() == {"policy": "argmax"}


def test_threshold_policy_prioritizes_positive_class():
    policy = ThresholdPolicy("urgente", 0.3)
    assert policy.decide(PROBAS, CLASSES).tolist() == [2, 2, 2]


def test_threshold_policy_falls_back_to_other_classes():
    policy = ThresholdPolicy("urgente", 0.35)
    assert policy.decide(PROBAS, CLASSES).tolist() == [0, 2, 1]


def test_threshold_policy_describe():
    assert ThresholdPolicy("urgente", 0.4).describe() == {
        "policy": "threshold",
        "positive_label": "urgente",
        "threshold": 0.4,
    }


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_threshold_policy_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="Limiar"):
        ThresholdPolicy("urgente", threshold)


def test_threshold_policy_unknown_label():
    with pytest.raises(ValueError, match="ausente"):
        ThresholdPolicy("critico", 0.5).decide(PROBAS, CLASSES)


def test_build_policy_defaults_to_argmax():
    assert isinstance(build_policy(None), ArgmaxPolicy)
    assert isinstance(build_policy({"policy": "argmax"}), ArgmaxPolicy)


def test_build_policy_threshold_from_metadata():
    policy = build_policy(
        {"policy": "threshold", "positive_label": "urgente", "threshold": 0.3}
    )
    assert isinstance(policy, ThresholdPolicy)
    assert policy.threshold == 0.3


def test_build_policy_errors():
    with pytest.raises(ValueError, match="desconhecida"):
        build_policy({"policy": "votacao"})
    with pytest.raises(ValueError, match="ausente"):
        build_policy({"policy": "threshold", "positive_label": "urgente"})
