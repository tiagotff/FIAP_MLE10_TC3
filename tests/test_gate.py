import pytest

from triage.pipeline.gate import (
    GateCheck,
    QualityGateError,
    enforce_gate,
    evaluate_gate,
)


def test_gate_passes_and_fails():
    checks = evaluate_gate({"f1_macro": 0.8, "recall_urgente": 0.7}, {"f1_macro": 0.75})
    assert checks == [GateCheck("f1_macro", 0.8, 0.75)]
    enforce_gate(checks)
    failing = evaluate_gate({"recall_urgente": 0.7}, {"recall_urgente": 0.8})
    with pytest.raises(QualityGateError, match="recall_urgente=0.7000 < 0.8000"):
        enforce_gate(failing)


def test_gate_requires_known_metrics():
    with pytest.raises(KeyError, match="ausentes"):
        evaluate_gate({"f1_macro": 0.8}, {"auc": 0.9})
