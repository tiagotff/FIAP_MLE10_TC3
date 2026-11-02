import argparse

import pytest

from triage.benchmark.compare import (
    batches,
    benchmark,
    build_report,
    label_agreement,
    markdown_table,
    speedup,
)
from triage.benchmark.latency import summarize
from triage.inference.loader import load_predictor


def test_batches_keeps_order_and_remainder():
    assert batches(["a", "b", "c", "d", "e"], 2) == [["a", "b"], ["c", "d"], ["e"]]


def test_speedup_uses_p50_and_p99():
    baseline = summarize([2.0, 2.0, 2.0, 10.0])
    optimized = summarize([1.0, 1.0, 1.0, 5.0])
    assert speedup(baseline, optimized) == {"p50": 2.0, "p99": 2.0}


def test_label_agreement_counts_identical_labels(fake_predictor):
    from triage.inference.predictor import Prediction

    class Other:
        def predict(self, texts):
            labels = ["urgente", "normal"]
            return [
                Prediction(label=labels[i % 2], confidence=0.5, probabilities={})
                for i, _ in enumerate(texts)
            ]

    agreement = label_agreement({"a": fake_predictor, "b": Other()}, ["x", "y"])
    assert agreement == pytest.approx(0.5)


@pytest.fixture
def exported_settings(project_settings):
    from triage.optimization.export_onnx import run_export
    from triage.training.train import run_training

    run_training(project_settings)
    run_export(project_settings)
    return project_settings


def test_benchmark_and_report_for_both_backends(exported_settings):
    predictors = {
        name: load_predictor(exported_settings, name) for name in ("sklearn", "onnx")
    }
    texts = ["acute stroke hemiparesis", "gastric ulcer", "routine follow up"] * 4
    results = benchmark(predictors, texts, batch_size=4, rounds=2, warmup=2)
    assert results["onnx"]["single"].n == len(texts)
    assert results["sklearn"]["batch"].n == 4
    args = argparse.Namespace(requests=len(texts), batch_size=4)
    report = build_report(
        exported_settings,
        args,
        results,
        {"sklearn": 10.0, "onnx": 5.0},
        label_agreement(predictors, texts),
    )
    assert report["label_agreement"] == 1.0
    assert report["backends"]["onnx"]["artifact"] == "model.onnx"
    assert set(report["speedup_onnx_vs_sklearn"]) == {"single", "batch"}
    table = markdown_table(report)
    assert "| onnx |" in table
    assert "Concordância de rótulos: 100.00%" in table
