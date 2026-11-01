from datetime import UTC, datetime

import joblib

from triage.training.artifacts import build_version, load_metadata, save_metadata
from triage.training.evaluate import evaluate
from triage.training.train import run_training


def test_build_version_format():
    moment = datetime(2026, 11, 1, 20, 15, 0, tzinfo=UTC)
    assert build_version(moment) == "20261101T201500Z"


def test_metadata_roundtrip(tmp_path):
    path = tmp_path / "sub" / "meta.json"
    save_metadata({"model_version": "v1", "nota": "laudo"}, path)
    assert load_metadata(path) == {"model_version": "v1", "nota": "laudo"}


def test_load_metadata_missing_returns_empty(tmp_path):
    assert load_metadata(tmp_path / "nada.json") == {}


def test_evaluate_returns_core_metrics(trained_pipeline, data_config, toy_frame):
    from triage.data.dataset import build_dataset

    dataset = build_dataset(toy_frame, data_config)
    metrics = evaluate(trained_pipeline, dataset, data_config.labels)
    assert {"accuracy", "f1_macro", "recall_urgente", "per_class"} <= set(metrics)
    assert 0.0 <= metrics["recall_urgente"] <= 1.0
    assert "urgente" in metrics["per_class"]


def test_run_training_creates_artifacts(project_settings):
    metadata = run_training(project_settings)
    assert project_settings.model_path.exists()
    assert project_settings.metadata_path.exists()
    assert metadata["classifier"] == "logistic_regression"
    assert metadata["dataset_sizes"] == {"train": 8, "test": 8}
    model = joblib.load(project_settings.model_path)
    assert model.predict(["acute myocardial infarction"])[0] == "urgente"
