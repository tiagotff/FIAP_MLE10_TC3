import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pytest

from triage.data.dataset import build_dataset
from triage.inference.policy import ArgmaxPolicy, ThresholdPolicy
from triage.training.artifacts import build_version, load_metadata, save_json
from triage.training.config import TrainingConfig
from triage.training.evaluate import classification_metrics, evaluate, predict_labels
from triage.training.threshold import calibrate_threshold
from triage.training.train import run_training

LABELS = ["normal", "atencao", "urgente"]


def test_build_version_format():
    moment = datetime(2026, 11, 1, 20, 15, 0, tzinfo=UTC)
    assert build_version(moment) == "20261101T201500Z"


def test_json_roundtrip_with_trailing_newline(tmp_path):
    path = tmp_path / "sub" / "meta.json"
    save_json({"model_version": "v1", "nota": "laudo"}, path)
    assert load_metadata(path) == {"model_version": "v1", "nota": "laudo"}
    assert path.read_text("utf-8").endswith("\n")


def test_load_metadata_missing_returns_empty(tmp_path):
    assert load_metadata(tmp_path / "nada.json") == {}


def test_calibrate_threshold_meets_target_recall():
    is_positive = np.array([True, True, True, True, False, False, False, False])
    scores = np.array([0.9, 0.8, 0.6, 0.3, 0.7, 0.4, 0.2, 0.1])
    result = calibrate_threshold(is_positive, scores, target_recall=0.75)
    assert result.recall >= 0.75
    assert result.threshold == pytest.approx(0.6)
    assert set(result.to_dict()) == {"threshold", "precision", "recall"}


def test_calibrate_threshold_requires_positives():
    with pytest.raises(ValueError, match="classe positiva"):
        calibrate_threshold(np.array([False, False]), np.array([0.1, 0.2]), 0.8)


def test_classification_metrics_structure():
    y_true = ["urgente", "normal", "atencao", "urgente"]
    y_pred = ["urgente", "urgente", "atencao", "normal"]
    metrics = classification_metrics(y_true, y_pred, LABELS, "urgente")
    assert metrics["recall_urgente"] == pytest.approx(0.5)
    assert metrics["precision_urgente"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"]["labels"] == LABELS
    assert metrics["confusion_matrix"]["matrix"][2] == [1, 0, 1]


def test_predict_labels_applies_policy(trained_pipeline):
    texts = ["gastric ulcer", "biopsy tumor"]
    assert predict_labels(trained_pipeline, texts, ThresholdPolicy("urgente", 0.0)) == [
        "urgente",
        "urgente",
    ]


def test_evaluate_returns_core_metrics(trained_pipeline, data_config, toy_frame):
    dataset = build_dataset(toy_frame, data_config)
    metrics = evaluate(trained_pipeline, dataset, ArgmaxPolicy(), LABELS)
    assert {"accuracy", "f1_macro", "recall_urgente", "per_class"} <= set(metrics)


def test_training_config_validation(training_config_dict):
    config = TrainingConfig.model_validate(training_config_dict)
    assert config.decision.positive_label == "urgente"
    assert config.split.train == pytest.approx(0.6)
    assert config.calibration.cv_folds == 3
    training_config_dict["split"] = {"validation": 0.5, "test": 0.5}
    with pytest.raises(ValueError, match="somar menos que 1"):
        TrainingConfig.model_validate(training_config_dict)


def test_run_training_creates_artifacts(project_settings):
    metadata = run_training(project_settings)
    assert project_settings.model_path.exists()
    assert metadata["decision"]["policy"] == "threshold"
    assert set(metadata["metrics"]) == {"cross_validation", "test"}
    assert metadata["decision"]["calibration"]["target_recall"] == 0.8
    assert set(metadata["metrics"]["test"]) == {"argmax", "decision_policy"}
    data = metadata["data"]
    assert data["deduplication"]["input_rows"] == 80
    assert data["deduplication"]["unique_texts"] == 40
    assert sum(data["sizes"].values()) == 40
    assert set(data["class_distribution"]) == {"train", "validation", "test"}
    report = project_settings.reports_dir / "model" / "metrics.json"
    assert json.loads(report.read_text("utf-8"))["labels"] == LABELS
    assert joblib.load(project_settings.model_path).predict(["acute stroke"])[0]


def test_project_training_config_is_valid():
    from pathlib import Path

    config = TrainingConfig.from_yaml(Path("configs/training.yaml"))
    assert config.decision.policy == "threshold"
    assert config.split.train == pytest.approx(0.70)
    assert config.calibration.cv_folds == 5
