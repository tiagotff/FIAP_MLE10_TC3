import json

import joblib
import numpy as np
import pytest

from triage.inference.loader import load_predictor
from triage.inference.policy import ThresholdPolicy
from triage.inference.predictor import (
    SklearnPredictor,
    column_order,
    to_predictions,
    token_coverage,
)

LABELS = ["normal", "atencao", "urgente"]


def test_column_order_maps_to_canonical_labels():
    assert column_order(["atencao", "normal", "urgente"], LABELS) == [1, 0, 2]


def test_column_order_rejects_mismatch():
    with pytest.raises(ValueError, match="diferem"):
        column_order(["a", "b"], LABELS)


def test_to_predictions_uses_chosen_index():
    probas = np.array([[0.2, 0.7, 0.1], [0.5, 0.1, 0.4]])
    result = to_predictions(probas, LABELS, np.array([1, 2]))
    assert [p.label for p in result] == ["atencao", "urgente"]
    assert result[1].confidence == pytest.approx(0.4)
    assert list(result[0].probabilities) == LABELS


def test_predictor_returns_canonical_order(trained_pipeline):
    predictor = SklearnPredictor(trained_pipeline, metadata={"labels": LABELS})
    [prediction] = predictor.predict(["cardiac arrest emergency"])
    assert list(prediction.probabilities) == LABELS
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)


def test_predictor_applies_injected_policy(trained_pipeline):
    predictor = SklearnPredictor(
        trained_pipeline, policy=ThresholdPolicy("urgente", 0.0)
    )
    labels = [p.label for p in predictor.predict(["gastric ulcer", "biopsy tumor"])]
    assert labels == ["urgente", "urgente"]


def test_predictor_version_from_metadata(trained_pipeline):
    predictor = SklearnPredictor(trained_pipeline, metadata={"model_version": "v9"})
    assert predictor.model_version == "v9"
    assert SklearnPredictor(trained_pipeline).model_version == "unknown"


def test_sklearn_predictor_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Modelo não encontrado"):
        SklearnPredictor.from_path(tmp_path / "model.joblib")


def test_load_predictor_builds_policy_from_metadata(project_settings, trained_pipeline):
    project_settings.models_dir.mkdir(parents=True)
    joblib.dump(trained_pipeline, project_settings.model_path)
    metadata = {
        "model_version": "20261101T000000Z",
        "labels": LABELS,
        "decision": {
            "policy": "threshold",
            "positive_label": "urgente",
            "threshold": 0.0,
        },
    }
    project_settings.metadata_path.write_text(json.dumps(metadata), "utf-8")
    predictor = load_predictor(project_settings)
    assert predictor.model_version == "20261101T000000Z"
    assert predictor.predict(["gastric ulcer"])[0].label == "urgente"


def test_load_predictor_unknown_backend(project_settings):
    with pytest.raises(ValueError, match="Backend 'tensorrt' desconhecido"):
        load_predictor(project_settings, backend="tensorrt")


def test_token_coverage():
    assert token_coverage(["a", "b", "c", "d"], {"a", "b"}) == 0.5
    assert token_coverage([], {"a"}) == 0.0


def test_vocabulary_coverage_drops_for_unknown_language(trained_pipeline):
    predictor = SklearnPredictor(trained_pipeline)
    english, portuguese, empty = predictor.vocabulary_coverage(
        ["acute stroke with hemiparesis", "acidente vascular com fraqueza", "the of"]
    )
    assert english == 1.0
    assert portuguese < 0.5
    assert empty == 0.0  # só stop words: nada a reconhecer


def test_predictor_requires_fitted_vectorizer(trained_pipeline):
    from sklearn.pipeline import Pipeline

    clf = trained_pipeline.named_steps["clf"]
    with pytest.raises(ValueError, match="vetorizador ajustado"):
        SklearnPredictor(Pipeline([("clf", clf)]))
