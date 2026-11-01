import joblib
import numpy as np
import pytest

from triage.inference.loader import load_predictor
from triage.inference.predictor import SklearnPredictor, to_predictions
from triage.training.artifacts import save_metadata


def test_to_predictions_picks_argmax():
    probas = np.array([[0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    result = to_predictions(probas, ["normal", "atencao", "urgente"])
    assert [p.label for p in result] == ["atencao", "urgente"]
    assert result[1].confidence == pytest.approx(0.8)
    assert result[0].probabilities["normal"] == pytest.approx(0.2)


def test_sklearn_predictor_probabilities_sum_to_one(trained_pipeline):
    predictor = SklearnPredictor(trained_pipeline, model_version="v1")
    [prediction] = predictor.predict(["cardiac arrest emergency"])
    assert prediction.label in {"normal", "atencao", "urgente"}
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
    assert predictor.model_version == "v1"


def test_sklearn_predictor_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Modelo não encontrado"):
        SklearnPredictor.from_path(tmp_path / "model.joblib")


def test_load_predictor_reads_version(project_settings, trained_pipeline):
    project_settings.models_dir.mkdir(parents=True)
    joblib.dump(trained_pipeline, project_settings.model_path)
    save_metadata({"model_version": "20261101T000000Z"}, project_settings.metadata_path)
    predictor = load_predictor(project_settings)
    assert predictor.model_version == "20261101T000000Z"


def test_load_predictor_unknown_backend(project_settings):
    with pytest.raises(ValueError, match="Backend 'tensorrt' desconhecido"):
        load_predictor(project_settings, backend="tensorrt")
