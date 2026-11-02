import copy
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from triage.api.main import create_app
from triage.inference.loader import load_predictor
from triage.inference.onnx_predictor import OnnxPredictor, TextProfile, create_session
from triage.inference.policy import ThresholdPolicy
from triage.inference.predictor import SklearnPredictor
from triage.optimization.export_onnx import (
    OnnxExportError,
    check_exportable,
    convert,
    export_onnx,
    run_export,
    text_profile,
)

LABELS = ["normal", "atencao", "urgente"]
SAMPLES = [
    "acute myocardial infarction with chest pain",
    "Gastric ULCER endoscopy",
    "routine follow up",
    "acidente vascular cerebral isquêmico com hemiparesia",
    "café naïve x² under_score 3d",
    "",
]


@pytest.fixture
def onnx_predictor(trained_pipeline):
    return OnnxPredictor(create_session(convert(trained_pipeline)))


def test_onnx_matches_sklearn_probabilities(trained_pipeline, onnx_predictor):
    expected = trained_pipeline.predict_proba(SAMPLES)
    actual = onnx_predictor.predict_proba(SAMPLES)
    assert np.abs(expected - actual).max() < 1e-5


def test_onnx_predictions_match_sklearn_with_policy(trained_pipeline):
    policy = ThresholdPolicy(positive_label="urgente", threshold=0.4)
    metadata = {"labels": LABELS}
    sklearn = SklearnPredictor(trained_pipeline, policy, metadata)
    onnx = OnnxPredictor(create_session(convert(trained_pipeline)), policy, metadata)
    assert [p.label for p in onnx.predict(SAMPLES)] == [
        p.label for p in sklearn.predict(SAMPLES)
    ]
    assert list(onnx.predict(SAMPLES[:1])[0].probabilities) == LABELS


def test_onnx_coverage_matches_sklearn(trained_pipeline, onnx_predictor):
    sklearn = SklearnPredictor(trained_pipeline)
    assert onnx_predictor.vocabulary_coverage(SAMPLES) == pytest.approx(
        sklearn.vocabulary_coverage(SAMPLES)
    )


def test_text_profile_tokens_match_sklearn_analyzer(trained_pipeline):
    vectorizer = trained_pipeline.steps[0][1]
    profile = text_profile(vectorizer)
    analyzer = vectorizer.build_analyzer()
    for text in SAMPLES:
        assert profile.tokens(text) == analyzer(text)


def test_text_profile_json_roundtrip(trained_pipeline):
    profile = text_profile(trained_pipeline.steps[0][1])
    assert TextProfile.from_json(profile.to_json()) == profile


def test_graph_has_no_locale_dependent_operator(trained_pipeline):
    import onnx

    model = onnx.load_from_string(convert(trained_pipeline))
    assert "StringNormalizer" not in {node.op_type for node in model.graph.node}
    props = {p.key: p.value for p in model.metadata_props}
    assert set(json.loads(props["triage.classes"])) == set(LABELS)


@pytest.mark.parametrize(
    ("param", "value"),
    [("ngram_range", (1, 2)), ("strip_accents", "unicode"), ("lowercase", False)],
)
def test_check_exportable_rejects_inexact_options(trained_pipeline, param, value):
    pipeline = copy.deepcopy(trained_pipeline)
    pipeline.steps[0][1].set_params(**{param: value})
    with pytest.raises(OnnxExportError, match=param):
        check_exportable(pipeline)


def test_export_writes_file_after_parity(trained_pipeline, tmp_path):
    path = tmp_path / "model.onnx"
    report = export_onnx(trained_pipeline, path, SAMPLES)
    assert report.passed
    assert report.label_agreement == 1.0
    assert path.exists()


def test_export_does_not_write_when_parity_fails(trained_pipeline, tmp_path):
    path = tmp_path / "model.onnx"
    with pytest.raises(OnnxExportError, match="diverge"):
        export_onnx(trained_pipeline, path, SAMPLES, tolerance=-1.0)
    assert not path.exists()


def test_from_path_requires_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_onnx"):
        OnnxPredictor.from_path(tmp_path / "ausente.onnx")


def test_run_export_and_load_onnx_backend(project_settings):
    from triage.training.train import run_training

    run_training(project_settings)
    block = run_export(project_settings)
    assert block["parity"]["passed"]
    metadata = json.loads(project_settings.metadata_path.read_text("utf-8"))
    assert metadata["onnx"]["file"] == "model.onnx"

    onnx = load_predictor(project_settings, backend="onnx")
    sklearn = load_predictor(project_settings, backend="sklearn")
    assert onnx.backend == "onnx"
    assert onnx.model_version == sklearn.model_version
    texts = ["acute stroke hemiparesis", "gastric ulcer endoscopy"]
    assert [p.label for p in onnx.predict(texts)] == [
        p.label for p in sklearn.predict(texts)
    ]


def test_api_serves_onnx_backend(onnx_predictor):
    with TestClient(create_app(lambda: onnx_predictor)) as client:
        assert client.get("/model").json()["backend"] == "onnx"
        response = client.post("/predict", json={"text": "acute stroke hemiparesis"})
        assert response.status_code == 200
        metrics = client.get("/metrics").text
    assert 'backend="onnx"' in metrics
    assert "triage_vocabulary_coverage_ratio_count 1.0" in metrics
