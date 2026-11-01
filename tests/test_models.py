from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from triage.models.factory import (
    CLASSIFIER_REGISTRY,
    ClassifierConfig,
    ModelConfig,
    build_classifier,
    build_pipeline,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("logistic_regression", LogisticRegression),
        ("random_forest", RandomForestClassifier),
    ],
)
def test_build_classifier_from_registry(name, expected):
    clf = build_classifier(ClassifierConfig(name=name), seed=1)
    assert isinstance(clf, expected)
    assert clf.random_state == 1


def test_build_classifier_applies_params():
    config = ClassifierConfig(
        name="logistic_regression", params={"logistic_regression": {"C": 0.5}}
    )
    assert build_classifier(config, seed=0).C == 0.5


def test_build_classifier_unknown_name():
    with pytest.raises(ValueError, match="desconhecido"):
        build_classifier(ClassifierConfig(name="xgboost"), seed=0)


def test_registry_exposes_expected_models():
    assert {"logistic_regression", "random_forest"} <= set(CLASSIFIER_REGISTRY)


def test_build_pipeline_steps(model_config):
    pipeline = build_pipeline(model_config, seed=42)
    assert list(pipeline.named_steps) == ["tfidf", "clf"]
    assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
    assert pipeline.named_steps["tfidf"].ngram_range == (1, 1)


def test_pipeline_learns_toy_data(trained_pipeline):
    assert set(trained_pipeline.classes_) == {"normal", "atencao", "urgente"}
    assert trained_pipeline.predict(["acute stroke hemiparesis"])[0] == "urgente"


def test_project_model_config_is_valid():
    config = ModelConfig.from_yaml(Path("configs/model.yaml"))
    assert config.classifier.name in CLASSIFIER_REGISTRY
    assert config.classifier.name in config.classifier.params
