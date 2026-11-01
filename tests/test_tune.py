from pathlib import Path

import optuna
import pytest
import yaml

from triage.models.factory import ModelConfig
from triage.training.tune import (
    SearchSpace,
    TuningConfig,
    run_tuning,
    sample_model_config,
    suggest,
)


def _trial(params: dict) -> optuna.trial.FixedTrial:
    return optuna.trial.FixedTrial(params)


def test_suggest_categorical_int_and_float():
    trial = _trial({"cat": "b", "inteiro": 3, "real": 0.5, "log": 0.01})
    assert suggest(trial, "cat", ["a", "b"]) == "b"
    assert suggest(trial, "inteiro", {"low": 1, "high": 5}) == 3
    assert suggest(trial, "real", {"low": 0.0, "high": 1.0}) == 0.5
    assert suggest(trial, "log", {"low": 0.001, "high": 1.0, "log": True}) == 0.01


def test_sample_model_config_merges_fixed_and_search(tuning_config_dict):
    config = TuningConfig.model_validate(tuning_config_dict)
    trial = _trial(
        {
            "classifier": "logistic_regression",
            "tfidf__sublinear_tf": True,
            "logistic_regression__C": 1.0,
        }
    )
    model_config = sample_model_config(trial, config)
    assert model_config.vectorizer == {"min_df": 1, "sublinear_tf": True}
    assert model_config.classifier.params == {"logistic_regression": {"C": 1.0}}


def test_search_space_defaults():
    assert SearchSpace().fixed == {} and SearchSpace().search == {}


def test_run_tuning_writes_config_and_report(project_settings):
    study = run_tuning(project_settings)
    assert len(study.trials) == 3
    text = project_settings.model_config_path.read_text("utf-8")
    assert text.startswith("# Gerado por")
    ModelConfig.model_validate(yaml.safe_load(text))
    tuning_dir = project_settings.reports_dir / "tuning"
    assert (tuning_dir / "trials.csv").exists()
    assert (tuning_dir / "summary.json").exists()


@pytest.mark.parametrize("name", ["tuning"])
def test_project_tuning_config_is_valid(name):
    config = TuningConfig.from_yaml(Path(f"configs/{name}.yaml"))
    assert {"logistic_regression", "complement_nb"} <= set(config.classifiers)
    assert "strip_accents" not in config.vectorizer.fixed
