from datetime import UTC, datetime

import pytest
import yaml

from triage.inference.loader import load_predictor
from triage.pipeline import steps
from triage.pipeline.gate import QualityGateError
from triage.pipeline.io import run_paths
from triage.pipeline.run import PIPELINE, default_run_id, run_pipeline
from triage.training.artifacts import load_metadata

RUN_ID = "scheduled__2026-11-01T03:00:00+00:00"


def test_default_run_id_format():
    moment = datetime(2026, 11, 1, 3, 0, 0, tzinfo=UTC)
    assert default_run_id(moment) == "manual__20261101T030000"


def test_pipeline_order_matches_dag_contract():
    assert [name for name, _ in PIPELINE] == [
        "prepare",
        "calibrate",
        "train",
        "evaluate",
        "quality_gate",
        "publish",
    ]


def _run_without_download(settings):
    results = {}
    for name, step in PIPELINE:
        results[name] = step(settings, RUN_ID)
    return results


def test_pipeline_publishes_model_loadable_by_api(project_settings):
    results = _run_without_download(project_settings)
    version = results["publish"]
    assert results["prepare"] == {"train": 24, "validation": 8, "test": 8}
    assert project_settings.model_path.exists()
    archive = project_settings.models_dir / "archive" / version
    assert (archive / "model.joblib").exists()
    metadata = load_metadata(project_settings.metadata_path)
    assert metadata["pipeline_run_id"] == RUN_ID
    assert set(metadata["metrics"]) == {"cross_validation", "test"}
    predictor = load_predictor(project_settings)
    assert predictor.model_version == version


def test_pipeline_matches_cli_training(project_settings):
    from triage.training.train import run_training

    cli_metadata = run_training(project_settings)
    _run_without_download(project_settings)
    pipeline_metadata = load_metadata(project_settings.metadata_path)
    assert pipeline_metadata["metrics"] == cli_metadata["metrics"]
    assert pipeline_metadata["decision"] == cli_metadata["decision"]


def test_quality_gate_blocks_publication(project_settings, training_config_dict):
    training_config_dict["quality_gate"] = {"min_test_metrics": {"f1_macro": 1.01}}
    project_settings.training_config_path.write_text(
        yaml.safe_dump(training_config_dict), "utf-8"
    )
    for _, step in PIPELINE[:4]:
        step(project_settings, RUN_ID)
    with pytest.raises(QualityGateError):
        steps.quality_gate_step(project_settings, RUN_ID)
    assert not project_settings.model_path.exists()


def test_steps_write_artifacts_in_run_directory(project_settings):
    for _, step in PIPELINE[:4]:
        step(project_settings, RUN_ID)
    paths = run_paths(project_settings.artifacts_dir, RUN_ID)
    assert paths.model.exists()
    assert paths.metrics.exists()
    assert "decision" in load_metadata(paths.decision)


def test_run_pipeline_includes_download(project_settings, monkeypatch):
    monkeypatch.setattr(steps, "download_step", lambda settings: {"train": "ok"})
    results = run_pipeline(project_settings, RUN_ID)
    assert results["download"] == {"train": "ok"}
    assert "publish" in results
