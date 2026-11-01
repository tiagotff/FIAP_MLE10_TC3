from pathlib import Path

import pytest

from triage.config.loader import load_yaml
from triage.config.settings import Settings


def test_settings_defaults_and_derived_paths():
    settings = Settings(_env_file=None)
    assert settings.model_path == Path("models/model.joblib")
    assert settings.metadata_path == Path("models/metadata.json")
    assert settings.inference_backend == "sklearn"


def test_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("MODELS_DIR", "/tmp/artefatos")
    monkeypatch.setenv("RANDOM_SEED", "7")
    settings = Settings(_env_file=None)
    assert settings.model_path == Path("/tmp/artefatos/model.joblib")
    assert settings.random_seed == 7


def test_settings_rejects_negative_seed():
    with pytest.raises(ValueError):
        Settings(random_seed=-1, _env_file=None)


def test_load_yaml_ok(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("a: 1\n", "utf-8")
    assert load_yaml(path) == {"a": 1}


def test_load_yaml_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="não encontrado"):
        load_yaml(tmp_path / "inexistente.yaml")


def test_load_yaml_requires_mapping(tmp_path):
    path = tmp_path / "lista.yaml"
    path.write_text("- 1\n- 2\n", "utf-8")
    with pytest.raises(ValueError, match="mapeamento"):
        load_yaml(path)
