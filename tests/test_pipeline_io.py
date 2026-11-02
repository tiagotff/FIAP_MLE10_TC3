import pytest

from triage.data.dataset import TextDataset
from triage.pipeline.io import read_dataset, run_paths, safe_run_id, write_dataset

RUN_ID = "scheduled__2026-11-01T03:00:00+00:00"


def test_safe_run_id_removes_unsafe_characters():
    assert safe_run_id(RUN_ID) == "scheduled__2026-11-01T03_00_00_00_00"
    with pytest.raises(ValueError, match="inválido"):
        safe_run_id(":::")


def test_dataset_roundtrip_preserves_unicode(tmp_path):
    dataset = TextDataset(["dor torácica", "AVC"], ["urgente", "urgente"])
    path = write_dataset(dataset, tmp_path / "splits" / "train.jsonl")
    assert read_dataset(path) == dataset


def test_read_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="etapa anterior"):
        read_dataset(tmp_path / "nada.jsonl")


def test_run_paths_layout(tmp_path):
    paths = run_paths(tmp_path, RUN_ID)
    assert paths.root.parent == tmp_path / "runs"
    assert paths.split_file("train").name == "train.jsonl"
    assert paths.model.name == "model.joblib"
