import hashlib
from pathlib import Path

import pandas as pd
import pytest

from triage.data.dataset import build_dataset, load_split, map_conditions
from triage.data.download import download_dataset, fetch_file, is_valid, sha256_of
from triage.data.schema import DataConfig, SourceFile


def test_map_conditions_ok(data_config):
    mapped = map_conditions(pd.Series([1, 3, 5]), data_config.urgency_mapping)
    assert mapped.tolist() == ["atencao", "urgente", "normal"]


def test_map_conditions_unknown_code(data_config):
    with pytest.raises(ValueError, match="sem mapeamento"):
        map_conditions(pd.Series([1, 9]), data_config.urgency_mapping)


def test_build_dataset_drops_blank_and_null(data_config):
    frame = pd.DataFrame(
        {
            "condition_label": [4, 5, 3, None],
            "medical_abstract": ["dor torácica", "   ", None, "x"],
        }
    )
    dataset = build_dataset(frame, data_config)
    assert dataset.texts == ["dor torácica"]
    assert dataset.labels == ["urgente"]
    assert len(dataset) == 1


def test_data_config_rejects_undeclared_label(data_config_dict):
    data_config_dict["urgency_mapping"][5] = "critico"
    with pytest.raises(ValueError, match="não declarados"):
        DataConfig.model_validate(data_config_dict)


def test_load_split_missing_file(tmp_path, data_config):
    with pytest.raises(FileNotFoundError, match="triage.data.download"):
        load_split(tmp_path, "train", data_config)


def test_load_split_reads_csv(tmp_path, data_config, toy_frame):
    toy_frame.to_csv(tmp_path / "train.csv", index=False)
    dataset = load_split(tmp_path, "train", data_config)
    assert len(dataset) == len(toy_frame)
    assert set(dataset.labels) == {"normal", "atencao", "urgente"}


def _remote_file(tmp_path, content: bytes) -> tuple[str, SourceFile]:
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "train.csv").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return remote.as_uri(), SourceFile(name="train.csv", sha256=digest)


def test_sha256_and_is_valid(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"laudo")
    expected = hashlib.sha256(b"laudo").hexdigest()
    assert sha256_of(path) == expected
    assert is_valid(path, expected)
    assert not is_valid(tmp_path / "nao_existe", expected)


def test_fetch_file_downloads_and_validates(tmp_path):
    base_url, source = _remote_file(tmp_path, b"a,b\n1,2\n")
    target = fetch_file(base_url, source, tmp_path / "raw")
    assert target.read_bytes() == b"a,b\n1,2\n"


def test_fetch_file_rejects_bad_checksum(tmp_path):
    base_url, source = _remote_file(tmp_path, b"conteudo")
    bad = SourceFile(name=source.name, sha256="f" * 64)
    with pytest.raises(ValueError, match="Checksum inválido"):
        fetch_file(base_url, bad, tmp_path / "raw")
    assert not (tmp_path / "raw" / "train.csv").exists()


def test_download_dataset_skips_valid_files(tmp_path, data_config_dict):
    base_url, source = _remote_file(tmp_path, b"x")
    data_config_dict["source"] = {
        "base_url": base_url,
        "files": {"train": source.model_dump()},
    }
    config = DataConfig.model_validate(data_config_dict)
    first = download_dataset(config, tmp_path / "raw")
    (tmp_path / "remote" / "train.csv").unlink()
    second = download_dataset(config, tmp_path / "raw")
    assert first == second


def test_project_data_config_is_valid():
    config = DataConfig.from_yaml(Path("configs/data.yaml"))
    assert set(config.urgency_mapping.values()) == set(config.labels)
    assert {"train", "test"} <= set(config.source.files)
