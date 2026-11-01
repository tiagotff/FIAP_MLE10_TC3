import hashlib
from pathlib import Path

import pandas as pd
import pytest

from triage.data.dataset import (
    TextDataset,
    build_dataset,
    load_split,
    map_conditions,
)
from triage.data.download import download_dataset, fetch_file, is_valid, sha256_of
from triage.data.schema import DataConfig, SourceFile
from triage.data.split import stratified_split, train_validation_test_split


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


def test_stratified_split_is_stratified_and_reproducible():
    dataset = TextDataset([f"t{i}" for i in range(30)], ["a"] * 20 + ["b"] * 10)
    rest, part = stratified_split(dataset, 0.3, seed=1)
    again, _ = stratified_split(dataset, 0.3, seed=1)
    assert len(rest) + len(part) == 30
    assert part.labels.count("b") == 3
    assert rest.texts == again.texts


@pytest.mark.parametrize("size", [0.0, 1.0, 1.5])
def test_stratified_split_rejects_invalid_size(size):
    with pytest.raises(ValueError, match="entre 0 e 1"):
        stratified_split(TextDataset(["a", "b"], ["x", "y"]), size, seed=0)


def test_train_validation_test_split_70_15_15():
    labels = ["a"] * 60 + ["b"] * 40
    dataset = TextDataset([f"t{i}" for i in range(100)], labels)
    splits = train_validation_test_split(dataset, 0.15, 0.15, seed=7)
    assert splits.sizes() == {"train": 70, "validation": 15, "test": 15}
    all_texts = splits.train.texts + splits.validation.texts + splits.test.texts
    assert sorted(all_texts) == sorted(dataset.texts)
    assert splits.test.labels.count("b") == 6


def test_train_validation_test_split_requires_room_for_train():
    dataset = TextDataset(["a", "b"], ["x", "y"])
    with pytest.raises(ValueError, match="somar menos que 1"):
        train_validation_test_split(dataset, 0.5, 0.5, seed=0)


def test_load_corpus_concatenates_all_files(tmp_path, data_config, toy_frame):
    from triage.data.dataset import load_corpus

    toy_frame.to_csv(tmp_path / "train.csv", index=False)
    toy_frame.head(3).to_csv(tmp_path / "test.csv", index=False)
    corpus = load_corpus(tmp_path, data_config)
    assert len(corpus) == len(toy_frame) + 3
