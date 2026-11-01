"""Fixtures compartilhadas.

Os imports de módulos do projeto ficam dentro das fixtures para que cada
arquivo de teste dependa apenas do que de fato utiliza.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

TOY_ROWS = [
    (4, "acute myocardial infarction with chest pain and troponin elevation"),
    (4, "cardiac arrest ventricular fibrillation emergency resuscitation"),
    (3, "acute ischemic stroke with hemiparesis and aphasia"),
    (3, "subarachnoid hemorrhage severe headache neurological deficit"),
    (1, "breast carcinoma tumor biopsy malignant neoplasm"),
    (2, "gastric ulcer dyspepsia endoscopy digestive tract"),
    (5, "routine follow up mild inflammation general condition"),
    (5, "general pathological condition benign findings stable"),
]


@pytest.fixture
def data_config_dict() -> dict:
    return {
        "source": {
            "base_url": "https://example.org",
            "files": {
                "train": {"name": "train.csv", "sha256": "0" * 64},
                "test": {"name": "test.csv", "sha256": "0" * 64},
            },
        },
        "columns": {"text": "medical_abstract", "condition": "condition_label"},
        "urgency_mapping": {
            1: "atencao",
            2: "atencao",
            3: "urgente",
            4: "urgente",
            5: "normal",
        },
        "labels": ["normal", "atencao", "urgente"],
    }


@pytest.fixture
def data_config(data_config_dict):
    from triage.data.schema import DataConfig

    return DataConfig.model_validate(data_config_dict)


@pytest.fixture
def model_config_dict() -> dict:
    return {
        "vectorizer": {"ngram_range": [1, 1], "min_df": 1},
        "classifier": {
            "name": "logistic_regression",
            "params": {"logistic_regression": {"C": 1.0}},
        },
    }


@pytest.fixture
def model_config(model_config_dict):
    from triage.models.factory import ModelConfig

    return ModelConfig.model_validate(model_config_dict)


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    return pd.DataFrame(TOY_ROWS, columns=["condition_label", "medical_abstract"])


@pytest.fixture
def trained_pipeline(model_config, toy_frame, data_config):
    from triage.data.dataset import build_dataset
    from triage.models.factory import build_pipeline

    dataset = build_dataset(toy_frame, data_config)
    pipeline = build_pipeline(model_config, seed=42)
    pipeline.fit(dataset.texts, dataset.labels)
    return pipeline


@pytest.fixture
def project_settings(tmp_path: Path, data_config_dict, model_config_dict, toy_frame):
    """Settings apontando para dados/configs temporários e isolados."""
    from triage.config.settings import Settings

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    toy_frame.to_csv(raw_dir / "train.csv", index=False)
    toy_frame.to_csv(raw_dir / "test.csv", index=False)
    data_path, model_path = tmp_path / "data.yaml", tmp_path / "model.yaml"
    data_path.write_text(yaml.safe_dump(data_config_dict), "utf-8")
    model_path.write_text(yaml.safe_dump(model_config_dict), "utf-8")
    return Settings(
        data_raw_dir=raw_dir,
        models_dir=tmp_path / "models",
        data_config_path=data_path,
        model_config_path=model_path,
        _env_file=None,
    )


@pytest.fixture
def fake_predictor():
    """Preditor determinístico que registra as chamadas recebidas."""
    from triage.inference.predictor import Prediction

    class FakePredictor:
        model_version = "test-version"

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def predict(self, texts: list[str]) -> list[Prediction]:
            self.calls.append(texts)
            probs = {"normal": 0.1, "atencao": 0.1, "urgente": 0.8}
            return [
                Prediction(label="urgente", confidence=0.8, probabilities=probs)
                for _ in texts
            ]

    return FakePredictor()
