"""Treino do classificador de urgência.

Uso:
    python -m triage.training.train

As funções são separadas por responsabilidade (carregar, treinar,
avaliar, persistir) para poderem ser reaproveitadas como tasks
independentes de um orquestrador (ex.: DAG do Airflow).
"""

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from triage.config.settings import Settings, get_settings
from triage.data.dataset import TextDataset, load_split
from triage.data.schema import DataConfig
from triage.models.factory import ModelConfig, build_pipeline
from triage.training.artifacts import build_version, save_metadata, save_model
from triage.training.evaluate import evaluate

logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    """Fixa as sementes aleatórias para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)


def fit(dataset: TextDataset, config: ModelConfig, seed: int) -> Pipeline:
    """Treina o pipeline TF-IDF + classificador."""
    model = build_pipeline(config, seed)
    model.fit(dataset.texts, dataset.labels)
    return model


def build_metadata(
    config: ModelConfig, metrics: dict[str, Any], sizes: dict[str, int]
) -> dict[str, Any]:
    """Monta o dicionário de metadados persistido ao lado do modelo."""
    return {
        "model_version": build_version(),
        "classifier": config.classifier.name,
        "vectorizer": config.vectorizer,
        "dataset_sizes": sizes,
        "metrics": metrics,
    }


def run_training(settings: Settings) -> dict[str, Any]:
    """Executa o fluxo completo: dados -> treino -> avaliação -> artefatos."""
    set_seeds(settings.random_seed)
    data_config = DataConfig.from_yaml(settings.data_config_path)
    model_config = ModelConfig.from_yaml(settings.model_config_path)
    train = load_split(settings.data_raw_dir, "train", data_config)
    test = load_split(settings.data_raw_dir, "test", data_config)
    model = fit(train, model_config, settings.random_seed)
    metrics = evaluate(model, test, data_config.labels)
    sizes = {"train": len(train), "test": len(test)}
    metadata = build_metadata(model_config, metrics, sizes)
    save_model(model, settings.model_path)
    save_metadata(metadata, settings.metadata_path)
    return metadata


def _log_summary(metadata: dict[str, Any], model_path: Path) -> None:
    metrics = metadata["metrics"]
    logger.info(
        "Modelo %s salvo em %s | acurácia=%.4f f1_macro=%.4f recall_urgente=%.4f",
        metadata["model_version"],
        model_path,
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["recall_urgente"],
    )


def main() -> None:
    """Ponto de entrada de linha de comando."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    metadata = run_training(settings)
    _log_summary(metadata, settings.model_path)


if __name__ == "__main__":
    main()
