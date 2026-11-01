"""Treino do classificador de urgência.

Uso:
    python -m triage.training.train

Protocolo (após a busca de hiperparâmetros, feita em `tune`):
  1. corpus deduplicado e dividido em treino/validação/teste (70/15/15);
  2. limiar de decisão calibrado com predições out-of-fold (validação
     cruzada estratificada) sobre treino + validação;
  3. modelo final ajustado em treino + validação;
  4. teste usado uma única vez, na avaliação final.

As funções são separadas por responsabilidade para poderem ser
reaproveitadas como tasks independentes de um orquestrador (Airflow).
"""

import logging
import random
from collections import Counter
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from triage.config.settings import Settings, get_settings
from triage.data.dataset import TextDataset, concat_datasets, load_corpus
from triage.data.dedup import deduplicate
from triage.data.schema import DataConfig
from triage.data.split import DatasetSplits, train_validation_test_split
from triage.inference.policy import ArgmaxPolicy, DecisionPolicy
from triage.models.factory import ModelConfig, build_pipeline
from triage.training.artifacts import build_version, save_json, save_model
from triage.training.calibration import (
    OutOfFoldPredictions,
    build_decision,
    out_of_fold_probabilities,
)
from triage.training.config import TrainingConfig
from triage.training.evaluate import (
    classification_metrics,
    evaluate,
    labels_from_probas,
)

logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    """Fixa as sementes aleatórias para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)


def prepare_datasets(
    settings: Settings, data_config: DataConfig, training_config: TrainingConfig
) -> tuple[DatasetSplits, dict[str, Any]]:
    """Carrega, deduplica e divide o corpus; retorna também o resumo dos dados."""
    corpus = load_corpus(settings.data_raw_dir, data_config)
    unique, report = deduplicate(corpus, data_config.labels)
    split = training_config.split
    splits = train_validation_test_split(
        unique, split.validation, split.test, settings.random_seed
    )
    summary = {
        "deduplication": report.to_dict(),
        "split_fractions": {
            "train": split.train,
            "validation": split.validation,
            "test": split.test,
        },
        "sizes": splits.sizes(),
        "class_distribution": class_distribution(splits),
    }
    return splits, summary


def class_distribution(splits: DatasetSplits) -> dict[str, dict[str, int]]:
    """Contagem de rótulos por conjunto (para conferir a estratificação)."""
    named = {"train": splits.train, "validation": splits.validation}
    named["test"] = splits.test
    return {name: dict(Counter(ds.labels)) for name, ds in named.items()}


def fit(dataset: TextDataset, config: ModelConfig, seed: int) -> Pipeline:
    """Treina o pipeline TF-IDF + classificador."""
    model = build_pipeline(config, seed)
    model.fit(dataset.texts, dataset.labels)
    return model


def evaluate_model(
    model: Pipeline,
    oof: OutOfFoldPredictions,
    policy: DecisionPolicy,
    test: TextDataset,
    labels: list[str],
) -> dict[str, Any]:
    """Métricas de validação cruzada (OOF) e de teste, nas duas políticas."""
    policies = {"argmax": ArgmaxPolicy(), "decision_policy": policy}
    return {
        "cross_validation": {
            name: classification_metrics(
                oof.labels,
                labels_from_probas(oof.probas, oof.classes, rule),
                labels,
                "urgente",
            )
            for name, rule in policies.items()
        },
        "test": {
            name: evaluate(model, test, rule, labels) for name, rule in policies.items()
        },
    }


def build_metadata(
    model_config: ModelConfig,
    decision: dict[str, Any],
    metrics: dict[str, Any],
    data_summary: dict[str, Any],
    labels: list[str],
) -> dict[str, Any]:
    """Monta o dicionário de metadados persistido ao lado do modelo."""
    return {
        "model_version": build_version(),
        "classifier": model_config.classifier.name,
        "model_config": model_config.model_dump(),
        "labels": labels,
        "decision": decision,
        "data": data_summary,
        "metrics": metrics,
    }


def run_training(settings: Settings) -> dict[str, Any]:
    """Executa o fluxo completo: dados -> limiar (OOF) -> treino -> teste."""
    set_seeds(settings.random_seed)
    data_config = DataConfig.from_yaml(settings.data_config_path)
    model_config = ModelConfig.from_yaml(settings.model_config_path)
    training_config = TrainingConfig.from_yaml(settings.training_config_path)
    splits, data_summary = prepare_datasets(settings, data_config, training_config)
    development = concat_datasets([splits.train, splits.validation])
    oof = out_of_fold_probabilities(
        development,
        model_config,
        training_config.calibration.cv_folds,
        settings.random_seed,
    )
    policy, calibration = build_decision(oof, training_config.decision)
    model = fit(development, model_config, settings.random_seed)
    metrics = evaluate_model(model, oof, policy, splits.test, data_config.labels)
    decision = {**policy.describe(), "calibration": calibration}
    metadata = build_metadata(
        model_config, decision, metrics, data_summary, data_config.labels
    )
    save_artifacts(model, metadata, settings)
    return metadata


def save_artifacts(
    model: Pipeline, metadata: dict[str, Any], settings: Settings
) -> None:
    """Persiste modelo, metadados e o relatório de métricas versionável."""
    save_model(model, settings.model_path)
    save_json(metadata, settings.metadata_path)
    save_json(metadata, settings.reports_dir / "model" / "metrics.json")


def _log_summary(metadata: dict[str, Any]) -> None:
    for name, metrics in metadata["metrics"]["test"].items():
        logger.info(
            "[teste/%s] acurácia=%.4f f1_macro=%.4f recall_urgente=%.4f",
            name,
            metrics["accuracy"],
            metrics["f1_macro"],
            metrics["recall_urgente"],
        )
    logger.info(
        "Modelo %s | decisão: %s", metadata["model_version"], metadata["decision"]
    )


def main() -> None:
    """Ponto de entrada de linha de comando."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    _log_summary(run_training(settings))


if __name__ == "__main__":
    main()
