"""Etapas do pipeline de retreino, isoladas e baseadas em arquivos.

Cada função recebe as configurações e o identificador da execução, lê o
que a etapa anterior gravou em `artifacts/runs/<run_id>/` e devolve um
resumo pequeno e serializável (adequado para XCom no Airflow). Assim cada
etapa pode rodar em um processo separado e ser repetida individualmente.

Ordem: download -> prepare -> calibrate -> train -> evaluate ->
quality_gate -> publish.
"""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from triage.config.settings import Settings
from triage.data.dataset import concat_datasets
from triage.data.download import download_dataset
from triage.data.schema import DataConfig
from triage.data.split import DatasetSplits
from triage.inference.policy import build_policy
from triage.models.factory import ModelConfig
from triage.pipeline.gate import enforce_gate, evaluate_gate
from triage.pipeline.io import (
    SPLIT_NAMES,
    RunPaths,
    read_dataset,
    run_paths,
    write_dataset,
)
from triage.training.artifacts import load_metadata, save_json, save_model
from triage.training.calibration import build_decision, out_of_fold_probabilities
from triage.training.config import TrainingConfig
from triage.training.train import (
    build_metadata,
    cross_validation_metrics,
    fit,
    holdout_metrics,
    prepare_datasets,
    set_seeds,
)

logger = logging.getLogger(__name__)

_SUMMARY_KEYS = ("accuracy", "f1_macro", "recall_urgente", "precision_urgente")


@dataclass(frozen=True)
class ProjectConfigs:
    """Configurações declarativas usadas pelo pipeline."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def load(cls, settings: Settings) -> "ProjectConfigs":
        """Lê e valida os arquivos YAML apontados pelas configurações."""
        return cls(
            data=DataConfig.from_yaml(settings.data_config_path),
            model=ModelConfig.from_yaml(settings.model_config_path),
            training=TrainingConfig.from_yaml(settings.training_config_path),
        )


def _paths(settings: Settings, run_id: str) -> RunPaths:
    return run_paths(settings.artifacts_dir, run_id)


def _load_splits(paths: RunPaths) -> DatasetSplits:
    datasets = {name: read_dataset(paths.split_file(name)) for name in SPLIT_NAMES}
    return DatasetSplits(**datasets)


def _summary(metrics: dict[str, Any]) -> dict[str, float]:
    return {k: round(float(metrics[k]), 4) for k in _SUMMARY_KEYS if k in metrics}


def download_step(settings: Settings) -> dict[str, str]:
    """Baixa o corpus (idempotente: arquivos íntegros não são baixados de novo)."""
    config = DataConfig.from_yaml(settings.data_config_path)
    files = download_dataset(config, settings.data_raw_dir)
    return {split: str(path) for split, path in files.items()}


def prepare_step(settings: Settings, run_id: str) -> dict[str, Any]:
    """Deduplica, divide em treino/validação/teste e grava os conjuntos."""
    set_seeds(settings.random_seed)
    configs = ProjectConfigs.load(settings)
    paths = _paths(settings, run_id)
    splits, summary = prepare_datasets(settings, configs.data, configs.training)
    for name in SPLIT_NAMES:
        write_dataset(getattr(splits, name), paths.split_file(name))
    save_json(summary, paths.data_summary)
    return summary["sizes"]


def calibrate_step(settings: Settings, run_id: str) -> dict[str, Any]:
    """Calibra a regra de decisão com validação cruzada em treino + validação."""
    configs = ProjectConfigs.load(settings)
    paths = _paths(settings, run_id)
    splits = _load_splits(paths)
    development = concat_datasets([splits.train, splits.validation])
    oof = out_of_fold_probabilities(
        development,
        configs.model,
        configs.training.calibration.cv_folds,
        settings.random_seed,
    )
    policy, calibration = build_decision(oof, configs.training.decision)
    cv_metrics = cross_validation_metrics(oof, policy, configs.data.labels)
    decision = {**policy.describe(), "calibration": calibration}
    save_json({"decision": decision, "cross_validation": cv_metrics}, paths.decision)
    return decision


def train_step(settings: Settings, run_id: str) -> str:
    """Treina o modelo final em treino + validação."""
    set_seeds(settings.random_seed)
    configs = ProjectConfigs.load(settings)
    paths = _paths(settings, run_id)
    splits = _load_splits(paths)
    development = concat_datasets([splits.train, splits.validation])
    model = fit(development, configs.model, settings.random_seed)
    return str(save_model(model, paths.model))


def evaluate_step(settings: Settings, run_id: str) -> dict[str, float]:
    """Avalia o modelo treinado no teste, com argmax e com a política."""
    configs = ProjectConfigs.load(settings)
    paths = _paths(settings, run_id)
    model = joblib.load(paths.model)
    policy = build_policy(load_metadata(paths.decision)["decision"])
    test = read_dataset(paths.split_file("test"))
    metrics = holdout_metrics(model, test, policy, configs.data.labels)
    save_json(metrics, paths.metrics)
    return _summary(metrics["decision_policy"])


def quality_gate_step(settings: Settings, run_id: str) -> dict[str, Any]:
    """Aprova o modelo ou interrompe o pipeline (QualityGateError)."""
    configs = ProjectConfigs.load(settings)
    metrics = load_metadata(_paths(settings, run_id).metrics)["decision_policy"]
    minimums = configs.training.quality_gate.min_test_metrics
    checks = evaluate_gate(metrics, minimums)
    enforce_gate(checks)
    return {c.metric: {"value": c.value, "minimum": c.minimum} for c in checks}


def build_run_metadata(settings: Settings, run_id: str) -> dict[str, Any]:
    """Monta os metadados do modelo no mesmo formato do treino via CLI."""
    configs = ProjectConfigs.load(settings)
    paths = _paths(settings, run_id)
    decision_file = load_metadata(paths.decision)
    metrics = {
        "cross_validation": decision_file["cross_validation"],
        "test": load_metadata(paths.metrics),
    }
    metadata = build_metadata(
        configs.model,
        decision_file["decision"],
        metrics,
        load_metadata(paths.data_summary),
        configs.data.labels,
    )
    return {**metadata, "pipeline_run_id": run_id}


def _atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def publish_step(settings: Settings, run_id: str) -> str:
    """Publica o modelo aprovado: arquiva a versão e troca o modelo em uso.

    A troca é atômica (escrita em arquivo temporário + `os.replace`), então a
    API nunca lê um artefato pela metade.
    """
    paths = _paths(settings, run_id)
    metadata = build_run_metadata(settings, run_id)
    save_json(metadata, paths.metadata)
    archive = settings.models_dir / "archive" / metadata["model_version"]
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths.model, archive / settings.model_filename)
    shutil.copyfile(paths.metadata, archive / settings.metadata_filename)
    _atomic_copy(paths.model, settings.model_path)
    _atomic_copy(paths.metadata, settings.metadata_path)
    logger.info("Modelo %s publicado em %s", metadata["model_version"], archive)
    return str(metadata["model_version"])
